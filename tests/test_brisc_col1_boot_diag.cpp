// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic for the Blackhole "column-1 multi-core boot" issue.
//
// Symptom observed across all 4 p150b boards (HEAD as of 2026-05-17):
// BRISC firmware loaded onto Tensix tiles in TRANSLATED column x=1, rows
// y>=3 (= LOGICAL (0, y>=1)) never advances past RUN_MSG_INIT. Every
// other column (x>=2) and the (1,2) tile boot fine. The pattern is
// identical on all 4 boards, persists across `tt-smi -r N`, predates
// tt-foil v8 changes (v6-4 commit fails too), and uses unchanged
// tt-metal source. Confirmed `Harvesting Tensix=0x0` (no harvest) and
// `ENABLED_TENSIX_COL=0x3fff` (all 14 cols ON). L1 RW + assert/deassert
// reset both succeed at the failing cores.
//
// Most likely root cause: stale chip ARC/SPI state that survives the
// soft `tt-smi -r` reset (the host had 51 days of uptime at observation
// time). Recovery: full host reboot.
//
// What the test does, per probe core:
//   1. Reset Tensix RISCs, zero L1, reload firmware, init mailboxes.
//   2. Deassert BRISC reset.
//   3. Sample GO_MSG.signal at 1/50/500 ms intervals.
//
// Expected output post-reboot: every probed core reaches
// `signal=0x00 (DONE)` within ~1 ms. If column 1 rows >=3 still stick
// at 0x40 (INIT), the chip-state explanation is wrong and a deeper
// regression is at play — start by re-bisecting against an older
// tt-metal HEAD and the SPI flash bundle version on the chip.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "device.hpp"
#include "reset.hpp"
#include "firmware_load.hpp"
#include "firmware_paths.hpp"
#include "mailbox_init.hpp"
#include "core_info_init.hpp"
#include "bank_tables_init.hpp"
#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/risc_type.hpp>

namespace {
using namespace tt::tt_metal;

struct LCore { uint32_t x, y; };

void prep_core(tt::umd::Cluster& driver, const Hal& hal, uint32_t chip,
               const tt::umd::CoreCoord& core, uint32_t lx, uint32_t ly,
               const tt::foil::FirmwarePaths& fw) {
    using FR = tt::foil::FwRisc;
    tt::foil::assert_tensix_reset(driver, chip, core);

    // Zero full L1 to match tt-metal's clear_l1_state.
    {
        const auto& soc_desc = driver.get_soc_descriptor(chip);
        const uint64_t l1_size = soc_desc.worker_l1_size;
        std::vector<uint8_t> zero_l1(l1_size, 0);
        driver.write_to_device(zero_l1.data(), zero_l1.size(), chip, core, 0);
        driver.l1_membar(chip);
    }

    tt::foil::load_tensix_firmware(driver, hal, chip, core, tt::foil::firmware_elf(fw, FR::BRISC),  tt::foil::kBrisc);

    // Write the JAL trampoline at L1[0] so BRISC, which has reset PC=0
    // hardcoded (no reset-PC register on BH), jumps into firmware at
    // MEM_BRISC_FIRMWARE_BASE = 0x38E0 instead of hitting an illegal
    // instruction at L1[0] (which we zeroed in the clear_l1 step).
    // Encoding from generate_risc_startup_addr(0x38E0):
    //   (0x38E0 & 0x7fe) << 20  = 0x0E000000
    //   (0x38E0 & 0x800) << 9   = 0x00100000
    //   (0x38E0 & 0xff000)      = 0x00003000
    //   opcode                  = 0x0000006f
    //   total                   = 0x0E10306F
    {
        uint32_t jal = 0x0e10306fu;
        driver.write_to_device(&jal, sizeof(jal), chip, core, /*addr=*/0);
        driver.l1_membar(chip);
    }

    tt::foil::load_tensix_firmware(driver, hal, chip, core, tt::foil::firmware_elf(fw, FR::NCRISC), tt::foil::kNcrisc);
    tt::foil::load_tensix_firmware(driver, hal, chip, core, tt::foil::firmware_elf(fw, FR::TRISC0), tt::foil::kTrisc0);
    tt::foil::load_tensix_firmware(driver, hal, chip, core, tt::foil::firmware_elf(fw, FR::TRISC1), tt::foil::kTrisc1);
    tt::foil::load_tensix_firmware(driver, hal, chip, core, tt::foil::firmware_elf(fw, FR::TRISC2), tt::foil::kTrisc2);
    tt::foil::zero_fill_bank_tables(driver, hal, chip, core);
    tt::foil::init_tensix_core_info_minimal(driver, hal, chip, core, lx, ly);
    tt::foil::init_tensix_mailboxes(driver, hal, chip, core);
}

uint8_t read_go_signal(tt::umd::Cluster& driver, const Hal& hal, uint32_t chip,
                       const tt::umd::CoreCoord& core) {
    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto go = factory.create<dev_msgs::go_msg_t>();
    uint64_t go_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    driver.read_from_device(go.data(), chip, core, go_addr, go.size());
    return go.view().signal();
}
}  // namespace

int main() try {
    const char* e = std::getenv("TT_FOIL_DEVICE");
    int idx = e ? std::stoi(e) : 0;

    auto dev = tt::foil::open_device(idx, "", {{0, 0}});
    auto& driver = *dev->umd_driver;
    const auto& hal = *dev->hal;
    uint32_t chip = dev->chip_id;
    const auto& soc_desc = driver.get_soc_descriptor(chip);

    auto fw = tt::foil::resolve_firmware_paths(
        []() -> std::string {
            if (const char* p = std::getenv("TT_METAL_RUNTIME_ROOT")) return p;
            return "/home/kyamaguchi/tt-metal";
        }());

    // Probe a representative grid; key data points are logical (0,0) → (1,2)
    // (works) vs logical (0,1) → (1,3) (fails) vs logical (1,*) → (2,*)
    // (all work).
    // Read each candidate core's NOC0 NODE_ID register (0xFFB20044). If the
    // core is a healthy Tensix tile, the lower 12 bits encode (my_y << 6 |
    // my_x) — matching the NoC0 coord of the tile. If the core is router-only
    // or otherwise non-Tensix-worker, NODE_ID will read junk or 0.
    auto read_node_id = [&](tt::umd::CoreCoord t) {
        uint32_t v = 0xdeadbeef;
        driver.read_from_device(&v, chip, t, 0xFFB20044ull, sizeof(v));
        return v;
    };
    // Round-trip logical -> translated -> logical for all logical cells.
    auto grid = soc_desc.get_grid_size(tt::CoreType::TENSIX);
    std::printf("Tensix LOGICAL grid: %zu x %zu\n", grid.x, grid.y);
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            try {
                tt::umd::CoreCoord lc{x, y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
                auto t = soc_desc.translate_coord_to(lc, tt::CoordSystem::TRANSLATED);
                auto lc_back = soc_desc.translate_coord_to(t, tt::CoordSystem::LOGICAL);
                if (lc_back.x != x || lc_back.y != y) {
                    std::printf("    L(%u,%u) -> T(%zu,%zu) -> L(%zu,%zu) MISMATCH\n",
                                x, y, t.x, t.y, lc_back.x, lc_back.y);
                }
            } catch (const std::exception& e) {
                std::printf("    L(%u,%u) THROW: %s\n", x, y, e.what());
            }
        }
    }
    std::printf("(no MISMATCH lines printed means round-trip is consistent)\n");

    LCore probes[] = {{0, 0}, {0, 1}, {0, 5}, {1, 0}, {1, 5}, {5, 5}};
    for (auto lc : probes) {
        tt::umd::CoreCoord lcoord{lc.x, lc.y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
        auto t = soc_desc.translate_coord_to(lcoord, tt::CoordSystem::TRANSLATED);

        // Initial state before we touch it.
        {
            auto rs_initial = driver.get_risc_reset_state(chip, t);
            std::printf(">>> logical (%u,%u) -> translated (%zu,%zu)  initial=0x%x\n",
                        lc.x, lc.y, t.x, t.y, static_cast<uint32_t>(rs_initial));
        }

        // Try broadcast assert first (covers ETH bcast for cluster-wide).
        driver.assert_risc_reset();

        // Read after broadcast.
        {
            auto rs_after_broadcast = driver.get_risc_reset_state(chip, t);
            std::printf("    after broadcast assert: 0x%x\n",
                        static_cast<uint32_t>(rs_after_broadcast));
        }

        prep_core(driver, hal, chip, t, lc.x, lc.y, fw);

        std::printf("=== logical (%u,%u) -> translated (%zu,%zu) ===\n", lc.x, lc.y, t.x, t.y);

        // Smoking-gun read: L1[0..16] right after firmware load, BEFORE
        // deassert. On BH, BRISC reset PC = 0 (no PC reg), so it executes
        // from L1[0]. tt-metal writes a JAL trampoline (opcode 0x6f, low
        // 7 bits) at L1[0] that jumps to MEM_BRISC_FIRMWARE_BASE = 0x38E0.
        // If L1[0] is zero on a failing core, BRISC runs NOPs forever
        // even though firmware is loaded at 0x38E0.
        uint32_t l1_head[4] = {0};
        uint32_t fw_head[4] = {0};
        driver.read_from_device(l1_head, chip, t, 0x0,    sizeof(l1_head));
        driver.read_from_device(fw_head, chip, t, 0x38E0, sizeof(fw_head));
        std::printf("    L1[0..15]      = %08x %08x %08x %08x  (opcode L1[0]=0x%02x)\n",
                    l1_head[0], l1_head[1], l1_head[2], l1_head[3], l1_head[0] & 0x7f);
        std::printf("    L1[0x38E0..]   = %08x %08x %08x %08x\n",
                    fw_head[0], fw_head[1], fw_head[2], fw_head[3]);

        // Read soft-reset state BEFORE deassert. Should show BRISC bit set
        // (we asserted it in prep_core).
        {
            auto rs = driver.get_risc_reset_state(chip, t);
            bool brisc_in_reset = (rs & tt::umd::RiscType::BRISC) != tt::umd::RiscType::NONE;
            std::printf("    soft-reset pre-deassert: BRISC=%s (raw=0x%x)\n",
                        brisc_in_reset ? "ASSERTED" : "DEASSERTED",
                        static_cast<uint32_t>(rs));
        }

        tt::foil::deassert_brisc_reset(driver, chip, t);

        // Read soft-reset state AFTER deassert. Should show BRISC bit cleared.
        {
            auto rs = driver.get_risc_reset_state(chip, t);
            bool brisc_in_reset = (rs & tt::umd::RiscType::BRISC) != tt::umd::RiscType::NONE;
            std::printf("    soft-reset post-deassert: BRISC=%s (raw=0x%x)\n",
                        brisc_in_reset ? "STILL ASSERTED" : "deasserted",
                        static_cast<uint32_t>(rs));
        }

        // Sample at a few intervals.
        for (int ms : {1, 50, 500}) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            uint8_t sig = read_go_signal(driver, hal, chip, t);
            std::printf("  +%4dms  go.signal=0x%02x  %s\n", ms, sig,
                        sig == dev_msgs::RUN_MSG_DONE ? "DONE"
                        : sig == dev_msgs::RUN_MSG_INIT ? "INIT (firmware did not start)"
                        : "other");
        }
    }

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& ex) {
    std::fprintf(stderr, "probe: FAIL — %s\n", ex.what());
    return 1;
}
