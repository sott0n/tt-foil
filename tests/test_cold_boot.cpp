// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (steps 6 + 7) end-to-end test: cold-boot one Tensix core using
// only our UMD-direct boot primitives. Composes:
//
//   firmware_load     (B2-3) — write 5 RISC firmware ELFs to fw_base
//   bank_tables_init  (B2-5) — zero-fill the two scratch regions
//   core_info_init    (B2-4b) — minimal core_info_msg (abs_logical + magic)
//   mailbox_init      (B2-4)  — zero launch_msg buffer, RUN_MSG_INIT, ptrs=0
//   reset             (B2-6)  — assert all RISCs, deassert BRISC
//   wait_init_done    (B2-7)  — poll GO_MSG until RUN_MSG_DONE
//
// We still piggyback on CreateDevice/MetalContext for HAL (Phase B2 keeps
// HAL via libtt_metal.so per plan). The test:
//
//   1. CreateDevice — tt-metal performs its own boot; gets us a HAL handle
//      and a fully-initialized chip.
//   2. assert_tensix_reset on logical (0,0) — halt BRISC + subordinates.
//   3. Run our boot sequence over the same core: load firmware, zero bank
//      tables, write minimal core_info, write boot-time mailboxes,
//      deassert BRISC.
//   4. wait_tensix_init_done — confirms BRISC firmware reached RUN_MSG_DONE.
//
// If anything in our chain is wrong, BRISC firmware hangs and the poll
// times out. Passing this test proves the full cold-boot path works.
//
// Usage: TT_FOIL_DEVICE=3 TT_FOIL_FIRMWARE_DIR=<dir matching tt-metal> ./test_cold_boot

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tt-metalium/host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "bank_tables_init.hpp"
#include "core_info_init.hpp"
#include "firmware_load.hpp"
#include "firmware_paths.hpp"
#include "mailbox_init.hpp"
#include "reset.hpp"

int main() try {
    using namespace tt::tt_metal;

    const char* root_env = std::getenv("TT_METAL_RUNTIME_ROOT");
    std::string tt_metal_root = root_env ? root_env : "/home/kyamaguchi/tt-metal";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto* dev = CreateDevice(static_cast<tt::ChipId>(pcie_index));
    auto& ctx = MetalContext::instance();
    const auto& hal = ctx.hal();
    auto* driver = ctx.get_cluster().get_driver().get();
    std::puts("test_cold_boot: CreateDevice done; tt-metal's boot finished");

    constexpr uint32_t kLogX = 0, kLogY = 0;
    auto virt = dev->worker_core_from_logical_core({kLogX, kLogY});
    tt::umd::CoreCoord core{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    std::printf("test_cold_boot: target translated coord (%zu,%zu)\n", core.x, core.y);

    auto paths = tt::foil::resolve_firmware_paths(tt_metal_root);
    std::printf("test_cold_boot: firmware root = %s\n", paths.root.c_str());

    // ---- 1. Halt the core ----
    tt::foil::assert_tensix_reset(*driver, pcie_index, core);
    std::puts("test_cold_boot: all RISCs reset asserted");

    // ---- 2. Reload firmware ELFs (B2-3) ----
    tt::foil::load_tensix_firmware(*driver, hal, pcie_index, core, paths.brisc,  tt::foil::kBrisc);
    tt::foil::load_tensix_firmware(*driver, hal, pcie_index, core, paths.ncrisc, tt::foil::kNcrisc);
    tt::foil::load_tensix_firmware(*driver, hal, pcie_index, core, paths.trisc0, tt::foil::kTrisc0);
    tt::foil::load_tensix_firmware(*driver, hal, pcie_index, core, paths.trisc1, tt::foil::kTrisc1);
    tt::foil::load_tensix_firmware(*driver, hal, pcie_index, core, paths.trisc2, tt::foil::kTrisc2);
    std::puts("test_cold_boot: firmware (5 RISCs) reloaded");

    // ---- 3. Zero-fill bank tables (B2-5) ----
    tt::foil::zero_fill_bank_tables(*driver, hal, pcie_index, core);
    std::puts("test_cold_boot: bank tables zeroed");

    // ---- 4. core_info_msg (B2-4b) ----
    tt::foil::init_tensix_core_info_minimal(*driver, hal, pcie_index, core, kLogX, kLogY);
    std::puts("test_cold_boot: core_info_msg written");

    // ---- 5. mailbox boot state (B2-4) ----
    tt::foil::init_tensix_mailboxes(*driver, hal, pcie_index, core);
    std::puts("test_cold_boot: mailboxes set to boot state (GO_MSG = RUN_MSG_INIT)");

    // ---- 6. Deassert BRISC reset (B2-6) ----
    tt::foil::deassert_brisc_reset(*driver, pcie_index, core);
    std::puts("test_cold_boot: BRISC reset deasserted");

    // ---- 7. Poll until RUN_MSG_DONE (B2-7) ----
    tt::foil::wait_tensix_init_done(*driver, hal, pcie_index, core, /*timeout_ms=*/10000);
    std::puts("test_cold_boot: BRISC firmware reached RUN_MSG_DONE ✓");

    CloseDevice(dev);
    std::puts("test_cold_boot: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_cold_boot: FAIL — %s\n", e.what());
    return 1;
}
