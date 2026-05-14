// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 3) integration test: byte-for-byte equivalence between our
// load_tensix_firmware() and what tt-metal's CreateDevice() writes.
//
// Strategy (RISCs stay in reset throughout — no firmware actually runs):
//   1. CreateDevice loads firmware via tt-metal's normal path; capture the
//      first N words of BRISC firmware from L1 as the ground-truth snapshot.
//   2. Overwrite that range with a marker (0xDEADBEEF...) via UMD direct.
//   3. Confirm the overwrite was visible (read-back matches marker).
//   4. Call our load_tensix_firmware() with the same BRISC ELF.
//   5. Read back and confirm we restored the exact bytes from step 1.
//
// Same dance for NCRISC. Trisc 0/1/2 are skipped: their firmware lives in
// IRAM and the simple "read fw_base bytes" check doesn't apply uniformly.
//
// After this test, BRISC/NCRISC firmware in L1 is restored to its tt-metal
// state; the chip is left as if only CreateDevice had run.
//
// Usage:
//   TT_METAL_RUNTIME_ROOT=/path/to/tt-metal TT_FOIL_DEVICE=3 ./test_firmware_load

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "firmware_load.hpp"
#include "firmware_paths.hpp"

namespace {
constexpr uint32_t kTensixIdx = 0;
constexpr std::size_t kSnapshotBytes = 1024;  // first 1KB of firmware text

bool verify_one(
    const char* risc_name,
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    const std::string& elf_path,
    tt::foil::TensixRiscId risc) {

    const auto& jit_cfg = hal.get_jit_build_config(
        kTensixIdx, risc.processor_class, risc.processor_type);
    const uint64_t fw_base = hal.relocate_dev_addr(
        jit_cfg.fw_base_addr, jit_cfg.local_init_addr, false);

    std::vector<uint32_t> snapshot(kSnapshotBytes / 4, 0);
    driver.read_from_device(snapshot.data(), chip_id, core, fw_base, kSnapshotBytes);

    bool nonzero = false;
    for (auto w : snapshot) if (w != 0) { nonzero = true; break; }
    if (!nonzero) {
        std::fprintf(stderr, "%s: firmware read-back at 0x%lx is all zero — "
            "CreateDevice did not load firmware here\n", risc_name, fw_base);
        return false;
    }
    std::printf("%s: snapshot captured @ 0x%lx (1KB, first word 0x%08x)\n",
                risc_name, fw_base, snapshot[0]);

    std::vector<uint32_t> marker(kSnapshotBytes / 4, 0xDEADBEEF);
    driver.write_to_device(marker.data(), kSnapshotBytes, chip_id, core, fw_base);

    std::vector<uint32_t> readback(kSnapshotBytes / 4, 0);
    driver.read_from_device(readback.data(), chip_id, core, fw_base, kSnapshotBytes);
    if (readback != marker) {
        std::fprintf(stderr, "%s: marker overwrite did not take effect\n", risc_name);
        return false;
    }
    std::printf("%s: fw region overwritten with marker, verified\n", risc_name);

    tt::foil::load_tensix_firmware(driver, hal, chip_id, core, elf_path, risc);

    std::vector<uint32_t> restored(kSnapshotBytes / 4, 0);
    driver.read_from_device(restored.data(), chip_id, core, fw_base, kSnapshotBytes);

    if (restored != snapshot) {
        std::fprintf(stderr, "%s: byte-for-byte mismatch vs tt-metal's load\n", risc_name);
        for (size_t i = 0; i < restored.size(); ++i) {
            if (restored[i] != snapshot[i]) {
                std::fprintf(stderr, "  word %zu: ours=0x%08x tt-metal=0x%08x\n",
                             i, restored[i], snapshot[i]);
                if (i > 4) break;
            }
        }
        return false;
    }
    std::printf("%s: load_tensix_firmware output matches tt-metal byte-for-byte\n", risc_name);
    return true;
}

}  // namespace

int main() try {
    const char* root_env = std::getenv("TT_METAL_RUNTIME_ROOT");
    std::string tt_metal_root = root_env ? root_env : "/home/kyamaguchi/tt-metal";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Use CreateDevice for HAL + firmware-loaded ground truth.
    auto* dev = tt::tt_metal::CreateDevice(static_cast<tt::ChipId>(pcie_index));
    auto& ctx = tt::tt_metal::MetalContext::instance();
    const auto& hal = ctx.hal();
    auto* umd_driver = ctx.get_cluster().get_driver().get();
    std::printf("test_firmware_load: CreateDevice done; using HAL via MetalContext\n");

    // Resolve translated coord for logical (0,0) — same as our other tests.
    auto virt = dev->worker_core_from_logical_core({0, 0});
    tt::umd::CoreCoord core{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    std::printf("test_firmware_load: target translated coord (%zu,%zu)\n",
                core.x, core.y);

    auto paths = tt::foil::resolve_firmware_paths(tt_metal_root);
    std::printf("test_firmware_load: firmware root = %s\n", paths.root.c_str());

    bool ok = true;
    ok &= verify_one("BRISC",  *umd_driver, hal, pcie_index, core, paths.brisc,  tt::foil::kBrisc);
    ok &= verify_one("NCRISC", *umd_driver, hal, pcie_index, core, paths.ncrisc, tt::foil::kNcrisc);

    tt::tt_metal::CloseDevice(dev);

    if (!ok) {
        std::puts("test_firmware_load: FAIL");
        return 1;
    }
    std::puts("test_firmware_load: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_firmware_load: FAIL — %s\n", e.what());
    return 1;
}
