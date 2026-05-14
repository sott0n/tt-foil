// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B3 (step 1) probe: instantiate tt::tt_metal::Hal directly, without
// going through MetalContext, and confirm we can read the same addresses
// our cold-boot path consumes (LAUNCH, GO_MSG, BANK_TO_NOC_SCRATCH,
// per-RISC local_init_addr, relocate_dev_addr).
//
// If this works, dropping MetalContext from device_open is mechanical —
// just newing a Hal of the right architecture is enough.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "llrt/hal.hpp"
#include <umd/device/types/arch.hpp>

int main() try {
    using namespace tt::tt_metal;

    // Same defaults MetalEnvImpl::initialize_base_objects() uses for a
    // single Blackhole MMIO chip — see metal_env.cpp:138-145.
    Hal hal(
        tt::ARCH::BLACKHOLE,
        /*is_base_routing_fw_enabled=*/false,
        /*enable_2_erisc_mode=*/false,
        /*profiler_dram_bank_size_per_risc_bytes=*/0,
        /*enable_dram_backed_cq=*/false,
        /*is_simulator=*/false,
        /*enable_blackhole_dram_programmable_cores=*/false);

    std::printf("hal arch = %d\n", static_cast<int>(hal.get_arch()));
    std::printf("LAUNCH                    = 0x%lx\n",
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH));
    std::printf("GO_MSG                    = 0x%lx\n",
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG));
    std::printf("BANK_TO_NOC_SCRATCH       = 0x%lx (size %u)\n",
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH),
        hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH));
    std::printf("CORE_INFO                 = 0x%lx\n",
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::CORE_INFO));
    std::printf("KERNEL_CONFIG             = 0x%lx\n",
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG));

    // Per-RISC local_init_addr (what load_tensix_firmware reads).
    for (uint32_t pc = 0; pc < 2; ++pc) {
        for (uint32_t pt = 0; pt < 3; ++pt) {
            try {
                const auto& cfg = hal.get_jit_build_config(0 /*TENSIX*/, pc, pt);
                std::printf("class=%u type=%u fw_base=0x%lx local_init=0x%lx\n",
                    pc, pt, cfg.fw_base_addr, cfg.local_init_addr);
            } catch (...) { /* not all combinations valid */ }
        }
    }

    // Sanity: relocate_dev_addr round-trip
    uint64_t reloc = hal.relocate_dev_addr(0xffb00000, 0, false);
    std::printf("relocate(0xffb00000, 0, false) = 0x%lx\n", reloc);

    std::puts("test_hal_standalone: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_hal_standalone: FAIL — %s\n", e.what());
    return 1;
}
