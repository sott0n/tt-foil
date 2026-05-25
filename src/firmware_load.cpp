// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "firmware_load.hpp"
#include "profiling.hpp"

#include <stdexcept>
#include <vector>

#include "llrt/hal.hpp"
#include "llrt_local/tt_memory.h"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

void load_tensix_firmware(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    const std::string& elf_path,
    TensixRiscId risc) {
    TF_ZONE_N("TF_firmware_load");

    // Tensix programmable_core_type_index is 0 on Blackhole.
    constexpr uint32_t kTensixIdx = 0;

    // local_init_addr feeds hal.relocate_dev_addr — different per RISC because
    // each RISC has its own per-core local memory region.
    const auto& jit_cfg = hal.get_jit_build_config(
        kTensixIdx, risc.processor_class, risc.processor_type);
    const uint64_t local_init_addr = jit_cfg.local_init_addr;

    // Firmware ELFs use DISCRETE loading — process_spans yields the original
    // ELF section addresses, which we then relocate.
    tt::foil::ll_api::memory fw_mem(elf_path, tt::foil::ll_api::memory::Loading::DISCRETE);

    fw_mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr,
                             uint64_t span_addr,
                             uint32_t len_words) {
        const uint64_t relo_addr = hal.relocate_dev_addr(
            span_addr, local_init_addr, /*has_shared_local_mem=*/false);
        driver.write_to_device(
            &*mem_ptr,
            static_cast<std::size_t>(len_words) * sizeof(uint32_t),
            chip_id, core, relo_addr);
    });

    // Program the per-RISC launch address. tt-metal's HAL gives us, for each
    // RISC, a write that has to land before deassert:
    //   - BRISC  on Blackhole: fw_launch_addr = 0x0 (L1[0]), fw_launch_addr_value =
    //     JAL trampoline that jumps to MEM_BRISC_FIRMWARE_BASE. BH BRISC has no
    //     reset-PC register (NCRISC/TRISC do), so it always executes from L1[0];
    //     without the trampoline a freshly-asserted BRISC hits 0x00000000
    //     (illegal instruction) at L1[0] and never reaches its firmware. Single-
    //     core tests passed in the past only because the chip's ARC bootrom had
    //     core (0,0) running, and our soft-reset assert→deassert merely halted
    //     and resumed BRISC inside leftover firmware.
    //   - NCRISC / TRISC0/1/2: fw_launch_addr is the per-RISC RESET_PC register,
    //     fw_launch_addr_value is the firmware base address. The Hal's value is
    //     a register write that programs the reset vector before deassert.
    //
    // The Hal exposes these uniformly via HalJitBuildConfig; we just honour them.
    if (jit_cfg.fw_launch_addr_value != 0 || jit_cfg.fw_launch_addr == 0) {
        // fw_launch_addr_value == 0 with fw_launch_addr != 0 means "no launch
        // write needed" (e.g., a RISC with a hardware-fixed reset vector). The
        // tt-1xx Tensix configs always populate both fields, but guard anyway.
        uint32_t launch_value = jit_cfg.fw_launch_addr_value;
        driver.write_to_device(
            &launch_value, sizeof(launch_value),
            chip_id, core, jit_cfg.fw_launch_addr);
    }
}

}  // namespace tt::foil
