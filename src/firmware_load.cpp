// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "firmware_load.hpp"

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
}

}  // namespace tt::foil
