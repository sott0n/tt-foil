// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "tt_foil/runtime.hpp"  // RiscBinary, CoreCoord

// Forward declaration (ll_api::memory is in the tt-metal submodule)
namespace ll_api { class memory; }

namespace tt::foil {

struct Device;

// Blackhole has 5 RISC processors per Tensix core.
// v1 exposes only BRISC (DM0) and NCRISC (DM1).
static constexpr uint32_t kMaxRiscs = 5;

// Processor indices within the Blackhole Tensix HAL config.
// processor_class 0 = DM, types: 0 = BRISC, 1 = NCRISC
// processor_class 1 = COMPUTE, types: 0 = TRISC0, 1 = TRISC1, 2 = TRISC2
static constexpr uint32_t kBriscProcClass = 0;
static constexpr uint32_t kBriscProcType  = 0;
static constexpr uint32_t kNcriscProcClass = 0;
static constexpr uint32_t kNcriscProcType  = 1;

// Index into the kernel_config_msg_t.rta_offset array.
// In Blackhole, BRISC = 0, NCRISC = 1 (processor index within MaxProcessorsPerCoreType).
static constexpr uint32_t kBriscProcessorIndex  = 0;
static constexpr uint32_t kNcriscProcessorIndex = 1;

struct LoadedRisc {
    uint32_t proc_class;
    uint32_t proc_type;
    uint32_t processor_index;  // index into rta_offset[] and kernel_text_offset[]
    std::unique_ptr<ll_api::memory> mem;
    std::vector<uint32_t> runtime_args;
};

struct Kernel {
    CoreCoord core;
    std::vector<LoadedRisc> riscs;

    // L1 address where runtime args for each risc will be written.
    // Layout: consecutive uint32_t arrays for each active risc.
    // Address is allocated from the core's L1 bump allocator at load_kernel() time.
    uint64_t rta_base_addr{0};
    uint32_t rta_region_size{0};  // total bytes reserved for all RTA arrays
};

// Resolve a RiscId to HAL processor_class + processor_type indices.
inline void risc_to_hal_indices(RiscBinary::RiscId id, uint32_t& proc_class, uint32_t& proc_type, uint32_t& proc_idx) {
    switch (id) {
        case RiscBinary::RiscId::BRISC:
            proc_class = kBriscProcClass; proc_type = kBriscProcType; proc_idx = kBriscProcessorIndex;
            break;
        case RiscBinary::RiscId::NCRISC:
            proc_class = kNcriscProcClass; proc_type = kNcriscProcType; proc_idx = kNcriscProcessorIndex;
            break;
    }
}

// Internal implementation (called by runtime.hpp wrappers).
Kernel* kernel_load(
    Device& dev,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core);

void kernel_set_runtime_args(
    Kernel& kernel,
    RiscBinary::RiscId risc,
    std::span<const uint32_t> args);

}  // namespace tt::foil
