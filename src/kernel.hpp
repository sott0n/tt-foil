// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "tt_foil/runtime.hpp"  // RiscBinary, CoreCoord

// Full header needed for unique_ptr<tt::foil::ll_api::memory> member
#include "llrt_local/tt_memory.h"

namespace tt::foil {

struct Device;

// Blackhole has 5 RISC processors per Tensix core.
// v1 exposes only BRISC (DM0) and NCRISC (DM1).
static constexpr uint32_t kMaxRiscs = 5;

// Max runtime args per RISC processor.
static constexpr uint32_t kMaxRtaWords = 64;

// Processor indices within the Blackhole Tensix HAL config.
//   processor_class 0 = DM      → type 0 = BRISC, type 1 = NCRISC
//   processor_class 1 = COMPUTE → type 0 = TRISC0 (UNPACK),
//                                 type 1 = TRISC1 (MATH),
//                                 type 2 = TRISC2 (PACK)
static constexpr uint32_t kDmProcClass      = 0;
static constexpr uint32_t kComputeProcClass = 1;
static constexpr uint32_t kBriscProcType    = 0;
static constexpr uint32_t kNcriscProcType   = 1;
static constexpr uint32_t kTrisc0ProcType   = 0;
static constexpr uint32_t kTrisc1ProcType   = 1;
static constexpr uint32_t kTrisc2ProcType   = 2;

// Index into the kernel_config_msg_t.rta_offset[] and kernel_text_offset[]
// arrays. On Blackhole all 5 Tensix RISCs share a single processor index
// space: BRISC=0, NCRISC=1, TRISC0=2, TRISC1=3, TRISC2=4.
static constexpr uint32_t kBriscProcessorIndex  = 0;
static constexpr uint32_t kNcriscProcessorIndex = 1;
static constexpr uint32_t kTrisc0ProcessorIndex = 2;
static constexpr uint32_t kTrisc1ProcessorIndex = 3;
static constexpr uint32_t kTrisc2ProcessorIndex = 4;

struct LoadedRisc {
    uint32_t proc_class;
    uint32_t proc_type;
    uint32_t processor_index;  // index into rta_offset[] and kernel_text_offset[]
    std::unique_ptr<tt::foil::ll_api::memory> mem;
    std::vector<uint32_t> runtime_args;
    uint64_t kernel_text_addr{0};  // L1 address where kernel binary is stored (in KERNEL_CONFIG region)
};

struct Kernel {
    CoreCoord core;           // logical coordinate (user-facing)
    uint32_t virt_x{0};      // virtual (translated) x for UMD write_core
    uint32_t virt_y{0};      // virtual (translated) y for UMD write_core
    std::vector<LoadedRisc> riscs;

    // L1 address where runtime args for each risc will be written.
    // Layout: consecutive uint32_t arrays for each active risc.
    // Address is allocated from the core's L1 bump allocator at load_kernel() time.
    uint64_t rta_base_addr{0};
    uint32_t rta_region_size{0};  // total bytes reserved for all RTA arrays
};

// Resolve a RiscId to HAL processor_class + processor_type indices, plus the
// flat processor index that launch_msg's rta_offset[] / kernel_text_offset[]
// arrays are indexed by.
inline void risc_to_hal_indices(
    RiscBinary::RiscId id,
    uint32_t& proc_class,
    uint32_t& proc_type,
    uint32_t& proc_idx) {
    switch (id) {
        case RiscBinary::RiscId::BRISC:
            proc_class = kDmProcClass;      proc_type = kBriscProcType;  proc_idx = kBriscProcessorIndex;  break;
        case RiscBinary::RiscId::NCRISC:
            proc_class = kDmProcClass;      proc_type = kNcriscProcType; proc_idx = kNcriscProcessorIndex; break;
        case RiscBinary::RiscId::TRISC0:
            proc_class = kComputeProcClass; proc_type = kTrisc0ProcType; proc_idx = kTrisc0ProcessorIndex; break;
        case RiscBinary::RiscId::TRISC1:
            proc_class = kComputeProcClass; proc_type = kTrisc1ProcType; proc_idx = kTrisc1ProcessorIndex; break;
        case RiscBinary::RiscId::TRISC2:
            proc_class = kComputeProcClass; proc_type = kTrisc2ProcType; proc_idx = kTrisc2ProcessorIndex; break;
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
