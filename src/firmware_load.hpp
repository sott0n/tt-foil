// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 3): RISC firmware ELF load via UMD direct.
//
// Mirrors tt_metal::llrt::test_load_multicast_write_risc_binary, but unicast
// to a single core via umd::Cluster::write_to_device. For each span emitted
// by ll_api::memory::process_spans(), the per-RISC `local_init_addr` from
// HAL is fed into hal.relocate_dev_addr() to get the actual target address.
//
// HAL is taken as a const reference rather than owned here — for B2 we still
// borrow it from MetalContext; standalone HAL instantiation lands in B3.

#pragma once

#include <cstdint>
#include <string>

namespace tt {
namespace umd {
class Cluster;
struct CoreCoord;
}  // namespace umd
namespace tt_metal {
class Hal;
}
}  // namespace tt

namespace tt::foil {

// Identifies a Tensix RISC by HAL (processor_class, processor_type) indices.
// Blackhole Tensix layout:
//   class 0 (DM):      type 0 = BRISC,  type 1 = NCRISC
//   class 1 (COMPUTE): type 0 = TRISC0, type 1 = TRISC1, type 2 = TRISC2
struct TensixRiscId {
    uint32_t processor_class;
    uint32_t processor_type;
};

inline constexpr TensixRiscId kBrisc{0, 0};
inline constexpr TensixRiscId kNcrisc{0, 1};
inline constexpr TensixRiscId kTrisc0{1, 0};
inline constexpr TensixRiscId kTrisc1{1, 1};
inline constexpr TensixRiscId kTrisc2{1, 2};

// Load one RISC firmware ELF into the L1 of `core` (UMD translated coord).
//
// Throws std::runtime_error if the ELF can't be opened or the HAL config
// can't be resolved.
void load_tensix_firmware(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    const std::string& elf_path,
    TensixRiscId risc);

}  // namespace tt::foil
