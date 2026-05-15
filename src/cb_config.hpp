// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v4-3: host-side Circular Buffer (CB) configuration blob layout.
//
// At boot, BRISC/NCRISC/TRISC firmware all call setup_local_cb_*() (see
// tt_metal/hw/inc/internal/circular_buffer_init.h) which reads a contiguous
// uint32 array of CB descriptors out of L1 starting at
// `kernel_config_base + local_cb_offset`. Each descriptor is 4 × uint32:
//
//   word[0]  fifo_addr      L1 byte address of the CB ring buffer
//   word[1]  fifo_size      total ring size in bytes
//   word[2]  fifo_num_pages number of tile-sized slots in the ring
//   word[3]  fifo_page_size bytes per slot (one tile worth)
//
// TRISC reads these with `>>cb_addr_shift` (4) so it interprets the byte
// fields in 16-byte units; BRISC/NCRISC read them raw. The host stores raw
// bytes — the >>4 happens on the device.
//
// The launch_msg fields
//   kernel_config.local_cb_offset        — byte offset of the blob from kcfg_base
//   kernel_config.min_local_cb_start_index — lowest cb_index covered
//   kernel_config.local_cb_mask          — bitmask of populated cb indices,
//                                          shifted right by min_local_cb_start_index
// drive the per-CB setup loop in firmware. dispatch wires those at launch.

#pragma once

#include <cstdint>
#include <span>

#include "tt_foil/runtime.hpp"

namespace tt::foil {

struct Device;
struct Kernel;

// One Circular Buffer to register on a single core.
struct CbConfig {
    uint8_t  cb_index;      // 0..NUM_CIRCULAR_BUFFERS-1; standard mapping:
                            //   c_0..c_7   = data movement inputs
                            //   c_16..c_23 = data movement outputs
    uint64_t fifo_addr;     // L1 byte address where the ring lives
    uint32_t fifo_size;     // total ring size in bytes (== num_pages * page_size)
    uint32_t num_pages;     // tile-slot count
    uint32_t page_size;     // bytes per slot
};

// Result of `register_cbs`. Stored on Kernel and consumed by dispatch_execute
// to populate launch_msg. Set on Kernel even when cbs is empty (all zeros)
// so dispatch can leave the CB fields default.
struct CbAllocation {
    uint64_t blob_l1_addr{0};       // absolute L1 address of the CB blob (debug)
    uint32_t local_cb_offset{0};    // byte offset from kernel_config_base
    uint32_t local_cb_mask{0};      // bitmask of populated cb indices, *unshifted*
    uint8_t  min_local_cb_start_index{0};
    bool     valid{false};          // true iff any CBs were registered
};

// Allocate space inside the core's KERNEL_CONFIG region for `cbs`, fill the
// blob with the descriptors, write it to L1, and record the allocation on
// `kernel`. Must be called after `load_kernel` (the kernel_config bump
// allocator state and kernel.core both need to exist) and before
// `execute()`.
//
// Indices in `cbs` need not be contiguous; entries in [min_idx, max_idx]
// that aren't listed are written as zeros so firmware's bounds-aware loop
// skips them via the mask.
void register_cbs(
    Device& dev,
    Kernel& kernel,
    std::span<const CbConfig> cbs);

}  // namespace tt::foil
