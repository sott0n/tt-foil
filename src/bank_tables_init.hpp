// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Boot-time initializer for the two L1 scratch regions that firmware
// copies into local-memory bank/coord arrays at startup.
//
// At boot, BRISC and NCRISC firmware unconditionally copy
//   - BANK_TO_NOC_SCRATCH        -> dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS]
//                                   l1_bank_to_noc_xy [NUM_NOCS][NUM_L1_BANKS]
//                                   bank_to_dram_offset[NUM_DRAM_BANKS]
//                                   bank_to_l1_offset [NUM_L1_BANKS]
//   - LOGICAL_TO_VIRTUAL_SCRATCH -> worker_logical_{col,row}_to_virtual_*
// via l1_to_local_mem_copy (see firmware_common.h::noc_bank_table_init
// and noc_worker_logical_to_virtual_map_init).
//
// These arrays back the modern `TensorAccessor` / `experimental::Noc`
// kernel API for DRAM-interleaved buffers, which resolves a page_id to
// `dram_bank_to_noc_xy[noc][page_id % NUM_DRAM_BANKS]`. Zero-fill makes
// every page resolve to NOC (0, 0), so kernels using that API silently
// read from the wrong endpoint.
//
// tt-foil's existing kernels still bypass this — they pack the full
// NOC address host-side via `make_noc_dram_addr` and pass it through
// RTA — so the previous zero-fill was correctness-safe for the
// legacy path. The real DRAM table is purely additive: it unlocks
// tt-metal's modern dataflow API for DRAM-interleaved reads while
// leaving the legacy host-RTA path untouched.
//
// L1-interleaved buffers and `get_noc_addr_from_logical_xy()` are
// still unused in tt-foil, so `l1_bank_to_noc_xy`,
// `bank_to_{dram,l1}_offset`, and the logical-to-virtual scratch
// remain zero. Wire them up the same way as DRAM when needed.

#pragma once

#include <cstdint>

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

// Populate the two scratch regions on one Tensix core. Like the other
// boot-time writers, must be called while RISCs are in reset.
void init_bank_tables(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core);

}  // namespace tt::foil
