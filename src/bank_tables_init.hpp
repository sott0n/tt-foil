// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 5): zero-fill the two bank-routing L1 scratch regions.
//
// At boot, BRISC and NCRISC firmware unconditionally copy
//   - BANK_TO_NOC_SCRATCH        -> dram/l1_bank_to_noc_xy + offset maps
//   - LOGICAL_TO_VIRTUAL_SCRATCH -> worker_logical_{col,row}_to_virtual_*
// via l1_to_local_mem_copy (see firmware_common.h::noc_bank_table_init and
// noc_worker_logical_to_virtual_map_init, called from brisc.cc:360-361 /
// ncrisc.cc:115-116).
//
// The copy targets are then consulted by *user kernels* via the dataflow_api
// (interleaved buffer addressing, get_noc_addr_from_logical_xy, etc). For
// tt-foil's slow-dispatch embedded use case kernels pass explicit L1
// addresses through runtime args and never touch the bank/virtual maps, so
// the contents don't matter for correctness — but we still have to write
// *something*, otherwise firmware copies whatever garbage was in L1.
//
// Zero-fill is the simplest valid initialization. If a future kernel needs
// interleaved buffer addressing, the corresponding writer needs to be
// upgraded to mirror tt-metal's risc_firmware_initializer logic.

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

// Zero-fill the two scratch regions on one Tensix core.
//
// Like the other boot-time writers, must be called while RISCs are in reset.
void zero_fill_bank_tables(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core);

}  // namespace tt::foil
