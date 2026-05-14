// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 4b): CORE_INFO mailbox init.
//
// risc_firmware_initializer's populate_core_info_msg() fills a long list of
// fields — PCIE/DRAM/ETH non_worker_cores lists, harvesting masks, virtual
// coord tables, magic numbers, address bounds, etc. — that exist primarily
// for *optional* runtime features:
//
//   - non_worker_cores / virtual_non_worker_cores: watcher NOC sanitizer
//     (sanitize.h reads these to flag invalid host->NOC writes)
//   - l1_unreserved_start:                         DPRINT TileSlice debug
//   - noc_pcie_addr_*, noc_dram_addr_*:            watcher address-range checks
//
// Boot firmware itself (brisc.cc:365-366, trisc.cc:128-129, etc.) only reads
// `absolute_logical_x` and `absolute_logical_y`. For an embedded runtime with
// watcher/DPRINT disabled, a minimal core_info with just those two fields plus
// core_magic_number set is sufficient — anything else stays zero.
//
// This minimal path is what we implement here. A "full" CORE_INFO populator
// (non_worker_cores, harvesting, …) can come later if we ever need watcher.

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

// Write a minimal CORE_INFO_msg for a Tensix worker core. Sets
// absolute_logical_x/y to the passed logical coord and the WORKER magic
// number; everything else stays zero.
//
// Like init_tensix_mailboxes(), should be called while RISCs are in reset.
void init_tensix_core_info_minimal(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    uint32_t logical_x,
    uint32_t logical_y);

}  // namespace tt::foil
