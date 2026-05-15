// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v3-4: host-side NOC unicast address construction.
//
// Blackhole NOC unicast address layout (`NOC_XY_ADDR(x, y, addr)` from
// `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`):
//
//     bits [47:42]  destination NOC y    (NOC_ADDR_NODE_ID_BITS = 6)
//     bits [41:36]  destination NOC x    (NOC_ADDR_NODE_ID_BITS = 6)
//     bits [35:0]   local L1 byte offset (NOC_ADDR_LOCAL_BITS = 36)
//
// dataflow_api's `noc_async_write_one_packet(src_l1, dst_noc_addr, size)`
// takes that 64-bit dst_noc_addr verbatim. Pre-computing it on the host
// from a translated CoreCoord lets producer/consumer kernels stay free of
// the worker_logical_to_virtual table (which v3 zero-fills, see v2-5).

#pragma once

#include <cstdint>

namespace tt::umd { struct CoreCoord; }

namespace tt::foil {

// Pack a NOC unicast destination address from a TRANSLATED-system Tensix
// coord and a local L1 byte offset, matching Blackhole's NOC_XY_ADDR.
uint64_t make_noc_unicast_addr(
    const tt::umd::CoreCoord& translated_dst,
    uint64_t local_l1_addr);

}  // namespace tt::foil
