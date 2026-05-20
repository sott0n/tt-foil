// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for transpose_2d: streams a [Rt, Ct] tile-format tensor
// from DRAM into cb_in in transposed tile order.
//
// For each output position (ct, rt), reads the input tile at (rt, ct) and
// pushes it to cb_in. The compute kernel then performs the within-tile
// WH transpose; the writer drains cb_out into [Ct, Rt] DRAM layout.
//
// Runtime args:
//   arg[0,1] = in_noc (lo, hi)
//   arg[2]   = Rt
//   arg[3]   = Ct

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void kernel_main() {
    const uint64_t in_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t Rt     = get_arg_val<uint32_t>(2);
    const uint32_t Ct     = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in      = 0;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // Visit output tiles in row-major order (ct outer, rt inner) and read
    // the corresponding input tile (rt, ct).
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        for (uint32_t rt = 0; rt < Rt; ++rt) {
            uint64_t in_addr = in_noc + (rt * Ct + ct) * kTileBytes;
            cb_reserve_back(cb_in, 1);
            noc_async_read(in_addr, get_write_ptr(cb_in), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
