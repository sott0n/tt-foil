// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for global_avg_pool — pushes one persistent scaler tile
// (full of 1/HW values) into CB c_1, then streams Nt input tiles into
// CB c_0 from DRAM. The compute kernel reduces each input row-wise
// (ReduceDim::REDUCE_ROW) and the scaler turns per-row sum into the
// per-channel mean.
//
// Runtime args:
//   arg[0..1] = input  DRAM NOC addr (lo, hi)
//   arg[2..3] = scaler DRAM NOC addr (lo, hi)
//   arg[4]    = Nt (number of input tiles to reduce, all share same Mt=1)

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t in_base     = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t scaler_base = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint32_t nt          = get_arg_val<uint32_t>(4);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // Load the scaler ONCE. The compute kernel never pops it; it stays
    // resident in c_1 for the full reduction.
    cb_reserve_back(1, 1);
    noc_async_read(scaler_base, get_write_ptr(1), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(1, 1);

    for (uint32_t i = 0; i < nt; ++i) {
        cb_reserve_back(0, 1);
        noc_async_read(in_base + i * kTileBytes, get_write_ptr(0), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
}
