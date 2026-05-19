// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for Softmax (per-row).
//
// Runtime args:
//   arg[0..1] = x DRAM NOC (lo, hi)
//   arg[2..3] = scaler DRAM NOC (lo, hi)  — tile filled with BF16(1.0) for SUM reduce
//   arg[4]    = NCHt  (number of token tile-rows)
//   arg[5]    = Wt    (tiles per row = W/32)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t x_noc      = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint32_t NCHt       = get_arg_val<uint32_t>(4);
    const uint32_t Wt         = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_inp    = 0;
    constexpr uint32_t cb_reduce = 1;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // Load scaler tile once (persistent — never popped by compute)
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    // Stream x tiles row by row
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_reserve_back(cb_inp, Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read(
                x_noc + static_cast<uint64_t>(r * Wt + wt) * kTileBytes,
                get_write_ptr(cb_inp) + wt * kTileBytes,
                kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_inp, Wt);
    }
}
