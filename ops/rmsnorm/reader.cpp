// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for RMSNorm.
//
// Runtime args:
//   arg[0..1] = x DRAM NOC (lo, hi)
//   arg[2..3] = gamma DRAM NOC (lo, hi)
//   arg[4..5] = scaler DRAM NOC (lo, hi)  — tile filled with BF16(1/H)
//   arg[6..7] = eps DRAM NOC (lo, hi)     — tile filled with BF16(eps)
//   arg[8]    = NCHt  (number of token tile-rows)
//   arg[9]    = Wt    (hidden tiles per row = H/32)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t x_noc      = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t gamma_noc  = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint64_t eps_noc    = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    const uint32_t NCHt       = get_arg_val<uint32_t>(8);
    const uint32_t Wt         = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_inp    = 0;
    constexpr uint32_t cb_reduce = 1;
    constexpr uint32_t cb_gamma  = 2;
    constexpr uint32_t cb_eps    = 3;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // Load scaler tile once (persistent — never popped by compute)
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    // Load eps tile once (persistent)
    cb_reserve_back(cb_eps, 1);
    noc_async_read(eps_noc, get_write_ptr(cb_eps), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_eps, 1);

    // Load gamma tiles once (persistent, Wt tiles)
    cb_reserve_back(cb_gamma, Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        noc_async_read(
            gamma_noc + static_cast<uint64_t>(wt) * kTileBytes,
            get_write_ptr(cb_gamma) + wt * kTileBytes,
            kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Wt);

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
