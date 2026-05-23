// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused RMSNorm + RoPE.
//
// Stream layout (per row r in [0, NCHt) where NCHt = St * num_heads):
//   Wt = 2 * Dt_half  tiles of x → cb_x (depth Wt)
// Persistent constants:
//   cb_reduce (scaler), cb_eps, cb_gamma (Wt tiles),
//   cb_cos (St * Dt_half tiles), cb_sin (St * Dt_half tiles).
//
// Runtime args:
//   arg[0..1]  = x DRAM NOC (lo, hi)
//   arg[2..3]  = gamma DRAM NOC (lo, hi)
//   arg[4..5]  = scaler DRAM NOC (lo, hi)
//   arg[6..7]  = eps DRAM NOC (lo, hi)
//   arg[8..9]  = cos DRAM NOC (lo, hi)
//   arg[10..11]= sin DRAM NOC (lo, hi)
//   arg[12]    = NCHt
//   arg[13]    = Wt        (= 2 * Dt_half)
//   arg[14]    = St
//   arg[15]    = Dt_half

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t x_noc      = join64(get_arg_val<uint32_t>(0),  get_arg_val<uint32_t>(1));
    const uint64_t gamma_noc  = join64(get_arg_val<uint32_t>(2),  get_arg_val<uint32_t>(3));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(4),  get_arg_val<uint32_t>(5));
    const uint64_t eps_noc    = join64(get_arg_val<uint32_t>(6),  get_arg_val<uint32_t>(7));
    const uint64_t cos_noc    = join64(get_arg_val<uint32_t>(8),  get_arg_val<uint32_t>(9));
    const uint64_t sin_noc    = join64(get_arg_val<uint32_t>(10), get_arg_val<uint32_t>(11));
    const uint32_t NCHt       = get_arg_val<uint32_t>(12);
    const uint32_t Wt         = get_arg_val<uint32_t>(13);
    const uint32_t St         = get_arg_val<uint32_t>(14);
    const uint32_t Dt_half    = get_arg_val<uint32_t>(15);

    constexpr uint32_t cb_x      = 0;
    constexpr uint32_t cb_reduce = 1;
    constexpr uint32_t cb_gamma  = 2;
    constexpr uint32_t cb_eps    = 3;
    constexpr uint32_t cb_cos    = 8;
    constexpr uint32_t cb_sin    = 9;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // Persistent constants
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    cb_reserve_back(cb_eps, 1);
    noc_async_read(eps_noc, get_write_ptr(cb_eps), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_eps, 1);

    cb_reserve_back(cb_gamma, Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        noc_async_read(gamma_noc + static_cast<uint64_t>(wt) * kTileBytes,
                       get_write_ptr(cb_gamma) + wt * kTileBytes,
                       kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Wt);

    const uint32_t cos_tiles = St * Dt_half;
    cb_reserve_back(cb_cos, cos_tiles);
    cb_reserve_back(cb_sin, cos_tiles);
    for (uint32_t i = 0; i < cos_tiles; ++i) {
        noc_async_read(cos_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_cos) + i * kTileBytes, kTileBytes);
        noc_async_read(sin_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_sin) + i * kTileBytes, kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_cos, cos_tiles);
    cb_push_back(cb_sin, cos_tiles);

    // Stream x row by row.
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_reserve_back(cb_x, Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read(
                x_noc + static_cast<uint64_t>(r * Wt + wt) * kTileBytes,
                get_write_ptr(cb_x) + wt * kTileBytes,
                kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, Wt);
    }
}
