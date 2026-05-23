// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused add+RMSNorm.
//   S = A + B  (residual sum, also written back to DRAM by NCRISC)
//   Y = RMSNorm(S, gamma)
//
// Streams A and B tile-by-tile in lockstep (depth-1 CBs cb_a/cb_b) plus
// the standard rmsnorm constants (gamma, scaler, eps).
//
// Runtime args:
//   arg[0..1]  = A DRAM NOC (lo, hi)
//   arg[2..3]  = B DRAM NOC (lo, hi)
//   arg[4..5]  = gamma DRAM NOC (lo, hi)
//   arg[6..7]  = scaler DRAM NOC (lo, hi)  — tile filled with BF16(1/H)
//   arg[8..9]  = eps DRAM NOC (lo, hi)     — tile filled with BF16(eps)
//   arg[10]    = NCHt
//   arg[11]    = Wt

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t a_noc      = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t b_noc      = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t gamma_noc  = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    const uint64_t eps_noc    = join64(get_arg_val<uint32_t>(8), get_arg_val<uint32_t>(9));
    const uint32_t NCHt       = get_arg_val<uint32_t>(10);
    const uint32_t Wt         = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_a      = 0;
    constexpr uint32_t cb_reduce = 1;
    constexpr uint32_t cb_gamma  = 2;
    constexpr uint32_t cb_eps    = 3;
    constexpr uint32_t cb_b      = 8;   // residual addend (depth 1)
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
        noc_async_read(
            gamma_noc + static_cast<uint64_t>(wt) * kTileBytes,
            get_write_ptr(cb_gamma) + wt * kTileBytes,
            kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Wt);

    // Tile-by-tile lockstep A/B stream (depth-1 CBs to keep L1 use low).
    for (uint32_t r = 0; r < NCHt; ++r) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_reserve_back(cb_a, 1);
            cb_reserve_back(cb_b, 1);
            const uint64_t off = static_cast<uint64_t>(r * Wt + wt) * kTileBytes;
            noc_async_read(a_noc + off, get_write_ptr(cb_a), kTileBytes);
            noc_async_read(b_noc + off, get_write_ptr(cb_b), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
            cb_push_back(cb_b, 1);
        }
    }
}
