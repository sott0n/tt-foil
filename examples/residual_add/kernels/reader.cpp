// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for residual_add: pulls RA_NT bf16 tiles from each of two
// DRAM streams (A, B) into CB c_0 and CB c_1, one tile at a time. Same
// per-tile noc_async_read + barrier pattern as matmul_dram.
//
// Runtime args:
//   arg[0..1] = A stream NOC addr (lo, hi)   — DRAM bank base
//   arg[2..3] = B stream NOC addr (lo, hi)   — DRAM bank base
//
// Tile count comes from -DRA_NT at compile time so reader/compute/writer
// stay in lock-step without runtime drift.

#include <cstdint>

#include "dataflow_api.h"

#if !defined(RA_NT)
#error "residual_add reader: define RA_NT (tile count) at compile time"
#endif

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

static inline void read_one_tile(uint32_t cb, uint64_t src_noc_addr) {
    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    cb_reserve_back(cb, 1);
    uint32_t write_ptr = get_write_ptr(cb);
    noc_async_read(src_noc_addr, write_ptr, kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void kernel_main() {
    uint64_t a_base = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t b_base = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));

    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    constexpr uint32_t Nt = RA_NT;

    for (uint32_t i = 0; i < Nt; ++i) {
        read_one_tile(0, a_base + i * kTileBytes);
        read_one_tile(1, b_base + i * kTileBytes);
    }
}
