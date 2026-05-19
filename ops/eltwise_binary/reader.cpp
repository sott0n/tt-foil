// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for eltwise_binary ops (mul, add, …).
//
// Streams num_tiles from each of two DRAM sources into CB c_0 and CB c_1.
//
// Runtime args (BRISC RTA region):
//   arg[0..1] = A src NOC addr (lo, hi)
//   arg[2..3] = B src NOC addr (lo, hi)
//   arg[4]    = num_tiles

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t a_base    = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t b_base    = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(0, 1);
        noc_async_read(a_base + i * kTileBytes, get_write_ptr(0), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(0, 1);

        cb_reserve_back(1, 1);
        noc_async_read(b_base + i * kTileBytes, get_write_ptr(1), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(1, 1);
    }
}
