// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for bias_relu_post — pulls an N-tile activation stream
// from DRAM into CB c_0 and re-reads the same 1-tile bias from DRAM
// into CB c_1 once per output tile (depth-1 ring, same address each
// time). The bias tile is laid out per-channel in column 0 (rows 0..31
// carry the channel biases, cols 1..31 are unused); the compute kernel
// uses BroadcastType::COL to fan it across the 32 spatial slots of
// each (Mt × Nt) output tile.
//
// Runtime args:
//   arg[0..1] = activation DRAM NOC addr (lo, hi)
//   arg[2..3] = bias       DRAM NOC addr (lo, hi)
//   arg[4]    = N tiles

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t in_base   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t bias_base = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint32_t n         = get_arg_val<uint32_t>(4);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        noc_async_read(in_base   + i * kTileBytes, get_write_ptr(0), kTileBytes);
        noc_async_read(bias_base,                  get_write_ptr(1), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
    }
}
