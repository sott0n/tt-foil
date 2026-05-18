// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for bias_relu_post — pulls an (Mt × Nt) activation tile
// stream from DRAM into CB c_0, and matches each tile with the right
// bias tile in CB c_1. The bias DRAM region holds Mt back-to-back tiles
// (channel biases in column 0, cols 1..31 unused); output tile (mt, nt)
// gets bias[mt], so we share the bias across all Nt spatial-slot
// tiles per Mt row. BroadcastType::COL fans bias[mt][:, 0] across the
// 32 spatial slots of each output tile.
//
// Runtime args:
//   arg[0..1] = activation DRAM NOC addr (lo, hi)
//   arg[2..3] = bias       DRAM NOC addr (lo, hi)
//   arg[4]    = Mt   (rows of output tile grid, also # of bias tiles)
//   arg[5]    = Nt   (cols of output tile grid)

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t in_base   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t bias_base = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint32_t Mt        = get_arg_val<uint32_t>(4);
    uint32_t Nt        = get_arg_val<uint32_t>(5);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    uint32_t i = 0;
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            noc_async_read(in_base   + i  * kTileBytes, get_write_ptr(0), kTileBytes);
            noc_async_read(bias_base + mt * kTileBytes, get_write_ptr(1), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(0, 1);
            cb_push_back(1, 1);
            ++i;
        }
    }
}
