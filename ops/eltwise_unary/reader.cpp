// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for eltwise_unary ops (SiLU, GELU, ReLU, …).
//
// Streams num_tiles bf16 32×32 tiles from DRAM into CB c_0 one at a time.
// The matching compute kernel (compute_<op>.cpp) reads from c_0 and pushes
// results to c_16; the writer drains c_16 back to DRAM.
//
// Runtime args (BRISC RTA region):
//   arg[0] = src NOC addr lo  (from make_noc_dram_addr on host)
//   arg[1] = src NOC addr hi
//   arg[2] = num_tiles

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t src_base  = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in     = 0;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(src_base + i * kTileBytes, get_write_ptr(cb_in), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
