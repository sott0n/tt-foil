// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for eltwise_unary ops (SiLU, GELU, ReLU, …).
//
// Drains num_tiles bf16 tiles from CB c_16 to a contiguous DRAM region.
//
// Runtime args (NCRISC RTA region):
//   arg[0] = dst NOC addr lo  (from make_noc_dram_addr on host)
//   arg[1] = dst NOC addr hi
//   arg[2] = num_tiles

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t dst_base  = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out    = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), dst_base + i * kTileBytes, kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
