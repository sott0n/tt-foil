// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for transpose_2d: drains cb_out sequentially into the
// output DRAM buffer. The reader already streamed tiles in transposed
// order, so a simple sequential write produces the [Ct, Rt] layout.
//
// Runtime args:
//   arg[0,1] = out_noc (lo, hi)
//   arg[2]   = total_tiles  (= Rt * Ct)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void kernel_main() {
    const uint64_t out_noc     = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t total_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < total_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), out_noc + i * kTileBytes, kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
