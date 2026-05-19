// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for Softmax: streams output tiles from CB16 to DRAM.
//
// Runtime args:
//   arg[0..1] = dst DRAM NOC (lo, hi)
//   arg[2]    = NCHt
//   arg[3]    = Wt

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t dst_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t NCHt    = get_arg_val<uint32_t>(2);
    const uint32_t Wt      = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t total_tiles = NCHt * Wt;
    for (uint32_t i = 0; i < total_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), dst_noc + static_cast<uint64_t>(i) * kTileBytes, kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
