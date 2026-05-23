// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for fused add+RMSNorm: drains TWO output streams.
//   Per row r:
//     1. cb_s_out (Wt tiles)  — residual sum S = A+B  →  S DRAM
//     2. cb_out   (Wt tiles)  — normed RMSNorm(S)·γ  →  Y DRAM
//
// Runtime args:
//   arg[0..1] = S dst DRAM NOC (lo, hi)  (sum output)
//   arg[2..3] = Y dst DRAM NOC (lo, hi)  (normed output)
//   arg[4]    = NCHt
//   arg[5]    = Wt

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t s_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t y_noc = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint32_t NCHt  = get_arg_val<uint32_t>(4);
    const uint32_t Wt    = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_s_out   = 10;
    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t r = 0; r < NCHt; ++r) {
        // Drain sum row (Wt tiles arrive together from phase 0).
        cb_wait_front(cb_s_out, Wt);
        uint32_t s_rp = get_read_ptr(cb_s_out);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            const uint64_t off = static_cast<uint64_t>(r * Wt + wt) * kTileBytes;
            noc_async_write(s_rp + wt * kTileBytes, s_noc + off, kTileBytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_s_out, Wt);

        // Drain normed row (compute phase 5 pushes tile-by-tile).
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_out, 1);
            const uint64_t off = static_cast<uint64_t>(r * Wt + wt) * kTileBytes;
            noc_async_write(get_read_ptr(cb_out), y_noc + off, kTileBytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
