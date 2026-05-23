// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for fused RMSNorm + RoPE.
// Drains pairs of (out_first, out_second) tiles from cb_out, writing them
// to the packed multi-head buffer at the correct positions.
//
// Output layout: [St, num_heads * Dt]  where Dt = 2 * Dt_half.
// Per (st, h, dh):
//   out_first  → out_noc[st, h*Dt + dh]
//   out_second → out_noc[st, h*Dt + Dt_half + dh]
//
// Runtime args:
//   arg[0..1] = out DRAM NOC (lo, hi)
//   arg[2]    = St
//   arg[3]    = num_heads
//   arg[4]    = Dt_half

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t out_noc   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t St        = get_arg_val<uint32_t>(2);
    const uint32_t num_heads = get_arg_val<uint32_t>(3);
    const uint32_t Dt_half   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t Dt       = 2 * Dt_half;
    const uint32_t total_Dt = num_heads * Dt;

    for (uint32_t st = 0; st < St; ++st) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t dh = 0; dh < Dt_half; ++dh) {
                const uint64_t out0_addr =
                    out_noc + static_cast<uint64_t>(st * total_Dt + h * Dt + dh)           * kTileBytes;
                const uint64_t out1_addr =
                    out_noc + static_cast<uint64_t>(st * total_Dt + h * Dt + Dt_half + dh) * kTileBytes;

                cb_wait_front(cb_out, 1);
                noc_async_write(get_read_ptr(cb_out), out0_addr, kTileBytes);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);

                cb_wait_front(cb_out, 1);
                noc_async_write(get_read_ptr(cb_out), out1_addr, kTileBytes);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
