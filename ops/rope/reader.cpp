// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for multi-head RoPE.
//
// Applies rotary position embedding to Q or K stored in packed multi-head
// layout [St, num_heads * Dt_per_head].  cos/sin tables have shape
// [St, Dt_half] where Dt_half = Dt_per_head / 2.
//
// For each (st, head, dt_half_idx) iteration the reader pushes:
//   cb_x0  ← x tile at column  (h*Dt + dt_half_idx)           [first-half]
//   cb_x1  ← x tile at column  (h*Dt + Dt_half + dt_half_idx) [second-half]
//   cb_cos ← cos tile at column dt_half_idx  (position block st)
//   cb_sin ← sin tile at column dt_half_idx
//
// Runtime args (BRISC):
//   arg[0,1]  = x_noc   (lo, hi)  input  [St, num_heads * Dt_per_head]
//   arg[2,3]  = cos_noc (lo, hi)  [St, Dt_half]
//   arg[4,5]  = sin_noc (lo, hi)  [St, Dt_half]
//   arg[6]    = St
//   arg[7]    = num_heads
//   arg[8]    = Dt_half  (= head_dim / 64; e.g. 1 for head_dim=64)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void kernel_main() {
    const uint64_t x_noc   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t cos_noc = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t sin_noc = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint32_t St        = get_arg_val<uint32_t>(6);
    const uint32_t num_heads = get_arg_val<uint32_t>(7);
    const uint32_t Dt_half   = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_x0  = 0;
    constexpr uint32_t cb_x1  = 1;
    constexpr uint32_t cb_cos = 2;
    constexpr uint32_t cb_sin = 3;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t Dt          = 2 * Dt_half;        // tiles per head
    const uint32_t total_Dt    = num_heads * Dt;      // tile columns in x

    for (uint32_t st = 0; st < St; ++st) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t dh = 0; dh < Dt_half; ++dh) {
                uint64_t x0_addr  = x_noc   + (st * total_Dt + h * Dt + dh)           * kTileBytes;
                uint64_t x1_addr  = x_noc   + (st * total_Dt + h * Dt + Dt_half + dh) * kTileBytes;
                uint64_t cos_addr = cos_noc + (st * Dt_half + dh)                      * kTileBytes;
                uint64_t sin_addr = sin_noc + (st * Dt_half + dh)                      * kTileBytes;

                cb_reserve_back(cb_x0, 1);
                noc_async_read(x0_addr, get_write_ptr(cb_x0), kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_x0, 1);

                cb_reserve_back(cb_x1, 1);
                noc_async_read(x1_addr, get_write_ptr(cb_x1), kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_x1, 1);

                cb_reserve_back(cb_cos, 1);
                noc_async_read(cos_addr, get_write_ptr(cb_cos), kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_cos, 1);

                cb_reserve_back(cb_sin, 1);
                noc_async_read(sin_addr, get_write_ptr(cb_sin), kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_sin, 1);
            }
        }
    }
}
