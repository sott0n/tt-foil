// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for MHA (with mask).
//
// Loads (all upfront — only valid for small St, Dt that fit in L1):
//   - scaler tile (cb_reduce, BF16(1.0))
//   - Q     [St × Dt]  tiles (cb_q)
//   - KT    [Dt × St]  tiles (cb_kt)
//   - V     [St × Dt]  tiles (cb_v)
//   - mask  [St × St]  tiles (cb_mask, BF16 0/1)
//
// Runtime args:
//   arg[0..1]  = Q_noc
//   arg[2..3]  = KT_noc
//   arg[4..5]  = V_noc
//   arg[6..7]  = scaler_noc
//   arg[8..9]  = mask_noc
//   arg[10]    = St
//   arg[11]    = Dt

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t q_noc      = join64(get_arg_val<uint32_t>(0),  get_arg_val<uint32_t>(1));
    const uint64_t kt_noc     = join64(get_arg_val<uint32_t>(2),  get_arg_val<uint32_t>(3));
    const uint64_t v_noc      = join64(get_arg_val<uint32_t>(4),  get_arg_val<uint32_t>(5));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(6),  get_arg_val<uint32_t>(7));
    const uint64_t mask_noc   = join64(get_arg_val<uint32_t>(8),  get_arg_val<uint32_t>(9));
    const uint32_t St         = get_arg_val<uint32_t>(10);
    const uint32_t Dt         = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_q      = 0;
    constexpr uint32_t cb_kt     = 1;
    constexpr uint32_t cb_v      = 2;
    constexpr uint32_t cb_reduce = 3;
    constexpr uint32_t cb_mask   = 10;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    const uint32_t qkv_tiles  = St * Dt;
    const uint32_t kt_tiles   = Dt * St;
    const uint32_t mask_tiles = St * St;

    cb_reserve_back(cb_q, qkv_tiles);
    for (uint32_t i = 0; i < qkv_tiles; ++i) {
        noc_async_read(q_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_q) + i * kTileBytes, kTileBytes);
    }

    cb_reserve_back(cb_kt, kt_tiles);
    for (uint32_t i = 0; i < kt_tiles; ++i) {
        noc_async_read(kt_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_kt) + i * kTileBytes, kTileBytes);
    }

    cb_reserve_back(cb_v, qkv_tiles);
    for (uint32_t i = 0; i < qkv_tiles; ++i) {
        noc_async_read(v_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_v) + i * kTileBytes, kTileBytes);
    }

    cb_reserve_back(cb_mask, mask_tiles);
    for (uint32_t i = 0; i < mask_tiles; ++i) {
        noc_async_read(mask_noc + static_cast<uint64_t>(i) * kTileBytes,
                       get_write_ptr(cb_mask) + i * kTileBytes, kTileBytes);
    }

    noc_async_read_barrier();
    cb_push_back(cb_q,    qkv_tiles);
    cb_push_back(cb_kt,   kt_tiles);
    cb_push_back(cb_v,    qkv_tiles);
    cb_push_back(cb_mask, mask_tiles);
}
