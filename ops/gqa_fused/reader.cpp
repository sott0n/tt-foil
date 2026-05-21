// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused multi-head GQA attention.
//
// Loads the full Q [St, num_q*Dt], KT [num_kv*Dt, St], V [St, num_kv*Dt],
// streaming each q_head's Q_h / KT_h / V_h tiles into cb_q / cb_kt / cb_v
// in order so the compute kernel can run the per-head attention math
// num_q times without re-launching the kernel.
//
// Mask and scaler are loaded once upfront and stay resident in their CBs
// across all heads.
//
// Runtime args (BRISC):
//   arg[0,1]  = q_noc   (lo, hi)   Q       [St, num_q*Dt]
//   arg[2,3]  = kt_noc  (lo, hi)   KT      [num_kv*Dt, St]
//   arg[4,5]  = v_noc   (lo, hi)   V       [St, num_kv*Dt]
//   arg[6,7]  = scaler_noc         scaler tile (BF16(1.0))
//   arg[8,9]  = mask_noc           mask    [St, St]
//   arg[10]   = St
//   arg[11]   = Dt
//   arg[12]   = num_q
//   arg[13]   = num_kv

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void kernel_main() {
    const uint64_t q_noc      = join64(get_arg_val<uint32_t>(0),  get_arg_val<uint32_t>(1));
    const uint64_t kt_noc     = join64(get_arg_val<uint32_t>(2),  get_arg_val<uint32_t>(3));
    const uint64_t v_noc      = join64(get_arg_val<uint32_t>(4),  get_arg_val<uint32_t>(5));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(6),  get_arg_val<uint32_t>(7));
    const uint64_t mask_noc   = join64(get_arg_val<uint32_t>(8),  get_arg_val<uint32_t>(9));
    const uint32_t St         = get_arg_val<uint32_t>(10);
    const uint32_t Dt         = get_arg_val<uint32_t>(11);
    const uint32_t num_q      = get_arg_val<uint32_t>(12);
    const uint32_t num_kv     = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_q      = 0;
    constexpr uint32_t cb_kt     = 1;
    constexpr uint32_t cb_v      = 2;
    constexpr uint32_t cb_reduce = 3;
    constexpr uint32_t cb_mask   = 10;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t gqa_groups = num_q / num_kv;
    const uint32_t qkv_tiles  = St * Dt;
    const uint32_t kt_tiles   = Dt * St;
    const uint32_t mask_tiles = St * St;
    const uint32_t total_Nq   = num_q  * Dt;          // tile-cols of Q / out
    const uint32_t total_Nk   = num_kv * Dt;          // tile-cols of V / tile-rows of KT

    // ---- One-shot loads (persistent across heads) ----
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    cb_reserve_back(cb_mask, mask_tiles);
    {
        uint32_t base = get_write_ptr(cb_mask);
        for (uint32_t i = 0; i < mask_tiles; ++i) {
            noc_async_read(mask_noc + i * kTileBytes, base + i * kTileBytes, kTileBytes);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_mask, mask_tiles);

    // ---- Per-head streaming ----
    for (uint32_t h = 0; h < num_q; ++h) {
        const uint32_t kv = h / gqa_groups;

        // Q_h tiles: (st, h*Dt + dt) in [St, num_q*Dt] layout
        cb_reserve_back(cb_q, qkv_tiles);
        {
            uint32_t base = get_write_ptr(cb_q);
            uint32_t idx  = 0;
            for (uint32_t st = 0; st < St; ++st) {
                for (uint32_t dt = 0; dt < Dt; ++dt, ++idx) {
                    uint64_t src = q_noc + (st * total_Nq + h * Dt + dt) * kTileBytes;
                    noc_async_read(src, base + idx * kTileBytes, kTileBytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_q, qkv_tiles);

        // KT_h tiles: (kv*Dt + rt, ct) in [num_kv*Dt, St] — contiguous block.
        cb_reserve_back(cb_kt, kt_tiles);
        {
            uint32_t base    = get_write_ptr(cb_kt);
            uint64_t src_blk = kt_noc + (kv * Dt) * St * kTileBytes;
            for (uint32_t i = 0; i < kt_tiles; ++i) {
                noc_async_read(src_blk + i * kTileBytes, base + i * kTileBytes, kTileBytes);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_kt, kt_tiles);

        // V_h tiles: (st, kv*Dt + dt) in [St, num_kv*Dt] layout
        cb_reserve_back(cb_v, qkv_tiles);
        {
            uint32_t base = get_write_ptr(cb_v);
            uint32_t idx  = 0;
            for (uint32_t st = 0; st < St; ++st) {
                for (uint32_t dt = 0; dt < Dt; ++dt, ++idx) {
                    uint64_t src = v_noc + (st * total_Nk + kv * Dt + dt) * kTileBytes;
                    noc_async_read(src, base + idx * kTileBytes, kTileBytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_v, qkv_tiles);
    }
}
