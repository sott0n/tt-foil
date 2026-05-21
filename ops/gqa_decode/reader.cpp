// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused multi-head GQA decode attention.
//
// Layouts (Q has just St_q = 1 query tile-row; K/V are the full cache):
//   Q   [St_q, num_q  * Dt]
//   KT  [num_kv * Dt, St_kv]
//   V   [St_kv, num_kv * Dt]
//   mask[St_q, St_kv]                     — 0/1 BF16, lets the kernel
//                                            ignore padding positions in
//                                            the last cache tile.
//
// Runtime args (BRISC):
//   arg[0,1]  = q_noc
//   arg[2,3]  = kt_noc
//   arg[4,5]  = v_noc
//   arg[6,7]  = scaler_noc
//   arg[8,9]  = mask_noc
//   arg[10]   = St_q
//   arg[11]   = St_kv
//   arg[12]   = Dt
//   arg[13]   = num_q
//   arg[14]   = num_kv

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
    const uint32_t St_q       = get_arg_val<uint32_t>(10);
    const uint32_t St_kv      = get_arg_val<uint32_t>(11);
    const uint32_t Dt         = get_arg_val<uint32_t>(12);
    const uint32_t num_q      = get_arg_val<uint32_t>(13);
    const uint32_t num_kv     = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_q      = 0;
    constexpr uint32_t cb_kt     = 1;
    constexpr uint32_t cb_v      = 2;
    constexpr uint32_t cb_reduce = 3;
    constexpr uint32_t cb_mask   = 10;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t gqa_groups = num_q / num_kv;
    const uint32_t q_tiles    = St_q  * Dt;     // per-head Q
    const uint32_t kt_tiles   = Dt    * St_kv;  // per-head KT
    const uint32_t v_tiles    = St_kv * Dt;     // per-head V
    const uint32_t mask_tiles = St_q  * St_kv;
    const uint32_t total_Nq   = num_q  * Dt;
    const uint32_t total_Nk   = num_kv * Dt;

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

    for (uint32_t h = 0; h < num_q; ++h) {
        const uint32_t kv = h / gqa_groups;

        // Q_h tiles: (st, h*Dt + dt) in [St_q, num_q*Dt]
        cb_reserve_back(cb_q, q_tiles);
        {
            uint32_t base = get_write_ptr(cb_q);
            uint32_t idx  = 0;
            for (uint32_t st = 0; st < St_q; ++st) {
                for (uint32_t dt = 0; dt < Dt; ++dt, ++idx) {
                    uint64_t src = q_noc + (st * total_Nq + h * Dt + dt) * kTileBytes;
                    noc_async_read(src, base + idx * kTileBytes, kTileBytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_q, q_tiles);

        // KT_h tiles: contiguous Dt*St_kv block in [num_kv*Dt, St_kv]
        cb_reserve_back(cb_kt, kt_tiles);
        {
            uint32_t base    = get_write_ptr(cb_kt);
            uint64_t src_blk = kt_noc + (kv * Dt) * St_kv * kTileBytes;
            for (uint32_t i = 0; i < kt_tiles; ++i) {
                noc_async_read(src_blk + i * kTileBytes, base + i * kTileBytes, kTileBytes);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_kt, kt_tiles);

        // V_h tiles: (st, kv*Dt + dt) in [St_kv, num_kv*Dt]
        cb_reserve_back(cb_v, v_tiles);
        {
            uint32_t base = get_write_ptr(cb_v);
            uint32_t idx  = 0;
            for (uint32_t st = 0; st < St_kv; ++st) {
                for (uint32_t dt = 0; dt < Dt; ++dt, ++idx) {
                    uint64_t src = v_noc + (st * total_Nk + kv * Dt + dt) * kTileBytes;
                    noc_async_read(src, base + idx * kTileBytes, kTileBytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_v, v_tiles);
    }
}
