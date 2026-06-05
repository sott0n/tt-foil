// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused multi-head GQA decode attention (FLASH STREAMING).
//
// Streams one key/value block at a time so the compute kernel's L1 is O(Dt),
// independent of the cache depth St_kv. Per (q_head h, query row-tile qt):
//   - push Q_qt once (Dt tiles)
//   - for each cache block j in 0..St_kv: push KT block j (Dt tiles), V block j
//     (Dt tiles), and the padding-mask tile (qt, j) (1 tile).
//
// KT and V caches are both slot-major [St_kv, num_kv*Dt]: block j occupies a
// contiguous num_kv*Dt tile run, so KT/V block j tiles are at
// (j*total_Nk + kv*Dt + d). (This preserves the per-(k) tile pairing the old
// non-streamed decode kernel used.)
//
// Runtime args (BRISC):
//   arg[0,1]  = q_noc
//   arg[2,3]  = kt_noc
//   arg[4,5]  = v_noc
//   arg[6,7]  = scaler_noc
//   arg[8,9]  = mask_noc        [St_q, St_kv] padding mask
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
    constexpr uint32_t cb_mask   = 14;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t gqa_groups = num_q / num_kv;
    const uint32_t total_Nq   = num_q  * Dt;
    const uint32_t total_Nk   = num_kv * Dt;

    // ---- One-shot scaler (resident) ----
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    for (uint32_t h = 0; h < num_q; ++h) {
        const uint32_t kv = h / gqa_groups;

        for (uint32_t qt = 0; qt < St_q; ++qt) {
            // Q_qt tiles: (qt, h*Dt + dt) in [St_q, num_q*Dt]
            cb_reserve_back(cb_q, Dt);
            {
                uint32_t base = get_write_ptr(cb_q);
                for (uint32_t dt = 0; dt < Dt; ++dt) {
                    uint64_t src = q_noc + (uint64_t)(qt * total_Nq + h * Dt + dt) * kTileBytes;
                    noc_async_read(src, base + dt * kTileBytes, kTileBytes);
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_q, Dt);

            for (uint32_t j = 0; j < St_kv; ++j) {
                // KT block j: cache tiles (j, kv*Dt + k) in [St_kv, num_kv*Dt].
                cb_reserve_back(cb_kt, Dt);
                {
                    uint32_t base = get_write_ptr(cb_kt);
                    for (uint32_t k = 0; k < Dt; ++k) {
                        uint64_t src = kt_noc + (uint64_t)(j * total_Nk + kv * Dt + k) * kTileBytes;
                        noc_async_read(src, base + k * kTileBytes, kTileBytes);
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_kt, Dt);

                // V block j: cache tiles (j, kv*Dt + dt) in [St_kv, num_kv*Dt].
                cb_reserve_back(cb_v, Dt);
                {
                    uint32_t base = get_write_ptr(cb_v);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        uint64_t src = v_noc + (uint64_t)(j * total_Nk + kv * Dt + dt) * kTileBytes;
                        noc_async_read(src, base + dt * kTileBytes, kTileBytes);
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_v, Dt);

                // Mask tile (qt, j) in [St_q, St_kv].
                cb_reserve_back(cb_mask, 1);
                noc_async_read(mask_noc + (uint64_t)(qt * St_kv + j) * kTileBytes,
                               get_write_ptr(cb_mask), kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_mask, 1);
            }
        }
    }
}
