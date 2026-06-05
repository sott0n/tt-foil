// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for fused multi-head GQA attention (FLASH STREAMING form).
//
// Streams ONE key/value block at a time so the compute kernel's L1 footprint
// is O(Dt), independent of St. Per (q_head h, query row-tile qt):
//   - push Q_qt once (Dt tiles)
//   - for each causal key block j in 0..=qt: push KT column j (Dt tiles,
//     strided by St in the [num_kv*Dt, St] layout) then V row j (Dt tiles,
//     contiguous in the [St, num_kv*Dt] layout).
//
// TRADEOFF: KT/V are re-read for every qt, so DRAM read traffic is
// O(St^2 * Dt * num_q) vs the old O(St * Dt * num_q). This is the price of a
// bounded L1; prefill perf is secondary to running at long sequence length.
// cb_q/cb_kt/cb_v are double-buffered (depth 2*Dt) so reads overlap compute.
//
// The causal mask is now a single 32x32 lower-triangular tile, loaded once
// into cb_tri and kept resident (off-diagonal blocks need no mask).
//
// Runtime args (BRISC):
//   arg[0,1]  = q_noc   (lo, hi)   Q       [St, num_q*Dt]
//   arg[2,3]  = kt_noc  (lo, hi)   KT      [num_kv*Dt, St]
//   arg[4,5]  = v_noc   (lo, hi)   V       [St, num_kv*Dt]
//   arg[6,7]  = scaler_noc         scaler tile (BF16(1.0))
//   arg[8,9]  = mask_noc           tri mask (single tile)
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
    constexpr uint32_t cb_tri    = 14;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    const uint32_t gqa_groups = num_q / num_kv;
    const uint32_t total_Nq   = num_q  * Dt;          // tile-cols of Q / out
    const uint32_t total_Nk   = num_kv * Dt;          // tile-cols of V

    // ---- One-shot loads (resident across heads) ----
    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    cb_reserve_back(cb_tri, 1);
    noc_async_read(mask_noc, get_write_ptr(cb_tri), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_tri, 1);

    for (uint32_t h = 0; h < num_q; ++h) {
        const uint32_t kv = h / gqa_groups;

        for (uint32_t qt = 0; qt < St; ++qt) {
            // Q_qt tiles: (qt, h*Dt + dt) in [St, num_q*Dt]
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

            for (uint32_t j = 0; j <= qt; ++j) {
                // KT column j tiles: (kv*Dt + k, j) in [num_kv*Dt, St] — strided by St.
                cb_reserve_back(cb_kt, Dt);
                {
                    uint32_t base = get_write_ptr(cb_kt);
                    for (uint32_t k = 0; k < Dt; ++k) {
                        uint64_t src = kt_noc + (uint64_t)((kv * Dt + k) * St + j) * kTileBytes;
                        noc_async_read(src, base + k * kTileBytes, kTileBytes);
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_kt, Dt);

                // V row j tiles: (j, kv*Dt + dt) in [St, num_kv*Dt] — contiguous.
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
            }
        }
    }
}
