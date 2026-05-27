// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC kernel for KvSnapshot. Replaces qwen3vl_run's host-side
// `pre:kv_cache_snapshot(host)` step (28 layers × 8 small PCIe
// transactions/layer = ~455 ms of UMD per-transaction overhead) with
// an all-on-device prefill snapshot.
//
// Inputs (block-major K^T already produced by `pre:transpose`, plus
// slot-major V):
//   T_Kt_pre  — shape (Mt=kNkDt, Nt=kSt) → tile (c, s) at offset
//               (c * kSt + s) * 2048. Tile content = K^T (rows=heads,
//               cols=seqs within the block).
//   T_V_pre   — shape (Mt=kSt, Nt=kNkDt) → tile (s, c) at offset
//               (s * kNkDt + c) * 2048. Tile content = V (rows=seqs,
//               cols=V heads within the block).
//
// Outputs (slot-major, matches kv_append / gqa_decode expectations):
//   T_Kt_cache — tile (slot, c) at offset (slot * kNkDt + c) * 2048,
//                slots 0..kSt-1 hold the prefill K^T; slots kSt..kStKv-1
//                zeroed (so gqa_decode's softmax doesn't see garbage).
//   T_V_cache  — same shape; slots 0..kSt-1 hold prefill V, rest zero.
//
// Tile-level moves:
//   for (s, c) in [0, kSt) × [0, kNkDt):
//     T_Kt_cache[(s * kNkDt + c) * 2048] ← T_Kt_pre[(c * kSt + s) * 2048]
//     T_V_cache [(s * kNkDt + c) * 2048] ← T_V_pre [(s * kNkDt + c) * 2048]
//   for s in [kSt, kStKv), c in [0, kNkDt):
//     T_Kt_cache[(s * kNkDt + c) * 2048] ← zero tile
//     T_V_cache [(s * kNkDt + c) * 2048] ← zero tile
//
// RTAs:
//   arg[ 0.. 1] = T_Kt_pre  base NOC (lo, hi)
//   arg[ 2.. 3] = T_V_pre   base NOC (lo, hi)
//   arg[ 4.. 5] = T_Kt_cache base NOC (lo, hi)
//   arg[ 6.. 7] = T_V_cache  base NOC (lo, hi)
//   arg[    8 ] = kSt    (prefill slot-tile count)
//   arg[    9 ] = kNkDt  (column-tile count = num_kv_heads)
//   arg[   10 ] = kStKv  (total cache slot-tile count = kSt + decode slots)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

constexpr uint32_t cb_io      = 16;
constexpr uint32_t kTileBytes = 2048;

void kernel_main() {
    const uint64_t kt_src   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t v_src    = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t kt_dst   = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint64_t v_dst    = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    const uint32_t kSt      = get_arg_val<uint32_t>(8);
    const uint32_t kNkDt    = get_arg_val<uint32_t>(9);
    const uint32_t kStKv    = get_arg_val<uint32_t>(10);

    const uint32_t kt_bytes = kNkDt * kSt * kTileBytes;
    const uint32_t v_bytes  = kSt * kNkDt * kTileBytes;

    cb_reserve_back(cb_io, 1);
    const uint32_t l1_base   = get_write_ptr(cb_io);
    const uint32_t kt_buf    = l1_base;
    const uint32_t v_buf     = l1_base + kt_bytes;
    const uint32_t zero_buf  = l1_base + kt_bytes + v_bytes;

    // Slurp both source tensors into L1 in one read each.
    noc_async_read(kt_src, kt_buf, kt_bytes);
    noc_async_read(v_src,  v_buf,  v_bytes);
    noc_async_read_barrier();

    // Build a 2 KB zero tile in L1 for the decode-slot fill.
    {
        volatile uint32_t* z = reinterpret_cast<volatile uint32_t*>(zero_buf);
        for (uint32_t i = 0; i < kTileBytes / 4; ++i) z[i] = 0u;
    }

    // ----- Prefill slots: re-position K^T tiles, copy V tiles ---------
    // K^T source is block-major (tile (c, s) at (c * kSt + s) * 2048);
    // destination is slot-major (tile (s, c) at (s * kNkDt + c) * 2048).
    // V is slot-major in both, so the per-tile offset stays the same.
    for (uint32_t s = 0; s < kSt; ++s) {
        for (uint32_t c = 0; c < kNkDt; ++c) {
            const uint32_t src_kt_off = (c * kSt + s) * kTileBytes;
            const uint32_t src_v_off  = (s * kNkDt + c) * kTileBytes;
            const uint64_t dst_off    = (uint64_t)(s * kNkDt + c) * kTileBytes;

            noc_async_write(kt_buf + src_kt_off, kt_dst + dst_off, kTileBytes);
            noc_async_write(v_buf  + src_v_off,  v_dst  + dst_off, kTileBytes);
        }
    }

    // ----- Decode slots: zero --------------------------------------------
    // Single L1 zero tile re-used for every destination slot.
    for (uint32_t s = kSt; s < kStKv; ++s) {
        for (uint32_t c = 0; c < kNkDt; ++c) {
            const uint64_t dst_off = (uint64_t)(s * kNkDt + c) * kTileBytes;
            noc_async_write(zero_buf, kt_dst + dst_off, kTileBytes);
            noc_async_write(zero_buf, v_dst  + dst_off, kTileBytes);
        }
    }

    noc_async_write_barrier();
    cb_push_back(cb_io, 1);
}
