// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for the weight-stationary matmul (prefill). Unlike
// ops/matmul/reader.cpp (which re-reads the whole B weight slice once per
// mt-row → Mt× DRAM traffic), this caches a BLOCK of Mb A-rows in cb_a and
// reads each B column exactly once per (block, nt). The compute kernel reuses
// that B column across all Mb rows of the block, so B is streamed only
// ceil(Mt/Mb)× from DRAM.
//
// cb_a holds Mb*Kt tiles (host sizes it). cb_b holds Kt tiles (one column).
// With Mb=1 this is byte-for-byte the ops/matmul behavior.
//
// Runtime args:
//   arg[0..1] = A stream NOC addr (lo, hi)
//   arg[2..3] = B stream NOC addr (lo, hi)
//   arg[4]    = Mt
//   arg[5]    = Kt
//   arg[6]    = Nt        — per-core output column count
//   arg[7]    = Nt_stride — B row stride between (kt) rows, in tiles
//   arg[8]    = Mb        — A-row block height (1..Mt)

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t a_base   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t b_base   = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint32_t Mt        = get_arg_val<uint32_t>(4);
    uint32_t Kt        = get_arg_val<uint32_t>(5);
    uint32_t Nt        = get_arg_val<uint32_t>(6);
    uint32_t Nt_stride = get_arg_val<uint32_t>(7);
    uint32_t Mb        = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t mt0 = 0; mt0 < Mt; mt0 += Mb) {
        const uint32_t mb = (mt0 + Mb <= Mt) ? Mb : (Mt - mt0);

        // Cache this block's mb A-rows (mb*Kt tiles), one batched barrier.
        cb_reserve_back(cb_a, mb * Kt);
        uint32_t a_wp = get_write_ptr(cb_a);
        for (uint32_t r = 0; r < mb; ++r) {
            const uint32_t mt = mt0 + r;
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                uint64_t a_src = a_base + (mt * Kt + kt) * kTileBytes;
                noc_async_read(a_src, a_wp + (r * Kt + kt) * kTileBytes, kTileBytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, mb * Kt);

        // Read each B column once; compute reuses it across the mb rows.
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_reserve_back(cb_b, Kt);
            uint32_t b_wp = get_write_ptr(cb_b);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                uint64_t b_src = b_base + (kt * Nt_stride + nt) * kTileBytes;
                noc_async_read(b_src, b_wp + kt * kTileBytes, kTileBytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_b, Kt);
        }
    }
}
