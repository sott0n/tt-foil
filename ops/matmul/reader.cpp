// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for matmul_dram (iter12): A-tile caching.
// Mt rows × Nt cols × Kt inner. Previously A was re-read from DRAM
// Nt times per (mt) row — for Mt=1 matmul this dominated NOC traffic.
// Now A's Kt tiles are read once per mt and held in cb_a (depth Kt);
// the compute kernel indexes them by kt for every nt iteration. B is
// still read per (kt, nt) at cb_b depth 1.
//
// Runtime args:
//   arg[0..1] = A stream NOC addr (lo, hi)
//   arg[2..3] = B stream NOC addr (lo, hi)
//   arg[4]    = Mt
//   arg[5]    = Kt
//   arg[6]    = Nt  — *per-core* output column count
//   arg[7]    = Nt_stride — row stride between B's (kt) rows, in tiles.

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

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Stage A: read Kt A tiles once into cb_a (depth Kt). Issue all
        // reads back-to-back, then single barrier — lets NOC pipeline
        // the requests instead of round-tripping per tile.
        cb_reserve_back(cb_a, Kt);
        uint32_t a_wp_base = get_write_ptr(cb_a);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint64_t a_src = a_base + (mt * Kt + kt) * kTileBytes;
            noc_async_read(a_src, a_wp_base + kt * kTileBytes, kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, Kt);

        // Stage B: per (nt), feed Kt B tiles one-at-a-time so cb_b can
        // stay at depth 1. Compute will pop each B tile after its
        // matmul_tiles call.
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(cb_b, 1);
                uint32_t b_wp = get_write_ptr(cb_b);
                uint64_t b_src = b_base + (kt * Nt_stride + nt) * kTileBytes;
                noc_async_read(b_src, b_wp, kTileBytes);
                noc_async_read_barrier();
                cb_push_back(cb_b, 1);
            }
        }
        // Compute pops cb_a (Kt tiles) at end of this mt row; reader's
        // next cb_reserve_back(cb_a, Kt) will block until that happens.
    }
}
