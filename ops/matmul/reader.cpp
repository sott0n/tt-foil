// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for matmul_dram (v5-4): same outer-product schedule as
// matmul_mnk, but the A and B tile streams live in DRAM. Each tile is
// pulled into its CB via noc_async_read from the DRAM bank's NOC
// endpoint instead of word-copy from local L1.
//
// Runtime args:
//   arg[0..1] = A stream NOC addr (lo, hi) — make_noc_dram_addr on host
//   arg[2..3] = B stream NOC addr (lo, hi)
//   arg[4]    = Mt
//   arg[5]    = Kt
//   arg[6]    = Nt

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

static inline void read_one_tile(uint32_t cb, uint64_t src_noc_addr) {
    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    cb_reserve_back(cb, 1);
    uint32_t write_ptr = get_write_ptr(cb);
    noc_async_read(src_noc_addr, write_ptr, kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void kernel_main() {
    uint64_t a_base = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t b_base = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint32_t Mt     = get_arg_val<uint32_t>(4);
    uint32_t Kt     = get_arg_val<uint32_t>(5);
    uint32_t Nt     = get_arg_val<uint32_t>(6);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                read_one_tile(0, a_base + (mt * Kt + kt) * kTileBytes);
                read_one_tile(1, b_base + (kt * Nt + nt) * kTileBytes);
            }
        }
    }
}
