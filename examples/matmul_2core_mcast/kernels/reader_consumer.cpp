// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC consumer reader for matmul_2core_mcast (v6-3).
//
// 1. Busy-waits until the producer core sets the ready flag (via NOC
//    write).  By the time the flag is observed, the producer's staging
//    NOC writes have already been barrier-flushed, so the consumer's
//    local staging buffer holds the full Kt × Nt B tile substream.
// 2. Iterates the matmul outer-product schedule, pulling A from DRAM
//    and B from local staging — no DRAM reads for B.
//
// Runtime args:
//   arg[0..1] = A NOC addr (lo, hi)               — DRAM
//   arg[2]    = local B staging L1 byte addr      — this core's L1
//   arg[3]    = local flag L1 byte addr           — this core's L1
//   arg[4]    = Mt
//   arg[5]    = Kt
//   arg[6]    = Nt

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t a_dram_base    = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint32_t local_b_staging = get_arg_val<uint32_t>(2);
    uint32_t local_flag      = get_arg_val<uint32_t>(3);
    uint32_t Mt              = get_arg_val<uint32_t>(4);
    uint32_t Kt              = get_arg_val<uint32_t>(5);
    uint32_t Nt              = get_arg_val<uint32_t>(6);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    // ---- Wait for producer ----------------------------------------
    volatile tt_l1_ptr uint32_t* flag_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_flag);
    while (*flag_ptr == 0) {
        // spin; volatile read forces refetch from L1 SRAM.
    }

    // ---- Outer-product schedule -----------------------------------
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // A from DRAM
                cb_reserve_back(0, 1);
                uint32_t a_wp = get_write_ptr(0);
                noc_async_read(a_dram_base + (mt * Kt + kt) * kTileBytes,
                               a_wp, kTileBytes);
                noc_async_read_barrier();
                cb_push_back(0, 1);

                // B from local staging at (kt*Nt + nt) tile offset
                cb_reserve_back(1, 1);
                uint32_t b_wp = get_write_ptr(1);
                volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        local_b_staging + (kt * Nt + nt) * kTileBytes);
                volatile tt_l1_ptr uint32_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_wp);
                constexpr uint32_t kTileWords = kTileBytes / 4;
                for (uint32_t w = 0; w < kTileWords; ++w) dst[w] = src[w];
                cb_push_back(1, 1);
            }
        }
    }
}
