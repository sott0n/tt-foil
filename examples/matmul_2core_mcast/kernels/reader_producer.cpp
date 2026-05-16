// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC producer reader for matmul_2core_mcast (v6-3).
//
// 1. Reads the full B sub-stream (Kt × Nt tiles, kt-major) from DRAM
//    into a local L1 staging buffer.
// 2. NOC-writes the staging buffer + a sync flag to the consumer
//    core's L1 so its reader can proceed without hitting DRAM.
// 3. Iterates the matmul outer-product schedule pulling A from DRAM
//    and B from the local staging buffer.
//
// Runtime args:
//   arg[0..1]  = A NOC addr (lo, hi)              — DRAM
//   arg[2..3]  = B NOC addr (lo, hi)              — DRAM
//   arg[4..5]  = consumer B staging NOC addr (lo, hi) — peer core L1
//   arg[6..7]  = consumer flag NOC addr (lo, hi)      — peer core L1
//   arg[8]     = local B staging L1 byte addr     — this core's L1
//   arg[9]     = local flag-source L1 byte addr   — value 1 pre-stored
//   arg[10]    = Mt
//   arg[11]    = Kt
//   arg[12]    = Nt
//   arg[13]    = ckpt L1 byte addr                — debug checkpoints

#include <cstdint>

#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t a_dram_base     = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint64_t b_dram_base     = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    uint64_t peer_b_staging  = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    uint64_t peer_flag       = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    uint32_t local_b_staging = get_arg_val<uint32_t>(8);
    uint32_t local_flag_src  = get_arg_val<uint32_t>(9);
    uint32_t Mt              = get_arg_val<uint32_t>(10);
    uint32_t Kt              = get_arg_val<uint32_t>(11);
    uint32_t Nt              = get_arg_val<uint32_t>(12);
    uint32_t ckpt            = get_arg_val<uint32_t>(13);

    volatile tt_l1_ptr uint32_t* ck =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ckpt);
    ck[0] = 0x11111111;

    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    const uint32_t kNumBTiles = Kt * Nt;

    // ---- Phase 1: load Kt*Nt B tiles from DRAM into staging --------
    ck[1] = 0x22222222;
    for (uint32_t i = 0; i < kNumBTiles; ++i) {
        noc_async_read(b_dram_base + i * kTileBytes,
                       local_b_staging + i * kTileBytes,
                       kTileBytes);
    }
    noc_async_read_barrier();
    ck[2] = 0x33333333;

    // ---- Phase 2: NOC-forward staging + ready flag to consumer ------
    // BRISC's default NOC 0 is the read direction; writes to peer
    // Tensix L1 over NOC 0 don't return ACKs reliably on Blackhole
    // (verified empirically: 4 B writes flake, multi-word writes hang).
    // NOC 1 (NCRISC's default writer NOC) handles them correctly.
    //
    // Barrier per tile keeps the NOC 1 outstanding queue bounded.
    constexpr uint8_t kWriteNoc = 1;
    for (uint32_t i = 0; i < kNumBTiles; ++i) {
        noc_async_write_one_packet(local_b_staging + i * kTileBytes,
                                   peer_b_staging  + i * kTileBytes,
                                   kTileBytes,
                                   kWriteNoc);
        noc_async_write_barrier(kWriteNoc);
    }
    ck[3] = 0x77777777;
    // Flag write happens AFTER all staging writes so the consumer
    // never observes flag=ready with stale/partial staging bytes.
    noc_async_write_one_packet(local_flag_src, peer_flag,
                               sizeof(uint32_t), kWriteNoc);
    noc_async_write_barrier(kWriteNoc);
    ck[15] = 0x88888888;

    // ---- Phase 3: outer-product schedule using local staging --------
    // A from DRAM, B from staging at offset (kt*Nt + nt).
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(0, 1);
                uint32_t a_wp = get_write_ptr(0);
                noc_async_read(a_dram_base + (mt * Kt + kt) * kTileBytes,
                               a_wp, kTileBytes);
                noc_async_read_barrier();
                cb_push_back(0, 1);

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
