// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for matmul_mnk (v5-3): outer-product feed for an
// Mt×Kt × Kt×Nt matmul.
//
// For each output tile (mt, nt), pushes the Kt A tiles A[mt, 0..Kt) and
// the Kt B tiles B[0..Kt, nt]. Same K-tile is re-read for every nt of
// the same mt; this is the simplest correct schedule.
//
// Layouts in L1:
//   a_buf : Mt rows × Kt cols of tiles, row-major (A[mt,kt] at offset
//            (mt*Kt + kt) * kTileBytes)
//   b_buf : Kt rows × Nt cols of tiles, row-major (B[kt,nt] at offset
//            (kt*Nt + nt) * kTileBytes)
//
// Runtime args:
//   arg[0] = a_buf L1 byte address
//   arg[1] = b_buf L1 byte address
//   arg[2] = Mt
//   arg[3] = Kt
//   arg[4] = Nt

#include <cstdint>

#include "dataflow_api.h"

static inline void push_one_tile(uint32_t cb, uint32_t src_addr) {
    constexpr uint32_t tile_words = 32 * 32 * 2 / 4;
    cb_reserve_back(cb, 1);
    uint32_t write_ptr = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ptr);
    for (uint32_t i = 0; i < tile_words; ++i) {
        dst[i] = src[i];
    }
    cb_push_back(cb, 1);
}

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt     = get_arg_val<uint32_t>(2);
    uint32_t Kt     = get_arg_val<uint32_t>(3);
    uint32_t Nt     = get_arg_val<uint32_t>(4);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                push_one_tile(0, a_addr + (mt * Kt + kt) * kTileBytes);
                push_one_tile(1, b_addr + (kt * Nt + nt) * kTileBytes);
            }
        }
    }
}
