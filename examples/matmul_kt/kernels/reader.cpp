// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for matmul_kt: pushes Kt A tiles and Kt B tiles into
// CB c_0 / c_1 respectively, one matmul step at a time.
//
// Layout: a_buf holds Kt contiguous 32×32 bf16 tiles (each in 4-face
// layout); b_buf likewise.
//
// Runtime args:
//   arg[0] = L1 byte address of A tile stream
//   arg[1] = L1 byte address of B tile stream
//   arg[2] = Kt (number of tiles in K dimension)

#include <cstdint>

#include "dataflow_api.h"

static inline void push_one_tile(uint32_t cb, uint32_t src_addr) {
    constexpr uint32_t tile_words = 32 * 32 * 2 / 4;  // bf16 tile in uint32
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
    uint32_t kt     = get_arg_val<uint32_t>(2);

    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    for (uint32_t i = 0; i < kt; ++i) {
        push_one_tile(0, a_addr + i * kTileBytes);
        push_one_tile(1, b_addr + i * kTileBytes);
    }
}
