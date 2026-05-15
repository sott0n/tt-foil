// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for matmul_1tile: pushes one bf16 32×32 A tile into CB c_0
// and one bf16 32×32 B tile into CB c_1, both from local L1.
//
// Runtime args:
//   arg[0] = L1 byte address of input tile A
//   arg[1] = L1 byte address of input tile B

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
    push_one_tile(0, a_addr);
    push_one_tile(1, b_addr);
}
