// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for the tile_copy example.
//
// Read one bf16 32×32 tile (2048 B) from an L1 source address and push
// it into CB c_0. Because the source lives in the same core's L1 as the
// CB, no NoC traffic is needed — a plain word-by-word copy from L1 to
// the CB's write pointer is enough.
//
// Runtime args:
//   arg[0] = L1 byte address of the input tile

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in     = 0;
    constexpr uint32_t one_tile  = 1;
    constexpr uint32_t tile_words = 32 * 32 * 2 / 4;  // bf16 tile in uint32

    cb_reserve_back(cb_in, one_tile);
    uint32_t write_ptr = get_write_ptr(cb_in);

    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ptr);
    for (uint32_t i = 0; i < tile_words; ++i) {
        dst[i] = src[i];
    }

    cb_push_back(cb_in, one_tile);
}
