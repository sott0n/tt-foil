// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for the maxpool_2x2 example (v8.1).
//
// A 2×2 stride=2 maxpool over (H, W, C) bf16 input is implemented as a
// 4-stream reduction: the host gathers the four window positions
// (di, dj) ∈ {(0,0),(0,1),(1,0),(1,1)} into separate tile streams in L1,
// and the compute kernel takes the elementwise max across them.
//
// Each stream is a (Hout*Wout, C) = (32*Nt, 32) matrix laid out as Nt
// 32×32 bf16 tiles in face order (matches tile_utils::row_major_to_tile).
//
// Runtime args:
//   arg[0..3] = L1 byte address of streams 0..3 (NT tiles each)
//   arg[4]    = Nt = number of output tiles to feed (per stream)

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addrs[4] = {
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
    };
    uint32_t nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr uint32_t tile_words = tile_bytes / 4;

    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t s = 0; s < 4; ++s) {
            cb_reserve_back(s, 1);
            uint32_t write_ptr = get_write_ptr(s);
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addrs[s] + t * tile_bytes);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ptr);
            for (uint32_t i = 0; i < tile_words; ++i) {
                dst[i] = src[i];
            }
            cb_push_back(s, 1);
        }
    }
}
