// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for maxpool_3x3 — 9 pool-window positions for a 3×3
// stride-2 pad-1 maxpool over a (H, W, C) bf16 tensor.
//
// The host gathers the nine (di, dj) ∈ {-1, 0, +1}² window-offset
// positions into separate L1 tile streams (Nt tiles each). For
// out-of-bounds positions caused by padding the host fills the
// corresponding element with a very negative bf16 sentinel so SFPU
// binary_max only selects them if every in-bounds neighbour was worse.
//
// Runtime args:
//   arg[0..8] = L1 byte address of streams 0..8 (Nt tiles each)
//   arg[9]    = Nt

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addrs[9];
    for (uint32_t s = 0; s < 9; ++s) {
        src_addrs[s] = get_arg_val<uint32_t>(s);
    }
    uint32_t nt = get_arg_val<uint32_t>(9);

    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr uint32_t tile_words = tile_bytes / 4;

    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t s = 0; s < 9; ++s) {
            cb_reserve_back(s, 1);
            uint32_t write_ptr = get_write_ptr(s);
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addrs[s] + t * tile_bytes);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ptr);
            for (uint32_t i = 0; i < tile_words; ++i) dst[i] = src[i];
            cb_push_back(s, 1);
        }
    }
}
