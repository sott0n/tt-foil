// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for the maxpool_2x2 example (v8.1).
//
// Drains Nt output tiles from CB c_16 and word-copies each to a
// contiguous L1 destination buffer.
//
// Runtime args:
//   arg[0] = L1 byte address of the output buffer (Nt tiles wide)
//   arg[1] = Nt

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t nt       = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr uint32_t tile_words = tile_bytes / 4;

    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_out, 1);
        uint32_t read_ptr = get_read_ptr(cb_out);
        volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_ptr);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr + t * tile_bytes);
        for (uint32_t i = 0; i < tile_words; ++i) {
            dst[i] = src[i];
        }
        cb_pop_front(cb_out, 1);
    }
}
