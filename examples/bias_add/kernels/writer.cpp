// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for the tile_copy example.
//
// Wait for one bf16 32×32 tile to land in CB c_16 (the compute kernel
// pushes it there), then word-copy it to an L1 destination address.
// Same-core L1, no NoC.
//
// Runtime args:
//   arg[0] = L1 byte address where the output tile should be written

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out    = 16;
    constexpr uint32_t one_tile  = 1;
    constexpr uint32_t tile_words = 32 * 32 * 2 / 4;  // bf16 tile in uint32

    cb_wait_front(cb_out, one_tile);
    uint32_t read_ptr = get_read_ptr(cb_out);

    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_ptr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t i = 0; i < tile_words; ++i) {
        dst[i] = src[i];
    }

    cb_pop_front(cb_out, one_tile);
}
