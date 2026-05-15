// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for matmul_mnk (v5-3): drains Mt*Nt output tiles from
// CB c_16 into the output L1 buffer in (mt, nt) row-major order — the
// same order the compute kernel produces them in.
//
// Runtime args:
//   arg[0] = L1 byte address of out_buf (Mt*Nt tiles, row-major)
//   arg[1] = Mt * Nt (total number of output tiles)

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t num_tiles  = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr uint32_t tile_words = tile_bytes / 4;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t read_ptr = get_read_ptr(cb_out);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_ptr);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr + i * tile_bytes);
        for (uint32_t w = 0; w < tile_words; ++w) {
            dst[w] = src[w];
        }
        cb_pop_front(cb_out, 1);
    }
}
