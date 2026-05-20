// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: within-tile WH transpose for each tile in cb_in.
//
// Runtime args (TRISC):
//   arg[0] = total_tiles  (= Rt * Ct)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t cb_in  = 0;
    constexpr uint32_t cb_out = 16;

    const uint32_t total_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    transpose_wh_init(cb_in, cb_out);

    for (uint32_t i = 0; i < total_tiles; ++i) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        transpose_wh_tile(cb_in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
