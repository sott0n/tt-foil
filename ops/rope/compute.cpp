// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: RoPE (Rotary Position Embedding).
//
// For each tile pair (x_first, x_second) with shared (cos, sin):
//
//   out_first  = x_first * cos - x_second * sin
//   out_second = x_second * cos + x_first * sin
//
// Implemented as 6 passes through intermediate CBs:
//   1. tmp0 = x0 * cos
//   2. tmp1 = x1 * sin
//   3. out0 = tmp0 - tmp1           → cb_out (first-half tile)
//   4. tmp0 = x1 * cos  (reuse cb_tmp0)
//   5. tmp1 = x0 * sin  (reuse cb_tmp1)
//   6. out1 = tmp0 + tmp1           → cb_out (second-half tile)
//
// Runtime args:
//   arg[0] = total_iters  (= St * num_heads * Dt_half)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

#define ACQ() tile_regs_acquire()
#define COM() tile_regs_commit()
#define WAI() tile_regs_wait()
#define REL() tile_regs_release()

void kernel_main() {
    const uint32_t total_iters = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x0   = 0;
    constexpr uint32_t cb_x1   = 1;
    constexpr uint32_t cb_cos  = 2;
    constexpr uint32_t cb_sin  = 3;
    constexpr uint32_t cb_tmp0 = 4;
    constexpr uint32_t cb_tmp1 = 5;
    constexpr uint32_t cb_out  = 16;

    binary_op_init_common(cb_x0, cb_cos, cb_tmp0);

    for (uint32_t i = 0; i < total_iters; ++i) {
        // Hold all four source tiles until all 6 passes are done.
        cb_wait_front(cb_x0,  1);
        cb_wait_front(cb_x1,  1);
        cb_wait_front(cb_cos, 1);
        cb_wait_front(cb_sin, 1);

        // Pass 1: tmp0 = x0 * cos
        reconfig_data_format(cb_x0, cb_cos);
        pack_reconfig_data_format(cb_tmp0);
        mul_tiles_init(cb_x0, cb_cos);
        cb_reserve_back(cb_tmp0, 1);
        ACQ(); mul_tiles(cb_x0, cb_cos, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_tmp0); REL();
        cb_push_back(cb_tmp0, 1);

        // Pass 2: tmp1 = x1 * sin
        reconfig_data_format(cb_x1, cb_sin);
        pack_reconfig_data_format(cb_tmp1);
        mul_tiles_init(cb_x1, cb_sin);
        cb_reserve_back(cb_tmp1, 1);
        ACQ(); mul_tiles(cb_x1, cb_sin, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_tmp1); REL();
        cb_push_back(cb_tmp1, 1);

        // Pass 3: out_first = tmp0 - tmp1
        reconfig_data_format(cb_tmp0, cb_tmp1);
        pack_reconfig_data_format(cb_out);
        sub_tiles_init(cb_tmp0, cb_tmp1);
        cb_reserve_back(cb_out, 1);
        cb_wait_front(cb_tmp0, 1);
        cb_wait_front(cb_tmp1, 1);
        ACQ(); sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_out); REL();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);

        // Pass 4: tmp0 = x1 * cos
        reconfig_data_format(cb_x1, cb_cos);
        pack_reconfig_data_format(cb_tmp0);
        mul_tiles_init(cb_x1, cb_cos);
        cb_reserve_back(cb_tmp0, 1);
        ACQ(); mul_tiles(cb_x1, cb_cos, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_tmp0); REL();
        cb_push_back(cb_tmp0, 1);

        // Pass 5: tmp1 = x0 * sin
        reconfig_data_format(cb_x0, cb_sin);
        pack_reconfig_data_format(cb_tmp1);
        mul_tiles_init(cb_x0, cb_sin);
        cb_reserve_back(cb_tmp1, 1);
        ACQ(); mul_tiles(cb_x0, cb_sin, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_tmp1); REL();
        cb_push_back(cb_tmp1, 1);

        // Pass 6: out_second = tmp0 + tmp1
        reconfig_data_format(cb_tmp0, cb_tmp1);
        pack_reconfig_data_format(cb_out);
        add_tiles_init(cb_tmp0, cb_tmp1);
        cb_reserve_back(cb_out, 1);
        cb_wait_front(cb_tmp0, 1);
        cb_wait_front(cb_tmp1, 1);
        ACQ(); add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0); COM();
        WAI(); pack_tile(0, cb_out); REL();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);

        // Release source tiles.
        cb_pop_front(cb_x0,  1);
        cb_pop_front(cb_x1,  1);
        cb_pop_front(cb_cos, 1);
        cb_pop_front(cb_sin, 1);
    }
}
