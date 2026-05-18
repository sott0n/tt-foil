// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for bias_relu_post — multi-tile column-broadcast bias
// add followed by an optional ReLU on each tile.
//
//   out[i, j] = relu_enable ? max(0, in[i, j] + bias[i, 0])
//                           :          in[i, j] + bias[i, 0]
//
// Layout assumption: each tile is (32 rows = channels, 32 cols = spatial
// slots). Bias is column 0 of a (32, 32) tile — BroadcastType::COL
// replicates it across the 32 spatial columns of every input tile.
//
// init_bcast<ELWADD, COL> configures the binary-op unpack/math/pack;
// relu_tile_init() just sets the SFPU's relu LUT and does NOT touch
// unpack/pack (same coexistence pattern that maxpool_*x* uses for
// binary_op_init_common + binary_max_tile_init).
//
// Runtime args:
//   arg[0] = N tiles
//   arg[1] = relu_enable (0 → bias only, non-zero → bias + ReLU)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    constexpr uint32_t cb_in   = 0;
    constexpr uint32_t cb_bias = 1;
    constexpr uint32_t cb_out  = 16;

    uint32_t n           = get_arg_val<uint32_t>(0);
    uint32_t relu_enable = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_in, cb_bias, cb_out);
    init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(cb_in, cb_bias, cb_out);
    relu_tile_init();

    for (uint32_t t = 0; t < n; ++t) {
        cb_wait_front(cb_in,   1);
        cb_wait_front(cb_bias, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        add_tiles_bcast_cols(cb_in, cb_bias, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
        if (relu_enable) {
            relu_tile(0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in,   1);
        cb_pop_front(cb_bias, 1);
        cb_push_back(cb_out, 1);
    }
}
