// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for residual_add: RA_NT-tile bf16 eltwise add, with
// optional fused ReLU.
//
// Same DST-register flow as the single-tile add_tiles example, wrapped in
// a per-tile loop. CB depth is 1 — we wait/pop/push one tile per
// iteration to keep L1 footprint minimal.
//
// CBs:        c_0 = A, c_1 = B, c_16 = OUT
// Runtime arg: arg[0] = relu_enable (0 → plain add, non-zero → add+ReLU).
//              When non-zero the per-tile pipeline runs relu_tile after
//              add_tiles, fusing the trailing bias_relu_post(relu=1)
//              that follows the residual_add in ResNet basic blocks.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

#if !defined(RA_NT)
#error "residual_add compute: define RA_NT at compile time"
#endif

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t Nt     = RA_NT;

    uint32_t relu_enable = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);
    relu_tile_init();

    for (uint32_t i = 0; i < Nt; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        add_tiles(cb_a, cb_b, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
        if (relu_enable) {
            relu_tile(0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(/*dst_idx*/ 0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_push_back(cb_out, 1);
    }
}
