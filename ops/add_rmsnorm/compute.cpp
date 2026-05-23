// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: fused add + RMSNorm.
//   S = A + B   (sum, also drained to DRAM by writer via cb_s_out)
//   Y = S * (1/sqrt(mean(S²) + eps)) * gamma
//
// Phases per token tile-row:
//   0: pack S = A + B into cb_sum (compute-private, depth Wt) and into
//      cb_s_out (writer-drain, depth Wt).
//   1: x² = S * S
//   2: reduce x² → E[x²]
//   3: rsqrt(E[x²] + eps)
//   4: x_normed = S * scale
//   5: out = x_normed * gamma

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"

#define ACQ() do { tile_regs_acquire(); tile_regs_wait(); } while (0)
#define REL() do { tile_regs_commit(); tile_regs_release(); } while (0)

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Wt   = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a          = 0;
    constexpr uint32_t cb_reduce     = 1;
    constexpr uint32_t cb_gamma      = 2;
    constexpr uint32_t cb_eps        = 3;
    constexpr uint32_t cb_x2         = 4;
    constexpr uint32_t cb_var        = 5;
    constexpr uint32_t cb_recip_sqrt = 6;
    constexpr uint32_t cb_x_normed   = 7;
    constexpr uint32_t cb_b          = 8;
    constexpr uint32_t cb_sum        = 9;   // S = A+B for compute (depth Wt)
    constexpr uint32_t cb_s_out      = 10;  // S = A+B drained by writer (depth Wt)
    constexpr uint32_t cb_out        = 16;
    constexpr uint32_t dst0          = 0;

    binary_op_init_common(cb_a, cb_b, cb_sum);

    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma, Wt);

    for (uint32_t r = 0; r < NCHt; ++r) {
        // Phase 0: S = A + B → cb_sum & cb_s_out
        reconfig_data_format(cb_a, cb_b);
        add_tiles_init(cb_a, cb_b);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
            cb_reserve_back(cb_sum, 1);
            cb_reserve_back(cb_s_out, 1);
            ACQ();
            add_tiles(cb_a, cb_b, 0, 0, dst0);
            pack_reconfig_data_format(cb_sum);
            pack_tile(dst0, cb_sum);
            pack_reconfig_data_format(cb_s_out);
            pack_tile(dst0, cb_s_out);
            REL();
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
            cb_push_back(cb_sum, 1);
            cb_push_back(cb_s_out, 1);
        }

        // Phase 1: x² = sum * sum  → cb_x2
        reconfig_data_format(cb_sum, cb_sum);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_sum, cb_sum);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_sum, wt + 1);
            cb_reserve_back(cb_x2, 1);
            ACQ();
            mul_tiles(cb_sum, cb_sum, wt, wt, dst0);
            pack_reconfig_data_format(cb_x2);
            pack_tile(dst0, cb_x2);
            REL();
            cb_push_back(cb_x2, 1);
        }

        // Phase 2: reduce x² → E[x²]  → cb_var
        cb_wait_front(cb_x2, Wt);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_reduce, cb_var);
        cb_reserve_back(cb_var, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_x2, Wt);
        reduce_uninit<>();
        pack_tile(dst0, cb_var);
        REL();
        cb_push_back(cb_var, 1);

        // Phase 3: rsqrt(E[x²] + eps)  → cb_recip_sqrt
        cb_wait_front(cb_var, 1);
        cb_reserve_back(cb_recip_sqrt, 1);
        reconfig_data_format(cb_var, cb_eps);
        ACQ();
        add_tiles_init(cb_var, cb_eps);
        add_tiles(cb_var, cb_eps, 0, 0, dst0);
        rsqrt_tile_init<>();
        rsqrt_tile<>(dst0);
        pack_reconfig_data_format(cb_recip_sqrt);
        pack_tile(dst0, cb_recip_sqrt);
        REL();
        cb_push_back(cb_recip_sqrt, 1);
        cb_pop_front(cb_var, 1);

        // Phase 4: x_normed = sum * scale  → cb_x_normed
        cb_wait_front(cb_recip_sqrt, 1);
        reconfig_data_format(cb_sum, cb_recip_sqrt);
        mul_bcast_cols_init_short(cb_sum, cb_recip_sqrt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_sum, wt + 1);
            cb_reserve_back(cb_x_normed, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_sum, cb_recip_sqrt, wt, 0, dst0);
            pack_reconfig_data_format(cb_x_normed);
            pack_tile(dst0, cb_x_normed);
            REL();
            cb_push_back(cb_x_normed, 1);
        }
        cb_pop_front(cb_recip_sqrt, 1);
        cb_pop_front(cb_sum, Wt);

        // Phase 5: out = x_normed * gamma  → cb_out
        reconfig_data_format(cb_x_normed, cb_gamma);
        mul_tiles_init(cb_x_normed, cb_gamma);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_x_normed, wt + 1);
            cb_reserve_back(cb_out, 1);
            ACQ();
            mul_tiles(cb_x_normed, cb_gamma, wt, wt, dst0);
            pack_reconfig_data_format(cb_out);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_x_normed, Wt);
    }
}
