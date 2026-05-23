// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: fused RMSNorm + RoPE (per head).
//
//   N = RMSNorm(x, gamma)            (5 standard phases → cb_normed, depth Wt)
//   out_first[dh]  = N[dh]      * cos[st, dh] - N[Dt_half + dh] * sin[st, dh]
//   out_second[dh] = N[Dt_half+dh] * cos[st, dh] + N[dh]      * sin[st, dh]
//
// Per token-tile-row r in [0, NCHt) where NCHt = St * num_heads:
//   1. rmsnorm phases (output to cb_normed depth Wt)
//   2. rope phase: for dh in [0, Dt_half) push (out_first, out_second) pairs
//      to cb_out using cb_cos / cb_sin (persistent, indexed by st*Dt_half+dh).
//
// Runtime args:
//   arg[0] = NCHt
//   arg[1] = Wt        (= 2 * Dt_half)
//   arg[2] = num_heads (so st = r / num_heads)
//   arg[3] = Dt_half

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
    const uint32_t NCHt      = get_arg_val<uint32_t>(0);
    const uint32_t Wt        = get_arg_val<uint32_t>(1);
    const uint32_t num_heads = get_arg_val<uint32_t>(2);
    const uint32_t Dt_half   = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x          = 0;
    constexpr uint32_t cb_reduce     = 1;
    constexpr uint32_t cb_gamma      = 2;
    constexpr uint32_t cb_eps        = 3;
    constexpr uint32_t cb_x2         = 4;
    constexpr uint32_t cb_var        = 5;
    constexpr uint32_t cb_recip_sqrt = 6;
    constexpr uint32_t cb_x_normed   = 7;
    constexpr uint32_t cb_cos        = 8;
    constexpr uint32_t cb_sin        = 9;
    constexpr uint32_t cb_normed     = 10;
    constexpr uint32_t cb_tmp0       = 11;
    constexpr uint32_t cb_tmp1       = 12;
    constexpr uint32_t cb_out        = 16;
    constexpr uint32_t dst0          = 0;

    binary_op_init_common(cb_x, cb_x, cb_x2);

    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma, Wt);
    // cos/sin loaded once for all rows; index = st * Dt_half + dh.
    // We don't pop them in this kernel (single-shot run).
    const uint32_t cos_tiles_total = (NCHt / num_heads) * Dt_half;
    cb_wait_front(cb_cos, cos_tiles_total);
    cb_wait_front(cb_sin, cos_tiles_total);

    for (uint32_t r = 0; r < NCHt; ++r) {
        const uint32_t st_for_row = r / num_heads;
        const uint32_t cos_base   = st_for_row * Dt_half;

        // ============================================================
        // RMSNorm phases (same shape as ops/rmsnorm/compute.cpp).
        // Output → cb_normed (depth Wt) instead of cb_out.
        // ============================================================
        // Phase 1: x² = x*x  → cb_x2
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_x, cb_x);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_x, wt + 1);
            cb_reserve_back(cb_x2, 1);
            ACQ();
            mul_tiles(cb_x, cb_x, wt, wt, dst0);
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

        // Phase 4: x_normed = x * scale  → cb_x_normed
        cb_wait_front(cb_recip_sqrt, 1);
        reconfig_data_format(cb_x, cb_recip_sqrt);
        mul_bcast_cols_init_short(cb_x, cb_recip_sqrt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_x, wt + 1);
            cb_reserve_back(cb_x_normed, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_x, cb_recip_sqrt, wt, 0, dst0);
            pack_reconfig_data_format(cb_x_normed);
            pack_tile(dst0, cb_x_normed);
            REL();
            cb_push_back(cb_x_normed, 1);
        }
        cb_pop_front(cb_recip_sqrt, 1);
        cb_pop_front(cb_x, Wt);

        // Phase 5: normed = x_normed * gamma  → cb_normed (depth Wt)
        reconfig_data_format(cb_x_normed, cb_gamma);
        mul_tiles_init(cb_x_normed, cb_gamma);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_x_normed, wt + 1);
            cb_reserve_back(cb_normed, 1);
            ACQ();
            mul_tiles(cb_x_normed, cb_gamma, wt, wt, dst0);
            pack_reconfig_data_format(cb_normed);
            pack_tile(dst0, cb_normed);
            REL();
            cb_push_back(cb_normed, 1);
        }
        cb_pop_front(cb_x_normed, Wt);

        // ============================================================
        // RoPE phase: x0 = cb_normed[dh], x1 = cb_normed[Dt_half + dh].
        // out_first  = x0*cos - x1*sin
        // out_second = x1*cos + x0*sin
        // Per dh: push out_first then out_second to cb_out.
        // ============================================================
        cb_wait_front(cb_normed, Wt);
        for (uint32_t dh = 0; dh < Dt_half; ++dh) {
            const uint32_t cs_idx = cos_base + dh;
            const uint32_t x0_idx = dh;
            const uint32_t x1_idx = Dt_half + dh;

            // tmp0 = x0 * cos
            reconfig_data_format(cb_normed, cb_cos);
            pack_reconfig_data_format(cb_tmp0);
            mul_tiles_init(cb_normed, cb_cos);
            cb_reserve_back(cb_tmp0, 1);
            ACQ();
            mul_tiles(cb_normed, cb_cos, x0_idx, cs_idx, dst0);
            pack_tile(dst0, cb_tmp0);
            REL();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = x1 * sin
            reconfig_data_format(cb_normed, cb_sin);
            pack_reconfig_data_format(cb_tmp1);
            mul_tiles_init(cb_normed, cb_sin);
            cb_reserve_back(cb_tmp1, 1);
            ACQ();
            mul_tiles(cb_normed, cb_sin, x1_idx, cs_idx, dst0);
            pack_tile(dst0, cb_tmp1);
            REL();
            cb_push_back(cb_tmp1, 1);

            // out_first = tmp0 - tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            reconfig_data_format(cb_tmp0, cb_tmp1);
            pack_reconfig_data_format(cb_out);
            sub_tiles_init(cb_tmp0, cb_tmp1);
            cb_reserve_back(cb_out, 1);
            ACQ();
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, dst0);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // tmp0 = x1 * cos
            reconfig_data_format(cb_normed, cb_cos);
            pack_reconfig_data_format(cb_tmp0);
            mul_tiles_init(cb_normed, cb_cos);
            cb_reserve_back(cb_tmp0, 1);
            ACQ();
            mul_tiles(cb_normed, cb_cos, x1_idx, cs_idx, dst0);
            pack_tile(dst0, cb_tmp0);
            REL();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = x0 * sin
            reconfig_data_format(cb_normed, cb_sin);
            pack_reconfig_data_format(cb_tmp1);
            mul_tiles_init(cb_normed, cb_sin);
            cb_reserve_back(cb_tmp1, 1);
            ACQ();
            mul_tiles(cb_normed, cb_sin, x0_idx, cs_idx, dst0);
            pack_tile(dst0, cb_tmp1);
            REL();
            cb_push_back(cb_tmp1, 1);

            // out_second = tmp0 + tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            reconfig_data_format(cb_tmp0, cb_tmp1);
            pack_reconfig_data_format(cb_out);
            add_tiles_init(cb_tmp0, cb_tmp1);
            cb_reserve_back(cb_out, 1);
            ACQ();
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, dst0);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);
        }
        cb_pop_front(cb_normed, Wt);
    }
}
