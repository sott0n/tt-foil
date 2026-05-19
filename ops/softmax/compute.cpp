// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: Softmax (per-row, minimum scope).
//
//   y[r, c] = exp(x[r, c]) / sum_c'(exp(x[r, c']))
//
// No max subtraction (numerical stability is at the caller's discretion);
// no causal mask. cb_reduce holds a tile of BF16(1.0) so the row-reduce
// computes the literal sum.
//
// Phases:
//   1. cb_inp → cb_exp     (exp_tile per Wt tiles in row)
//   2. reduce(cb_exp, ROW) → cb_sum  (single tile, col 0 = sum per row)
//   3. recip(cb_sum) → cb_recip
//   4. cb_exp * cb_recip (bcast cols) → cb_out (Wt tiles)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"

#define ACQ() do { tile_regs_acquire(); tile_regs_wait(); } while (0)
#define REL() do { tile_regs_commit(); tile_regs_release(); } while (0)

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Wt   = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_inp    = 0;
    constexpr uint32_t cb_reduce = 1;
    constexpr uint32_t cb_exp    = 2;
    constexpr uint32_t cb_sum    = 3;
    constexpr uint32_t cb_recip  = 4;
    constexpr uint32_t cb_out    = 16;
    constexpr uint32_t dst0      = 0;

    // PACK init via binary_op_init_common; ocb anywhere with same format works
    // (see CLAUDE.md: configure_pack sets PCK_EDGE_OFFSET_SEC0_mask=0xffff once).
    binary_op_init_common(cb_inp, cb_inp, cb_exp);

    cb_wait_front(cb_reduce, 1);

    for (uint32_t r = 0; r < NCHt; ++r) {
        // ---- Phase 1: exp(x) ---------------------------------------------
        init_sfpu(cb_inp, cb_exp);
        exp_tile_init<>();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_inp, wt + 1);
            cb_reserve_back(cb_exp, 1);
            ACQ();
            copy_tile(cb_inp, wt, dst0);
            exp_tile<>(dst0);
            pack_tile(dst0, cb_exp);
            REL();
            cb_push_back(cb_exp, 1);
        }
        cb_pop_front(cb_inp, Wt);

        // ---- Phase 2: row sum --------------------------------------------
        cb_wait_front(cb_exp, Wt);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_reduce, cb_sum);
        cb_reserve_back(cb_sum, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_reduce, wt, 0, dst0);
        }
        reduce_uninit<>();
        pack_tile(dst0, cb_sum);
        REL();
        cb_push_back(cb_sum, 1);

        // ---- Phase 3: 1 / sum --------------------------------------------
        // Init calls must be OUTSIDE ACQ() — they issue CFG writes that
        // would race with the locked-DST state otherwise.
        copy_tile_to_dst_init_short(cb_sum);
        recip_tile_init<>();
        cb_wait_front(cb_sum, 1);
        cb_reserve_back(cb_recip, 1);
        ACQ();
        copy_tile(cb_sum, 0, dst0);
        recip_tile<>(dst0);
        pack_tile(dst0, cb_recip);
        REL();
        cb_push_back(cb_recip, 1);
        cb_pop_front(cb_sum, 1);

        // ---- Phase 4: y = exp * (1/sum) ----------------------------------
        cb_wait_front(cb_recip, 1);
        mul_bcast_cols_init_short(cb_exp, cb_recip);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_reserve_back(cb_out, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_exp, cb_recip, wt, 0, dst0);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_exp, Wt);
        cb_pop_front(cb_recip, 1);
    }
}
