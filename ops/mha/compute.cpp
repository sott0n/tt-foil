// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: MHA (single-head, single-tile, no mask, no cache).
//
//   out = softmax(Q · K^T) · V
//
// Caller responsibilities (all on host):
//   - scale Q by 1/sqrt(D) before tiling (so Q here is already Q/sqrt(D))
//   - transpose K to KT before tiling
//   - cb_reduce contains a tile of BF16(1.0) so SUM reduce is exact
//
// Pipeline:
//   Phase 1: scores = matmul_tiles(cb_q, cb_kt) → cb_scores
//   Phase 2: softmax(cb_scores) → cb_softmaxed
//            (exp_tile → reduce SUM ROW → recip_tile → mul_bcast_cols)
//   Phase 3: out = matmul_tiles(cb_softmaxed, cb_v) → cb_out

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"

#define ACQ() do { tile_regs_acquire(); tile_regs_wait(); } while (0)
#define REL() do { tile_regs_commit(); tile_regs_release(); } while (0)

void kernel_main() {
    constexpr uint32_t cb_q         = 0;
    constexpr uint32_t cb_kt        = 1;
    constexpr uint32_t cb_v         = 2;
    constexpr uint32_t cb_reduce    = 3;
    constexpr uint32_t cb_scores    = 4;
    constexpr uint32_t cb_exp       = 5;
    constexpr uint32_t cb_sum       = 6;
    constexpr uint32_t cb_recip     = 7;
    constexpr uint32_t cb_softmaxed = 8;
    constexpr uint32_t cb_out       = 16;
    constexpr uint32_t dst0         = 0;

    compute_kernel_hw_startup(cb_q, cb_kt, cb_scores);
    mm_init(cb_q, cb_kt, cb_scores);

    cb_wait_front(cb_reduce, 1);

    // ---- Phase 1: scores = Q · KT ----
    cb_wait_front(cb_q, 1);
    cb_wait_front(cb_kt, 1);
    cb_reserve_back(cb_scores, 1);
    ACQ();
    matmul_tiles(cb_q, cb_kt, 0, 0, dst0);
    pack_tile(dst0, cb_scores);
    REL();
    cb_push_back(cb_scores, 1);
    cb_pop_front(cb_q, 1);
    cb_pop_front(cb_kt, 1);

    // ---- Phase 2: softmax(scores) ----
    // 2a: cb_exp = exp(cb_scores)
    init_sfpu(cb_scores, cb_exp);
    exp_tile_init<>();
    cb_wait_front(cb_scores, 1);
    cb_reserve_back(cb_exp, 1);
    ACQ();
    copy_tile(cb_scores, 0, dst0);
    exp_tile<>(dst0);
    pack_tile(dst0, cb_exp);
    REL();
    cb_push_back(cb_exp, 1);
    cb_pop_front(cb_scores, 1);

    // 2b: cb_sum = reduce(cb_exp, SUM, ROW)
    cb_wait_front(cb_exp, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_reduce, cb_sum);
    cb_reserve_back(cb_sum, 1);
    ACQ();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_reduce, 0, 0, dst0);
    reduce_uninit<>();
    pack_tile(dst0, cb_sum);
    REL();
    cb_push_back(cb_sum, 1);

    // 2c: cb_recip = 1 / cb_sum
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

    // 2d: cb_softmaxed = cb_exp * cb_recip (bcast col 0 of recip across all cols)
    cb_wait_front(cb_recip, 1);
    mul_bcast_cols_init_short(cb_exp, cb_recip);
    cb_reserve_back(cb_softmaxed, 1);
    ACQ();
    mul_tiles_bcast_cols(cb_exp, cb_recip, 0, 0, dst0);
    pack_tile(dst0, cb_softmaxed);
    REL();
    cb_push_back(cb_softmaxed, 1);
    cb_pop_front(cb_exp, 1);
    cb_pop_front(cb_recip, 1);

    // ---- Phase 3: out = softmaxed · V ----
    mm_init(cb_softmaxed, cb_v, cb_out);
    cb_wait_front(cb_softmaxed, 1);
    cb_wait_front(cb_v, 1);
    cb_reserve_back(cb_out, 1);
    ACQ();
    matmul_tiles(cb_softmaxed, cb_v, 0, 0, dst0);
    pack_tile(dst0, cb_out);
    REL();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_softmaxed, 1);
    cb_pop_front(cb_v, 1);
}
