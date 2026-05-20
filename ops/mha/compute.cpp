// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: MHA with optional causal mask (single-head, no cache).
//
//   out = softmax(Q · K^T  ⊙  mask) · V
//
// "Mask" is multiplicative (0/1 BF16): exp(scores) * mask zeros out the
// non-causal positions before reduce. Mask is host-supplied as a [St × St]
// grid of tiles, indexed by qt * St + st. For the no-mask case the host can
// pass an all-1 mask (effectively unmasked attention).
//
// Caller responsibilities (all on host):
//   - scale Q by 1/sqrt(D)
//   - transpose K to KT
//   - provide cb_reduce = BF16(1.0) tile (SUM reduce exact)
//   - provide cb_mask as St*St tiles of 0/1 BF16
//
// Pipeline per qt row:
//   Phase 1:  scores[qt, st] = Σ_k Q[qt, k] · KT[k, st]
//   Phase 2a: cb_exp   = exp(scores)
//   Phase 2m: cb_exp_m = exp * mask                       (NEW)
//   Phase 2b: cb_sum   = reduce SUM ROW over cb_exp_m
//   Phase 2c: cb_recip = 1 / cb_sum
//   Phase 2d: cb_softmaxed = cb_exp_m * cb_recip (bcast)
//   Phase 3:  out[qt, dt] = Σ_k softmaxed[k] · V[k, dt]
//
// Runtime args:
//   arg[0] = St
//   arg[1] = Dt

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
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
    const uint32_t St = get_arg_val<uint32_t>(0);
    const uint32_t Dt = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_q         = 0;
    constexpr uint32_t cb_kt        = 1;
    constexpr uint32_t cb_v         = 2;
    constexpr uint32_t cb_reduce    = 3;
    constexpr uint32_t cb_scores    = 4;
    constexpr uint32_t cb_exp       = 5;
    constexpr uint32_t cb_sum       = 6;
    constexpr uint32_t cb_recip     = 7;
    constexpr uint32_t cb_softmaxed = 8;
    constexpr uint32_t cb_exp_m     = 9;
    constexpr uint32_t cb_mask      = 10;
    constexpr uint32_t cb_out       = 16;
    constexpr uint32_t dst0         = 0;

    compute_kernel_hw_startup(cb_q, cb_kt, cb_scores);
    mm_init(cb_q, cb_kt, cb_scores);

    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_mask, St * St);
    cb_wait_front(cb_q,  St * Dt);
    cb_wait_front(cb_kt, Dt * St);
    cb_wait_front(cb_v,  St * Dt);

    for (uint32_t qt = 0; qt < St; ++qt) {
        // ---- Phase 1: scores[qt, st] = Σ_k Q[qt, k] · KT[k, st] ----
        mm_init(cb_q, cb_kt, cb_scores);
        for (uint32_t st = 0; st < St; ++st) {
            cb_reserve_back(cb_scores, 1);
            ACQ();
            for (uint32_t k = 0; k < Dt; ++k) {
                matmul_tiles(cb_q, cb_kt, qt * Dt + k, k * St + st, dst0);
            }
            pack_tile(dst0, cb_scores);
            REL();
            cb_push_back(cb_scores, 1);
        }

        // ---- Phase 2a: exp(scores) ----
        init_sfpu(cb_scores, cb_exp);
        exp_tile_init<>();
        cb_wait_front(cb_scores, St);
        for (uint32_t st = 0; st < St; ++st) {
            cb_reserve_back(cb_exp, 1);
            ACQ();
            copy_tile(cb_scores, st, dst0);
            exp_tile<>(dst0);
            pack_tile(dst0, cb_exp);
            REL();
            cb_push_back(cb_exp, 1);
        }
        cb_pop_front(cb_scores, St);

        // ---- Phase 2m: cb_exp_m = cb_exp * cb_mask (causal zeroing) ----
        cb_wait_front(cb_exp, St);
        mul_tiles_init(cb_exp, cb_mask);
        for (uint32_t st = 0; st < St; ++st) {
            cb_reserve_back(cb_exp_m, 1);
            ACQ();
            mul_tiles(cb_exp, cb_mask, st, qt * St + st, dst0);
            pack_tile(dst0, cb_exp_m);
            REL();
            cb_push_back(cb_exp_m, 1);
        }
        cb_pop_front(cb_exp, St);

        // ---- Phase 2b: row sum across cb_exp_m ----
        cb_wait_front(cb_exp_m, St);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp_m, cb_reduce, cb_sum);
        cb_reserve_back(cb_sum, 1);
        ACQ();
        for (uint32_t st = 0; st < St; ++st) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp_m, cb_reduce, st, 0, dst0);
        }
        reduce_uninit<>();
        pack_tile(dst0, cb_sum);
        REL();
        cb_push_back(cb_sum, 1);

        // ---- Phase 2c: 1 / sum ----
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

        // ---- Phase 2d: softmaxed = exp_m * recip (bcast col 0) ----
        cb_wait_front(cb_recip, 1);
        mul_bcast_cols_init_short(cb_exp_m, cb_recip);
        for (uint32_t st = 0; st < St; ++st) {
            cb_reserve_back(cb_softmaxed, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_exp_m, cb_recip, st, 0, dst0);
            pack_tile(dst0, cb_softmaxed);
            REL();
            cb_push_back(cb_softmaxed, 1);
        }
        cb_pop_front(cb_exp_m, St);
        cb_pop_front(cb_recip, 1);

        // ---- Phase 3: out[qt, dt] = Σ_k softmaxed[k] · V[k, dt] ----
        cb_wait_front(cb_softmaxed, St);
        mm_init(cb_softmaxed, cb_v, cb_out);
        for (uint32_t dt = 0; dt < Dt; ++dt) {
            cb_reserve_back(cb_out, 1);
            ACQ();
            for (uint32_t k = 0; k < St; ++k) {
                matmul_tiles(cb_softmaxed, cb_v, k, k * Dt + dt, dst0);
            }
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_softmaxed, St);
    }

    cb_pop_front(cb_q,    St * Dt);
    cb_pop_front(cb_kt,   Dt * St);
    cb_pop_front(cb_v,    St * Dt);
    cb_pop_front(cb_mask, St * St);
}
