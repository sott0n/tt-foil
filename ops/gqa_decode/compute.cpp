// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: fused multi-head GQA decode attention,
// FLASH-ATTENTION STREAMING form.
//
//   For each q_head h, each query row-tile qt:
//       out[qt] = ( Σ_j exp(Q_qt·KTj) ⊙ mask_j · V_j )
//               / ( Σ_j rowsum(exp(Q_qt·KTj) ⊙ mask_j) )
//
// Same math as the prefill flash kernel, but St_q (Q tile-rows) is decoupled
// from St_kv (cache tile-rows), the j-loop runs the FULL cache (no causal
// truncation), and the [St_q, St_kv] padding mask is streamed one tile per
// block and multiplied on EVERY block (it zeros padding columns of the last
// cache slot). L1 is O(Dt), independent of St_kv.
//
// O / l use two ping/pong CBs each (see gqa_fused/compute.cpp for the rationale).
//
// Runtime args (TRISC):
//   arg[0] = St_q
//   arg[1] = St_kv
//   arg[2] = Dt
//   arg[3] = num_q

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

namespace {
constexpr uint32_t cb_q      = 0;
constexpr uint32_t cb_kt     = 1;
constexpr uint32_t cb_v      = 2;
constexpr uint32_t cb_reduce = 3;
constexpr uint32_t cb_s      = 4;
constexpr uint32_t cb_p      = 5;
constexpr uint32_t cb_pm     = 6;
constexpr uint32_t cb_rs     = 7;
constexpr uint32_t cb_pv     = 8;
constexpr uint32_t cb_o_a    = 9;
constexpr uint32_t cb_o_b    = 10;
constexpr uint32_t cb_l_a    = 11;
constexpr uint32_t cb_l_b    = 12;
constexpr uint32_t cb_recip  = 13;
constexpr uint32_t cb_mask   = 14;  // streamed [St_q, St_kv] padding mask, 1 tile/block
constexpr uint32_t cb_out    = 16;
constexpr uint32_t dst0      = 0;
}  // namespace

void kernel_main() {
    const uint32_t St_q  = get_arg_val<uint32_t>(0);
    const uint32_t St_kv = get_arg_val<uint32_t>(1);
    const uint32_t Dt    = get_arg_val<uint32_t>(2);
    const uint32_t num_q = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_q, cb_kt, cb_s);
    mm_init(cb_q, cb_kt, cb_s);

    cb_wait_front(cb_reduce, 1);  // scaler, resident

    const uint32_t j_last = St_kv - 1;

    for (uint32_t h = 0; h < num_q; ++h) {
        for (uint32_t qt = 0; qt < St_q; ++qt) {
            cb_wait_front(cb_q, Dt);   // Q_qt streamed once per (h,qt)

            for (uint32_t j = 0; j < St_kv; ++j) {
                cb_wait_front(cb_kt,   Dt);
                cb_wait_front(cb_v,    Dt);
                cb_wait_front(cb_mask, 1);

                // ---- S_j = Σ_k Q_qt[k] · KT_j[k]  (1 tile) ----
                mm_init(cb_q, cb_kt, cb_s);
                cb_reserve_back(cb_s, 1);
                ACQ();
                for (uint32_t k = 0; k < Dt; ++k)
                    matmul_tiles(cb_q, cb_kt, k, k, dst0);
                pack_tile(dst0, cb_s);
                REL();
                cb_push_back(cb_s, 1);

                // ---- P_j = exp(S_j) ----
                init_sfpu(cb_s, cb_p);
                exp_tile_init<>();
                cb_wait_front(cb_s, 1);
                cb_reserve_back(cb_p, 1);
                ACQ();
                copy_tile(cb_s, 0, dst0);
                exp_tile<>(dst0);
                pack_tile(dst0, cb_p);
                REL();
                cb_push_back(cb_p, 1);
                cb_pop_front(cb_s, 1);

                // ---- P_j *= mask_j  (every block) ----
                cb_wait_front(cb_p, 1);
                mul_tiles_init(cb_p, cb_mask);
                cb_reserve_back(cb_pm, 1);
                ACQ();
                mul_tiles(cb_p, cb_mask, 0, 0, dst0);
                pack_tile(dst0, cb_pm);
                REL();
                cb_push_back(cb_pm, 1);
                cb_pop_front(cb_p, 1);
                cb_pop_front(cb_mask, 1);
                cb_wait_front(cb_pm, 1);

                const uint32_t o_dst = (j & 1u) ? cb_o_b : cb_o_a;
                const uint32_t o_src = (j & 1u) ? cb_o_a : cb_o_b;
                const uint32_t l_dst = (j & 1u) ? cb_l_b : cb_l_a;
                const uint32_t l_src = (j & 1u) ? cb_l_a : cb_l_b;

                if (j == 0) {
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pm, cb_reduce, l_dst);
                    cb_reserve_back(l_dst, 1);
                    ACQ();
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pm, cb_reduce, 0, 0, dst0);
                    reduce_uninit<>();
                    pack_tile(dst0, l_dst);
                    REL();
                    cb_push_back(l_dst, 1);

                    mm_init(cb_pm, cb_v, o_dst);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        cb_reserve_back(o_dst, 1);
                        ACQ();
                        matmul_tiles(cb_pm, cb_v, 0, dt, dst0);
                        pack_tile(dst0, o_dst);
                        REL();
                        cb_push_back(o_dst, 1);
                    }
                    cb_pop_front(cb_pm, 1);
                } else {
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pm, cb_reduce, cb_rs);
                    cb_reserve_back(cb_rs, 1);
                    ACQ();
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pm, cb_reduce, 0, 0, dst0);
                    reduce_uninit<>();
                    pack_tile(dst0, cb_rs);
                    REL();
                    cb_push_back(cb_rs, 1);

                    mm_init(cb_pm, cb_v, cb_pv);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        cb_reserve_back(cb_pv, 1);
                        ACQ();
                        matmul_tiles(cb_pm, cb_v, 0, dt, dst0);
                        pack_tile(dst0, cb_pv);
                        REL();
                        cb_push_back(cb_pv, 1);
                    }
                    cb_pop_front(cb_pm, 1);

                    cb_wait_front(l_src, 1);
                    cb_wait_front(cb_rs, 1);
                    add_tiles_init(l_src, cb_rs);
                    cb_reserve_back(l_dst, 1);
                    ACQ();
                    add_tiles(l_src, cb_rs, 0, 0, dst0);
                    pack_tile(dst0, l_dst);
                    REL();
                    cb_push_back(l_dst, 1);
                    cb_pop_front(l_src, 1);
                    cb_pop_front(cb_rs, 1);

                    cb_wait_front(o_src, Dt);
                    cb_wait_front(cb_pv, Dt);
                    add_tiles_init(o_src, cb_pv);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        cb_reserve_back(o_dst, 1);
                        ACQ();
                        add_tiles(o_src, cb_pv, dt, dt, dst0);
                        pack_tile(dst0, o_dst);
                        REL();
                        cb_push_back(o_dst, 1);
                    }
                    cb_pop_front(o_src, Dt);
                    cb_pop_front(cb_pv, Dt);
                }

                cb_pop_front(cb_kt, Dt);
                cb_pop_front(cb_v,  Dt);
            }  // j

            const uint32_t o_fin = (j_last & 1u) ? cb_o_b : cb_o_a;
            const uint32_t l_fin = (j_last & 1u) ? cb_l_b : cb_l_a;

            // ---- recip = 1 / l ----
            copy_tile_to_dst_init_short(l_fin);
            recip_tile_init<>();
            cb_wait_front(l_fin, 1);
            cb_reserve_back(cb_recip, 1);
            ACQ();
            copy_tile(l_fin, 0, dst0);
            recip_tile<>(dst0);
            pack_tile(dst0, cb_recip);
            REL();
            cb_push_back(cb_recip, 1);
            cb_pop_front(l_fin, 1);

            // ---- out[qt] = O * recip  (bcast col 0) ----
            cb_wait_front(o_fin, Dt);
            cb_wait_front(cb_recip, 1);
            mul_bcast_cols_init_short(o_fin, cb_recip);
            for (uint32_t dt = 0; dt < Dt; ++dt) {
                cb_reserve_back(cb_out, 1);
                ACQ();
                mul_tiles_bcast_cols(o_fin, cb_recip, dt, 0, dst0);
                pack_tile(dst0, cb_out);
                REL();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(o_fin, Dt);
            cb_pop_front(cb_recip, 1);

            cb_pop_front(cb_q, Dt);
        }  // qt
    }  // h
}
