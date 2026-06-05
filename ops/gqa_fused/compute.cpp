// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: fused multi-head GQA causal attention,
// FLASH-ATTENTION STREAMING form.
//
//   For each q_head h, each query row-tile qt:
//       out[qt] = ( Σ_{j<=qt} exp(Q_qt·KTj) · V_j )
//               / ( Σ_{j<=qt} rowsum(exp(Q_qt·KTj)) )
//
// This is algebraically identical to the old full-row-materialized softmax
// (no row-max subtraction; the 1/sqrt(d) score scale is pre-folded into Q
// upstream), but we stream ONE key/value block at a time and keep a running
// output accumulator O (Dt tiles) and denominator l (1 tile) in L1. L1 is
// therefore O(Dt), independent of St — so seq=1024 (St=32) fits.
//
// Causal masking collapses to a SINGLE 32x32 lower-triangular tile (cb_tri):
//   - off-diagonal blocks (j < qt) are fully in-range -> no mask multiply.
//   - the diagonal block (j == qt) is always the same within-block triangle.
//
// O / l are accumulated with TWO ping/pong CBs each (cb_o_a/cb_o_b,
// cb_l_a/cb_l_b): a compute kernel must not cb_wait_front and cb_reserve_back
// the same CB in one iteration. j==0 writes the first buffer directly (matmul
// /reduce straight into it) so no zero-init primitive is needed.
//
// Runtime args (TRISC):
//   arg[0] = St
//   arg[1] = Dt
//   arg[2] = num_q

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
constexpr uint32_t cb_q      = 0;   // Q_qt block (Dt tiles, streamed per (h,qt))
constexpr uint32_t cb_kt     = 1;   // KT col j block (Dt tiles, streamed per block)
constexpr uint32_t cb_v      = 2;   // V row j block (Dt tiles, streamed per block)
constexpr uint32_t cb_reduce = 3;   // scaler 1.0 (REDUCE_ROW), resident
constexpr uint32_t cb_s      = 4;   // score tile S_j (1)
constexpr uint32_t cb_p      = 5;   // exp(S_j) (1)
constexpr uint32_t cb_pm     = 6;   // masked exp, diagonal block only (1)
constexpr uint32_t cb_rs     = 7;   // rowsum scratch, j>0 (1)
constexpr uint32_t cb_pv     = 8;   // P_j·V[j] scratch, j>0 (Dt)
constexpr uint32_t cb_o_a    = 9;   // running O ping (Dt)
constexpr uint32_t cb_o_b    = 10;  // running O pong (Dt)
constexpr uint32_t cb_l_a    = 11;  // running denom l ping (1)
constexpr uint32_t cb_l_b    = 12;  // running denom l pong (1)
constexpr uint32_t cb_recip  = 13;  // 1/l (1)
constexpr uint32_t cb_tri    = 14;  // lower-tri diagonal mask (1), resident
constexpr uint32_t cb_out    = 16;  // output tile (1)
constexpr uint32_t dst0      = 0;
}  // namespace

void kernel_main() {
    const uint32_t St    = get_arg_val<uint32_t>(0);
    const uint32_t Dt    = get_arg_val<uint32_t>(1);
    const uint32_t num_q = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_q, cb_kt, cb_s);
    mm_init(cb_q, cb_kt, cb_s);

    // Resident across the whole call.
    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_tri,    1);

    for (uint32_t h = 0; h < num_q; ++h) {
        for (uint32_t qt = 0; qt < St; ++qt) {
            cb_wait_front(cb_q, Dt);   // Q_qt streamed once per (h,qt)

            for (uint32_t j = 0; j <= qt; ++j) {
                cb_wait_front(cb_kt, Dt);
                cb_wait_front(cb_v,  Dt);

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

                // ---- diagonal block: P_j *= tri mask ----
                uint32_t cb_psrc = cb_p;
                if (j == qt) {
                    cb_wait_front(cb_p, 1);
                    mul_tiles_init(cb_p, cb_tri);
                    cb_reserve_back(cb_pm, 1);
                    ACQ();
                    mul_tiles(cb_p, cb_tri, 0, 0, dst0);
                    pack_tile(dst0, cb_pm);
                    REL();
                    cb_push_back(cb_pm, 1);
                    cb_pop_front(cb_p, 1);
                    cb_psrc = cb_pm;
                }
                cb_wait_front(cb_psrc, 1);

                // Ping/pong parity: write o[j&1], read o[!(j&1)].
                const uint32_t o_dst = (j & 1u) ? cb_o_b : cb_o_a;
                const uint32_t o_src = (j & 1u) ? cb_o_a : cb_o_b;
                const uint32_t l_dst = (j & 1u) ? cb_l_b : cb_l_a;
                const uint32_t l_src = (j & 1u) ? cb_l_a : cb_l_b;

                if (j == 0) {
                    // ---- l = rowsum(P_j)  (straight into l_dst) ----
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_psrc, cb_reduce, l_dst);
                    cb_reserve_back(l_dst, 1);
                    ACQ();
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_psrc, cb_reduce, 0, 0, dst0);
                    reduce_uninit<>();
                    pack_tile(dst0, l_dst);
                    REL();
                    cb_push_back(l_dst, 1);

                    // ---- O = P_j · V[j]  (straight into o_dst) ----
                    mm_init(cb_psrc, cb_v, o_dst);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        cb_reserve_back(o_dst, 1);
                        ACQ();
                        matmul_tiles(cb_psrc, cb_v, 0, dt, dst0);
                        pack_tile(dst0, o_dst);
                        REL();
                        cb_push_back(o_dst, 1);
                    }
                    cb_pop_front(cb_psrc, 1);
                } else {
                    // ---- rs = rowsum(P_j) ----
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_psrc, cb_reduce, cb_rs);
                    cb_reserve_back(cb_rs, 1);
                    ACQ();
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_psrc, cb_reduce, 0, 0, dst0);
                    reduce_uninit<>();
                    pack_tile(dst0, cb_rs);
                    REL();
                    cb_push_back(cb_rs, 1);

                    // ---- pv = P_j · V[j] ----
                    mm_init(cb_psrc, cb_v, cb_pv);
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        cb_reserve_back(cb_pv, 1);
                        ACQ();
                        matmul_tiles(cb_psrc, cb_v, 0, dt, dst0);
                        pack_tile(dst0, cb_pv);
                        REL();
                        cb_push_back(cb_pv, 1);
                    }
                    cb_pop_front(cb_psrc, 1);

                    // ---- l_dst = l_src + rs ----
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

                    // ---- o_dst = o_src + pv ----
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

            // Final accumulators live in the parity of the last block (j == qt).
            const uint32_t o_fin = (qt & 1u) ? cb_o_b : cb_o_a;
            const uint32_t l_fin = (qt & 1u) ? cb_l_b : cb_l_a;

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
