// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: RMSNorm
//   y = x * (1/sqrt(mean(x²) + eps)) * gamma
//
// Diagnostic mode: compile with -DDIAG=N to output phase N result to cb_out.
//   DIAG=1  → output x²         (expect: x²[0][0] ≈ 4.0 for x=-2)
//   DIAG=2  → output E[x²]      (expect: col-0 ≈ mean(x²) ≈ 2.something)
//   DIAG=3  → output rsqrt tile  (expect: col-0 ≈ 1/sqrt(E[x²]+eps))
//   DIAG=4  → output x_normed    (expect: x / rms, no gamma)
//   DIAG=7  → reduce cb_inp directly (skip Phase 1); col-0 ≈ sum(x)/32 ≈ -2 scaled
//   DIAG=8  → reduce cb_inp with REDUCE_SCALAR (no transpose); expect scalar ≈ sum/32
//   (no DIAG or DIAG=0) → full pipeline

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

    constexpr uint32_t cb_inp        = 0;
    constexpr uint32_t cb_reduce     = 1;
    constexpr uint32_t cb_gamma      = 2;
    constexpr uint32_t cb_eps        = 3;
    constexpr uint32_t cb_x2         = 4;
    constexpr uint32_t cb_var        = 5;
    constexpr uint32_t cb_recip_sqrt = 6;
    constexpr uint32_t cb_x_normed   = 7;
    constexpr uint32_t cb_out        = 16;
    constexpr uint32_t dst0          = 0;

    // binary_op_init_common must come first: it calls llk_pack_hw_configure → configure_pack
    // (which sets PCK_EDGE_OFFSET_SEC0_mask=0xffff) and llk_pack_init (programs PACK MOP).
    // Without it the PACK edge mask is 0 and nothing gets written to L1, producing all-zeros.
    binary_op_init_common(cb_inp, cb_inp, cb_x2);

    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma, Wt);

#if defined(DIAG) && DIAG == 7
    // DIAG=7: reduce cb_inp directly (skip Phase 1), test reduce on raw x.
    // All x=-2 → sum(x) * scaler ≈ -2.0 * scale; col-0 row-0 should be non-zero.
    // No reconfig_data_format — match reference rmsnorm.cpp pattern exactly.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, cb_out);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_inp, Wt);
        reduce_uninit<>();
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
    }
    return;
#endif

#if defined(DIAG) && DIAG == 8
    // DIAG=8: reduce cb_inp with REDUCE_SCALAR (no row/col transpose).
    // No reconfig_data_format — match reference rmsnorm.cpp pattern exactly.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_inp, cb_reduce, cb_out);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_inp, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_inp, Wt);
        reduce_uninit<>();
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
    }
    return;
#endif

#if defined(DIAG) && DIAG == 10
    // DIAG=10: reduce_tile then pack WITHOUT reduce_uninit.
    // PACK SEC1_mask=0x0001 (col 0 only) stays set; pack_tile writes col 0 of DST.
    // If GAPOOL produced non-zero, we should see it in col 0. If still zeros,
    // GAPOOL itself isn't writing DST.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, cb_out);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_inp, Wt);
        // NO reduce_uninit — keep SEC1_mask=0x0001
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
    }
    return;
#endif

#if defined(DIAG) && DIAG == 12
    // DIAG=12: reduce_init for UNPACK config, but use copy_tile (not reduce_tile)
    // to verify SrcA → DST path works.  If output == x, UNPACK is functional.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, cb_out);
    copy_tile_to_dst_init_short(cb_inp);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        copy_tile(cb_inp, 0, dst0);
        cb_pop_front(cb_inp, Wt);
        reduce_uninit<>();
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
    }
    return;
#endif

#if defined(DIAG) && DIAG == 14
    // DIAG=14: mul_tiles(cb_inp, cb_reduce, 0, 0, dst0).
    // Tests: does UNPACK B from cb_reduce work? Is scaler tile loaded with 1/32?
    // Expected output: x[0,0] * (1/32) = -2 * 0.03125 = -0.0625
    // If got=0.0000, cb_reduce data is bad. If got≈-0.0625, scaler is fine.
    mul_tiles_init(cb_inp, cb_reduce);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        mul_tiles(cb_inp, cb_reduce, 0, 0, dst0);
        pack_tile(dst0, cb_out);
        REL();
        cb_pop_front(cb_inp, Wt);
        cb_push_back(cb_out, 1);
    }
    return;
#endif

#if defined(DIAG) && DIAG == 13
    // DIAG=13: NO binary_op_init_common, ONLY reduce_init.
    // Note: binary_op_init_common is BEFORE this block, but DIAG=13 tries to
    // re-init from scratch by calling reduce-only paths.
    // Actually — we can't skip binary_op_init_common since it's above. Instead,
    // re-issue PACK hw_configure for cb_out manually before reduce.
    pack_reconfig_data_format(cb_out);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, cb_out);
    for (uint32_t r = 0; r < NCHt; ++r) {
        cb_wait_front(cb_inp, Wt);
        cb_reserve_back(cb_out, 1);
        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_inp, Wt);
        reduce_uninit<>();
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
    }
    return;
#endif

    for (uint32_t r = 0; r < NCHt; ++r) {
        // -----------------------------------------------------------------------
        // Phase 1: x² = x * x
        // -----------------------------------------------------------------------
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);

        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_inp, wt + 1);
#if defined(DIAG) && DIAG == 1
            cb_reserve_back(cb_out, 1);
            ACQ();
            mul_tiles(cb_inp, cb_inp, wt, wt, dst0);
            pack_reconfig_data_format(cb_out);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
#else
            cb_reserve_back(cb_x2, 1);
            ACQ();
            mul_tiles(cb_inp, cb_inp, wt, wt, dst0);
            pack_reconfig_data_format(cb_x2);
            pack_tile(dst0, cb_x2);
            REL();
            cb_push_back(cb_x2, 1);
#endif
        }
#if defined(DIAG) && DIAG == 1
        cb_pop_front(cb_inp, Wt);
        continue;
#endif

#if defined(DIAG) && DIAG == 5
        // Verify cb_x2 by squaring it again: expect x⁴ = 16 for x=-2
        cb_wait_front(cb_x2, Wt);
        reconfig_data_format(cb_x2, cb_x2);
        pack_reconfig_data_format(cb_out);
        mul_tiles_init(cb_x2, cb_x2);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_reserve_back(cb_out, 1);
            ACQ();
            mul_tiles(cb_x2, cb_x2, wt, wt, dst0);
            pack_reconfig_data_format(cb_out);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_x2, Wt);
        cb_pop_front(cb_inp, Wt);
        continue;
#endif

        // -----------------------------------------------------------------------
        // Phase 2: reduce x² → E[x²]
        // -----------------------------------------------------------------------
        // Pattern from tt-metal reference rmsnorm.cpp: reduce_uninit() before pack_tile.
        cb_wait_front(cb_x2, Wt);
#if defined(DIAG) && DIAG == 2
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_reduce, cb_out);
        cb_reserve_back(cb_out, 1);
#else
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_reduce, cb_var);
        cb_reserve_back(cb_var, 1);
#endif

        ACQ();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_reduce, wt, 0, dst0);
        }
        cb_pop_front(cb_x2, Wt);
        reduce_uninit<>();
#if defined(DIAG) && DIAG == 2
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_inp, Wt);
        continue;
#else
        pack_tile(dst0, cb_var);
        REL();
        cb_push_back(cb_var, 1);
#endif
#if defined(DIAG) && DIAG == 6
        // DIAG=6: verify cb_var by squaring it: expect E[x²]² = 4.0²=16 if reduce works
        cb_wait_front(cb_var, 1);
        reconfig_data_format(cb_var, cb_var);
        pack_reconfig_data_format(cb_out);
        mul_tiles_init(cb_var, cb_var);
        cb_reserve_back(cb_out, 1);
        ACQ();
        mul_tiles(cb_var, cb_var, 0, 0, dst0);
        pack_reconfig_data_format(cb_out);
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_var, 1);
        cb_pop_front(cb_inp, Wt);
        continue;
#endif

        // -----------------------------------------------------------------------
        // Phase 3: scale = 1/sqrt(E[x²] + eps)
        // -----------------------------------------------------------------------
        cb_wait_front(cb_var, 1);
        cb_reserve_back(cb_recip_sqrt, 1);
        reconfig_data_format(cb_var, cb_eps);

        ACQ();
        add_tiles_init(cb_var, cb_eps);
        add_tiles(cb_var, cb_eps, 0, 0, dst0);
        rsqrt_tile_init<>();
        rsqrt_tile<>(dst0);
#if defined(DIAG) && DIAG == 3
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(dst0, cb_out);
        REL();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_var, 1);
        cb_pop_front(cb_inp, Wt);
        continue;
#else
        pack_reconfig_data_format(cb_recip_sqrt);
        pack_tile(dst0, cb_recip_sqrt);
        REL();
        cb_push_back(cb_recip_sqrt, 1);
        cb_pop_front(cb_var, 1);
#endif

        // -----------------------------------------------------------------------
        // Phase 4: x_normed = x * scale
        // -----------------------------------------------------------------------
        cb_wait_front(cb_recip_sqrt, 1);
        reconfig_data_format(cb_inp, cb_recip_sqrt);
        mul_bcast_cols_init_short(cb_inp, cb_recip_sqrt);

        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_inp, wt + 1);
#if defined(DIAG) && DIAG == 4
            cb_reserve_back(cb_out, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_inp, cb_recip_sqrt, wt, 0, dst0);
            pack_reconfig_data_format(cb_out);
            pack_tile(dst0, cb_out);
            REL();
            cb_push_back(cb_out, 1);
#else
            cb_reserve_back(cb_x_normed, 1);
            ACQ();
            mul_tiles_bcast_cols(cb_inp, cb_recip_sqrt, wt, 0, dst0);
            pack_reconfig_data_format(cb_x_normed);
            pack_tile(dst0, cb_x_normed);
            REL();
            cb_push_back(cb_x_normed, 1);
#endif
        }
        cb_pop_front(cb_recip_sqrt, 1);
        cb_pop_front(cb_inp, Wt);
#if defined(DIAG) && DIAG == 4
        continue;
#endif

        // -----------------------------------------------------------------------
        // Phase 5: output = x_normed * gamma
        // -----------------------------------------------------------------------
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
