// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for matmul_kt (v5-2).
//
// Performs ONE output tile = Σ_{kt=0..Kt-1} A_tile[kt] × B_tile[kt],
// accumulating into DST register 0 inside the matrix engine.  Pack
// happens once at the end so all Kt partial products stay in DST.
//
// Kt is a compile-time constant: build_kernels.sh passes -DMM_KT=<N> on
// the compile line.  v5-2 fixes Kt=4 in the build script and matching
// test; v5-3 will parameterise it via a real compile-time-args path.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

#ifndef MM_KT
#error "matmul_kt: define MM_KT (number of K-tiles) at compile time"
#endif

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t Kt     = MM_KT;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    mm_init(cb_a, cb_b, cb_out);

    // Acquire DST regs ONCE for the whole K accumulation. With
    // DST_ACCUM_MODE=true, DST stores fp32, so all Kt partial products
    // sum without intermediate bf16 rounding — that's the whole point
    // of accum mode for matmul.
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        // matmul_tiles ACCUMULATES into DST[dst_idx]; the dst_idx=0
        // slot holds the running C tile.
        matmul_tiles(cb_a, cb_b, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(/*dst_idx*/ 0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
