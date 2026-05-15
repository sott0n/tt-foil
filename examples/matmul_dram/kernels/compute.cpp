// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for matmul_mnk (v5-3): Mt×Kt × Kt×Nt → Mt×Nt matmul.
//
// One output tile at a time: outer loop over (mt, nt), inner K-loop
// accumulates Kt partial products into DST[0] (DST_ACCUM_MODE=true), then
// pack→push. Each output tile reads Kt A tiles + Kt B tiles from the
// reader. Tile counts Mt/Kt/Nt are baked in at compile time via
// -DMM_MT / -DMM_KT / -DMM_NT in build_kernels.sh.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

#if !defined(MM_MT) || !defined(MM_KT) || !defined(MM_NT)
#error "matmul_mnk: define MM_MT, MM_KT, MM_NT at compile time"
#endif

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t Mt     = MM_MT;
    constexpr uint32_t Kt     = MM_KT;
    constexpr uint32_t Nt     = MM_NT;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    mm_init(cb_a, cb_b, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);
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
    }
}
