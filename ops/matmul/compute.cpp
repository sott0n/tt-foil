// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel for ops/matmul (iter12 A-cache variant).
// Mt×Kt · Kt×Nt → Mt×Nt with DST_ACCUM_MODE=true so the inner K-loop
// accumulates in FP32.
//
// Reader streams Kt A tiles once per mt (cb_a depth Kt) and Kt B tiles
// per (mt, nt) (cb_b depth 1). The compute kernel waits on the whole
// cb_a batch up-front and indexes by kt; B is consumed one tile at a
// time as usual.
//
// Runtime args:
//   arg[0] = Mt
//   arg[1] = Kt
//   arg[2] = Nt

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;

    const uint32_t Mt = get_arg_val<uint32_t>(0);
    const uint32_t Kt = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    mm_init(cb_a, cb_b, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // A is staged once for this whole mt row.
        cb_wait_front(cb_a, Kt);

        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // iter13: B is staged in a single Kt-batch per nt; index by kt.
            cb_wait_front(cb_b, Kt);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(cb_a, cb_b, /*a_idx*/ kt, /*b_idx*/ kt, /*dst_idx*/ 0);
            }
            tile_regs_commit();
            cb_pop_front(cb_b, Kt);

            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(/*dst_idx*/ 0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }

        cb_pop_front(cb_a, Kt);
    }
}
