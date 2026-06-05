// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute for matmul (weight-stationary, runtime Mb block height). For
// each block of Mb cached A-rows and each B column nt, it produces mb output
// tiles by reusing the single cb_b column across the block's rows.
// DST_ACCUM_MODE=true accumulates the K-loop in FP32. Mb=1 is the original
// per-(mt,nt) behavior.
//
// Output tiles are emitted in (block, nt, row) order — the writer drains them
// in the same order.
//
// Runtime args:
//   arg[0] = Mt
//   arg[1] = Kt
//   arg[2] = Nt
//   arg[3] = Mb

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
    const uint32_t Mb = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    mm_init(cb_a, cb_b, cb_out);

    for (uint32_t mt0 = 0; mt0 < Mt; mt0 += Mb) {
        const uint32_t mb = (mt0 + Mb <= Mt) ? Mb : (Mt - mt0);
        cb_wait_front(cb_a, mb * Kt);

        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_b, Kt);
            for (uint32_t r = 0; r < mb; ++r) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    matmul_tiles(cb_a, cb_b, /*a_idx*/ r * Kt + kt, /*b_idx*/ kt, /*dst*/ 0);
                }
                tile_regs_commit();

                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile(/*dst*/ 0, cb_out);
                cb_push_back(cb_out, 1);
                tile_regs_release();
            }
            cb_pop_front(cb_b, Kt);
        }
        cb_pop_front(cb_a, mb * Kt);
    }
}
