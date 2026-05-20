// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel for ops/matmul: Mt×Kt · Kt×Nt → Mt×Nt with
// DST_ACCUM_MODE=true so the inner K-loop accumulates in FP32.
//
// Mt / Kt / Nt are runtime args so a single prebuilt ELF works for
// every Qwen3-style matmul shape (QKV projection, output projection,
// MLP up/gate/down, LM head, …).
//
// Reader pushes Kt A-tiles then Kt B-tiles for each output tile in
// (mt, nt) order; the compute kernel pops the same way.
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
