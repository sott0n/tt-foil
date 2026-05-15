// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the add_tiles example (v5.6).
//
// 2-input eltwise binary: c[i] = a[i] + b[i] per element of a 32×32 tile.
// add_tiles dispatches on the matrix engine; it's the same DST-register
// flow as matmul_tiles but without the K-loop.
//
// CBs:  c_0 = A, c_1 = B, c_16 = OUT

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(/*dst_idx*/ 0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    cb_push_back(cb_out, 1);
}
