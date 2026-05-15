// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the relu example (v5.5).
//
// Pulls one bf16 32×32 tile from CB c_0, copies it into DST[0] via the
// UNPACK→MATH→PACK pipeline, applies ReLU in-place on DST, then packs to
// CB c_16. The same kernel source compiles 3× for UNPACK/MATH/PACK.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    constexpr uint32_t cb_in  = 0;
    constexpr uint32_t cb_out = 16;

    // Bring all three TRISCs to a known state, then configure them for
    // an SFPU eltwise op (datacopy init + math pack-sync + pack dest init).
    compute_kernel_hw_startup(cb_in, cb_out);
    init_sfpu(cb_in, cb_out);
    // Per-op init: tells SFPU we'll use relu (must be re-issued if we
    // switch SFPU ops within the same kernel).
    relu_tile_init();

    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_in, /*in_tile_index*/ 0, /*dst_tile_index*/ 0);
    relu_tile(0);   // in-place SFPU op on DST[0]
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(/*dst_tile_index*/ 0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
}
