// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for matmul_1tile.
//
// Pulls one A tile from CB c_0 and one B tile from CB c_1, multiplies
// them via the matrix engine (matmul_tiles), packs the resulting tile to
// CB c_16. K-loop is degenerate (Kt=1) — that comes in v5-2.
//
// Build pattern is identical to tile_copy: build_kernels.sh compiles
// this source three times, defining TRISC_UNPACK/MATH/PACK so the
// compute_kernel_api headers select the appropriate body per TRISC.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t cb_a   = 0;   // A tile input
    constexpr uint32_t cb_b   = 1;   // B tile input
    constexpr uint32_t cb_out = 16;  // result tile output

    // Bring all three TRISCs into a known state (PACK dest init, MATH
    // pack-sync state, UNPACK config).  Required for the pipeline to
    // make forward progress; see tile_copy/compute.cpp commentary.
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    // Configure the matrix engine + UNPACK for matmul on these CBs.
    mm_init(cb_a, cb_b, cb_out);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    // matmul_tiles(cb_a, cb_b, a_tile_idx, b_tile_idx, dst_tile_idx)
    matmul_tiles(cb_a, cb_b, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(/*dst_tile_index*/ 0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    cb_push_back(cb_out, 1);
}
