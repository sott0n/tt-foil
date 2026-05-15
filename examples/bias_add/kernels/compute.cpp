// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the bias_add example (v5.7).
//
// Row-broadcast bias add: out[i, j] = in[i, j] + bias[0, j], for all
// rows i.  The bias tile only needs row 0 populated; the LLK unpack
// engine reads bcast_row_idx (default 0) and replicates it.
//
// CBs:  c_0 = INPUT (Y), c_1 = BIAS (only row 0 used), c_16 = OUT

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t cb_in   = 0;
    constexpr uint32_t cb_bias = 1;
    constexpr uint32_t cb_out  = 16;

    compute_kernel_hw_startup(cb_in, cb_bias, cb_out);
    init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(cb_in, cb_bias, cb_out);

    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_bias, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    add_tiles_bcast_rows(cb_in, cb_bias, /*a_idx*/ 0, /*b_idx*/ 0, /*dst_idx*/ 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(/*dst_idx*/ 0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_bias, 1);
    cb_push_back(cb_out, 1);
}
