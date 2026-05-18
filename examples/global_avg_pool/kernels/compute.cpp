// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for global_avg_pool — reduce-sum across Nt tiles with
// a constant scaler (1/HW) so the accumulator lands at the per-channel
// mean.
//
// Input layout: (C=32, HW = Nt × 32) tiled as Mt=1, Nt tiles. Within
// each tile the 32 rows are channels and the 32 columns are spatial
// slots. ReduceDim::REDUCE_ROW collapses each row to a single value
// (col 0 of the output tile); PoolType::SUM means add. The scaler
// (CB c_1) is filled with 1/HW so the running DST[0] accumulator equals
// the spatial mean once all Nt input tiles have been folded in.
//
// Output: one (32-row × 32-col) tile in CB c_16 — col 0 holds the per-
// channel mean, cols 1..31 are zero-filled by the packer's reduce mask.
//
// Runtime args:
//   arg[0] = Nt

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"

void kernel_main() {
    constexpr uint32_t cb_in     = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_out    = 16;

    uint32_t nt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out);

    // The scaler is loaded once by the reader and never popped — every
    // reduce_tile call references it at index 0.
    cb_wait_front(cb_scaler, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_in, 1);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_in, cb_scaler, /*in_idx*/ 0, /*scaler_idx*/ 0, /*dst_idx*/ 0);
        cb_pop_front(cb_in, 1);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
