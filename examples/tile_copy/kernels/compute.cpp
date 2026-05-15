// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the tile_copy example.
//
// Pull one tile from CB c_0, copy it through the dst register file via
// the UNPACK → MATH → PACK pipeline, push it to CB c_16. The host build
// script (build_kernels.sh) compiles this single source three times,
// defining one of TRISC_UNPACK / TRISC_MATH / TRISC_PACK each time, so
// the UNPACK/MATH/PACK macros in the compute_kernel_api headers select
// the right body for each TRISC.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in    = 0;
    constexpr uint32_t cb_out   = 16;
    constexpr uint32_t one_tile = 1;

    copy_tile_init(cb_in);

    cb_wait_front(cb_in, one_tile);
    cb_reserve_back(cb_out, one_tile);

    tile_regs_acquire();
    copy_tile(cb_in, /*in_tile_index*/ 0, /*dst_tile_index*/ 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(/*dst_tile_index*/ 0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in, one_tile);
    cb_push_back(cb_out, one_tile);
}
