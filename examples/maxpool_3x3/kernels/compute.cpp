// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for maxpool_3x3 — pairwise SFPU binary_max reduction
// across 9 input streams into one output stream:
//
//   out[i] = max(c_0[i], c_1[i], ..., c_8[i])
//
// One iteration per output tile; Nt is passed as a runtime arg.
//
// Runtime args:
//   arg[0] = Nt (number of output tiles)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/binary_max_min.h"

void kernel_main() {
    constexpr uint32_t cb_in[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    constexpr uint32_t cb_out   = 16;

    uint32_t nt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in[0], cb_in[1], cb_out);
    binary_op_init_common(cb_in[0], cb_in[1], cb_out);
    binary_max_tile_init();

    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t s = 0; s < 9; ++s) cb_wait_front(cb_in[s], 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // Seed DST[0] with stream 0, then max-fold streams 1..8 through DST[1].
        copy_tile_to_dst_init_short(cb_in[0]);
        copy_tile(cb_in[0], 0, 0);
        for (uint32_t s = 1; s < 9; ++s) {
            copy_tile_to_dst_init_short(cb_in[s]);
            copy_tile(cb_in[s], 0, 1);
            binary_max_tile(0, 1, 0);
        }

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        for (uint32_t s = 0; s < 9; ++s) cb_pop_front(cb_in[s], 1);
        cb_push_back(cb_out, 1);
    }
}
