// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the maxpool_2x2 example (v8.1).
//
// Reduces four input streams pairwise with the SFPU binary_max op:
//
//     out[i] = max(c_0[i], c_1[i], c_2[i], c_3[i])
//
// One iteration per output tile; Nt is read from a compile-time arg via
// the firmware-generated runtime-args path (NCRISC/BRISC handle args;
// TRISC reads its own runtime args via get_arg_val too — but for
// simplicity we baked Nt into the BRISC stream count, and the compute
// loop also uses Nt via runtime args).
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
    constexpr uint32_t cb_0   = 0;
    constexpr uint32_t cb_1   = 1;
    constexpr uint32_t cb_2   = 2;
    constexpr uint32_t cb_3   = 3;
    constexpr uint32_t cb_out = 16;

    uint32_t nt = get_arg_val<uint32_t>(0);

    // Datacopy + SFPU pack-sync init; binary_op_init_common wires the
    // UNPACK/PACK formats for two input CBs and one output CB. The
    // per-CB copy_tile_to_dst_init_short calls below re-tag UNPACK for
    // streams 2 and 3 so copy_tile reads them with the right format.
    compute_kernel_hw_startup(cb_0, cb_1, cb_out);
    binary_op_init_common(cb_0, cb_1, cb_out);
    binary_max_tile_init();

    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_0, 1);
        cb_wait_front(cb_1, 1);
        cb_wait_front(cb_2, 1);
        cb_wait_front(cb_3, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // Load stream 0 → DST[0], stream 1 → DST[1], max → DST[0].
        copy_tile_to_dst_init_short(cb_0);
        copy_tile(cb_0, 0, 0);
        copy_tile_to_dst_init_short(cb_1);
        copy_tile(cb_1, 0, 1);
        binary_max_tile(0, 1, 0);

        // Stream 2 → DST[1], max-into-DST[0].
        copy_tile_to_dst_init_short(cb_2);
        copy_tile(cb_2, 0, 1);
        binary_max_tile(0, 1, 0);

        // Stream 3 → DST[1], max-into-DST[0].
        copy_tile_to_dst_init_short(cb_3);
        copy_tile(cb_3, 0, 1);
        binary_max_tile(0, 1, 0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_0, 1);
        cb_pop_front(cb_1, 1);
        cb_pop_front(cb_2, 1);
        cb_pop_front(cb_3, 1);
        cb_push_back(cb_out, 1);
    }
}
