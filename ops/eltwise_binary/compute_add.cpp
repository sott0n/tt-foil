// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: elementwise add (A + B).
// Used for residual connections in the Transformer.
//
// Runtime args (TRISC RTA region):
//   arg[0] = num_tiles

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_out = 16;

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_push_back(cb_out, 1);
    }
}
