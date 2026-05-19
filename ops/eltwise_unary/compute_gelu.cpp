// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: GELU eltwise unary.
// Used in SigLIP-2 ViT FFN layers.
//
// Runtime args (TRISC RTA region):
//   arg[0] = num_tiles

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"

void kernel_main() {
    constexpr uint32_t cb_in  = 0;
    constexpr uint32_t cb_out = 16;

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    init_sfpu(cb_in, cb_out);
    gelu_tile_init();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        gelu_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
