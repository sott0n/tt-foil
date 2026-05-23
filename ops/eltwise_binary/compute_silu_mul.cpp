// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TRISC compute kernel: fused SiLU then elementwise multiply.
// out = SiLU(A) * B per tile. Used for SwiGLU FFN: out = SiLU(gate) * up.
//
// Saves one host dispatch (≈3.5 ms floor) per layer-step versus the
// previous two-op chain (compute_silu → cb_out → DRAM → next op reader →
// cb_a → compute_mul → cb_out).
//
// CBs:
//   c_0  : A (gate) tiles  — depth 1
//   c_1  : B (up)   tiles  — depth 1
//   c_24 : SiLU(A) scratch — depth 1 (set by op_lib)
//   c_16 : output           — depth 1
//
// Runtime args (TRISC RTA region):
//   arg[0] = num_tiles

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_a   = 0;
    constexpr uint32_t cb_b   = 1;
    constexpr uint32_t cb_s   = 24;  // intermediate: SiLU(A)
    constexpr uint32_t cb_out = 16;

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Set up hardware for both unary (SFPU SiLU) and binary (mul_tiles)
    // paths. We re-init each phase per tile because the unpacker config
    // differs for cb_a (unary) vs (cb_s, cb_b) (binary). The init calls
    // touch register file only and are cheap relative to the SFPU+MAC
    // work for a single 32×32 bf16 tile.
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // --- Phase 1: SiLU(A) → cb_s ---
        init_sfpu(cb_a, cb_s);
        silu_tile_init();

        cb_wait_front(cb_a, 1);
        cb_reserve_back(cb_s, 1);

        tile_regs_acquire();
        copy_tile(cb_a, 0, 0);
        silu_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_s);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_push_back(cb_s, 1);

        // --- Phase 2: cb_s * cb_b → cb_out ---
        binary_op_init_common(cb_s, cb_b, cb_out);
        mul_tiles_init(cb_s, cb_b);

        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        mul_tiles(cb_s, cb_b, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_b, 1);
        cb_push_back(cb_out, 1);
    }
}
