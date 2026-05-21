// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for fused multi-head GQA decode attention.
//
// Per head, the compute pushes St_q*Dt tiles (qt-outer, dt-inner). We
// scatter them into the [St_q, num_q*Dt] output buffer.
//
// Runtime args:
//   arg[0,1] = out_noc
//   arg[2]   = St_q
//   arg[3]   = Dt
//   arg[4]   = num_q

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

void kernel_main() {
    const uint64_t out_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t St_q    = get_arg_val<uint32_t>(2);
    const uint32_t Dt      = get_arg_val<uint32_t>(3);
    const uint32_t num_q   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    const uint32_t total_Nq = num_q * Dt;

    for (uint32_t h = 0; h < num_q; ++h) {
        for (uint32_t qt = 0; qt < St_q; ++qt) {
            for (uint32_t dt = 0; dt < Dt; ++dt) {
                cb_wait_front(cb_out, 1);
                uint64_t addr = out_noc + (qt * total_Nq + h * Dt + dt) * kTileBytes;
                noc_async_write(get_read_ptr(cb_out), addr, kTileBytes);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
