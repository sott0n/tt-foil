// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for matmul_dram (v5-4): drains num_tiles output tiles
// from cb_out and writes them to DRAM at sequential offsets via NOC.
//
// Runtime args:
//   arg[0..1] = output DRAM NOC addr (lo, hi)
//   arg[2]    = num_tiles (Mt * Nt)

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint64_t dst_base   = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32)
                        |  static_cast<uint64_t>(get_arg_val<uint32_t>(0));
    uint32_t num_tiles  = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out      = 16;
    constexpr uint32_t kTileBytes  = 32 * 32 * 2;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t read_ptr = get_read_ptr(cb_out);
        noc_async_write(read_ptr, dst_base + i * kTileBytes, kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
