// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for global_avg_pool — writes the single output tile
// (Mt × 1 = 1 tile for our fixture) from CB c_16 to DRAM.
//
// Runtime args:
//   arg[0..1] = output DRAM NOC addr (lo, hi)

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint64_t dst_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32)
                      |  static_cast<uint64_t>(get_arg_val<uint32_t>(0));
    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), dst_base, kTileBytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
