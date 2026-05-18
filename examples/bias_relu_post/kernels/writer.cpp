// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for bias_relu_post — drains the N post-op tiles from
// CB c_16 to a sequential DRAM destination.
//
// Runtime args:
//   arg[0..1] = output DRAM NOC addr (lo, hi)
//   arg[2]    = N tiles

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint64_t dst_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32)
                      |  static_cast<uint64_t>(get_arg_val<uint32_t>(0));
    uint32_t n        = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out),
                        dst_base + i * kTileBytes,
                        kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
