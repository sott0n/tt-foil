// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for residual_add: drains RA_NT tiles from CB c_16 and
// writes them out to DRAM at sequential 2 KB offsets.
//
// Runtime args:
//   arg[0..1] = output DRAM NOC addr (lo, hi)

#include <cstdint>

#include "dataflow_api.h"

#if !defined(RA_NT)
#error "residual_add writer: define RA_NT at compile time"
#endif

void kernel_main() {
    uint64_t dst_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32)
                      |  static_cast<uint64_t>(get_arg_val<uint32_t>(0));

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;
    constexpr uint32_t Nt         = RA_NT;

    for (uint32_t i = 0; i < Nt; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t read_ptr = get_read_ptr(cb_out);
        noc_async_write(read_ptr, dst_base + i * kTileBytes, kTileBytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
