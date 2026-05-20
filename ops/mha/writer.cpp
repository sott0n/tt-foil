// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for MHA: writes single output tile from CB16 to DRAM.
//
// Runtime args:
//   arg[0..1] = dst NOC (lo, hi)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t dst_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));

    constexpr uint32_t cb_out     = 16;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), dst_noc, kTileBytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
