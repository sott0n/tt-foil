// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for ArgmaxRow0. Drains the 4-byte uint32 result from
// cb_out (CB 16) to DRAM.
//
// Runtime args:
//   arg[0..1] = dst_dram NOC (lo, hi)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t dst_noc = join64(get_arg_val<uint32_t>(0),
                                    get_arg_val<uint32_t>(1));

    constexpr uint32_t cb_out = 16;

    cb_wait_front(cb_out, 1);
    // Only the first 4 bytes are the result; we still write 4 bytes
    // (the CB page is 64 B but the rest is junk we don't care about).
    noc_async_write(get_read_ptr(cb_out), dst_noc, /*bytes=*/4);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
