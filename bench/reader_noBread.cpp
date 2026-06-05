// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PROFILING VARIANT of ops/matmul/reader.cpp.
// Identical CB protocol (pushes Kt A tiles per mt, Kt B tiles per nt) so
// the compute + writer kernels run unchanged — but it SKIPS the B-tile
// DRAM reads (noc_async_read for cb_b). A is still read once per mt
// (tiny). The CB slots therefore carry stale L1 bytes; correctness is
// irrelevant, we only measure wall time. Comparing this against the
// real reader isolates the DRAM cost of streaming the B weight matrix.

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    uint64_t a_base   = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    uint32_t Mt        = get_arg_val<uint32_t>(4);
    uint32_t Kt        = get_arg_val<uint32_t>(5);
    uint32_t Nt        = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        cb_reserve_back(cb_a, Kt);
        uint32_t a_wp_base = get_write_ptr(cb_a);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint64_t a_src = a_base + (mt * Kt + kt) * kTileBytes;
            noc_async_read(a_src, a_wp_base + kt * kTileBytes, kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, Kt);

        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_reserve_back(cb_b, Kt);
            // *** B DRAM reads intentionally skipped ***
            cb_push_back(cb_b, Kt);
        }
    }
}
