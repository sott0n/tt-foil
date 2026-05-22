// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NCRISC writer for ops/matmul: drains Mt*Nt output tiles from cb_out
// in (mt, nt) order and writes each to DRAM at
// `dst_base + (mt * Nt_stride + nt) * kTileBytes`. Nt_stride lets a
// column-sharded grid of kernels write per-core slices into a single
// global C buffer without overlap; single-core callers pass
// Nt_stride = Nt for the original contiguous-write behavior.
//
// Runtime args:
//   arg[0..1] = output DRAM NOC addr (lo, hi) — column-shard offset
//              already baked in by caller
//   arg[2]    = Mt
//   arg[3]    = Nt        — per-core output column count
//   arg[4]    = Nt_stride — row stride between (mt) rows, in tiles

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint64_t dst_base    = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32)
                         |  static_cast<uint64_t>(get_arg_val<uint32_t>(0));
    uint32_t Mt          = get_arg_val<uint32_t>(2);
    uint32_t Nt          = get_arg_val<uint32_t>(3);
    uint32_t Nt_stride   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out      = 16;
    constexpr uint32_t kTileBytes  = 32 * 32 * 2;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_out, 1);
            uint32_t read_ptr = get_read_ptr(cb_out);
            noc_async_write(read_ptr,
                            dst_base + (mt * Nt_stride + nt) * kTileBytes,
                            kTileBytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
