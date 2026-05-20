// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for MHA (single-tile, single-head, first cut).
//
// Loads:
//   - Q   tile (cb_q)
//   - KT  tile (cb_kt, K transposed on host)
//   - V   tile (cb_v)
//   - scaler tile (cb_reduce, BF16(1.0) — for SUM reduce in softmax)
//
// Runtime args:
//   arg[0..1]  = Q_noc  (lo, hi)
//   arg[2..3]  = KT_noc (lo, hi)
//   arg[4..5]  = V_noc  (lo, hi)
//   arg[6..7]  = scaler_noc (lo, hi)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t q_noc      = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t kt_noc     = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t v_noc      = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint64_t scaler_noc = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));

    constexpr uint32_t cb_q      = 0;
    constexpr uint32_t cb_kt     = 1;
    constexpr uint32_t cb_v      = 2;
    constexpr uint32_t cb_reduce = 3;
    constexpr uint32_t kTileBytes = 32 * 32 * 2;

    cb_reserve_back(cb_reduce, 1);
    noc_async_read(scaler_noc, get_write_ptr(cb_reduce), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(cb_reduce, 1);

    cb_reserve_back(cb_q, 1);
    noc_async_read(q_noc, get_write_ptr(cb_q), kTileBytes);

    cb_reserve_back(cb_kt, 1);
    noc_async_read(kt_noc, get_write_ptr(cb_kt), kTileBytes);

    cb_reserve_back(cb_v, 1);
    noc_async_read(v_noc, get_write_ptr(cb_v), kTileBytes);

    noc_async_read_barrier();
    cb_push_back(cb_q,  1);
    cb_push_back(cb_kt, 1);
    cb_push_back(cb_v,  1);
}
