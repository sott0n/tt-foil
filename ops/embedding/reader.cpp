// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for Embedding lookup (gather N rows from a [V, D] BF16 table).
//
// Output layout: row-major [N, D] in cb_out (CB 16). NOT tile-format — the
// downstream consumer is expected to either treat the output as row-major or
// run a separate to_tile pass before feeding a tile-consuming op.
//
// Runtime args:
//   arg[0..1] = emb_table NOC (lo, hi)
//   arg[2]    = N                       (number of tokens to look up)
//   arg[3]    = D_bytes                 (row size in bytes = D * 2 for BF16)
//   arg[4..]  = token_ids[N]            (uint32 ids)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

void kernel_main() {
    const uint64_t emb_noc = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t N        = get_arg_val<uint32_t>(2);
    const uint32_t D_bytes  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = 16;

    cb_reserve_back(cb_out, 1);
    const uint32_t l1_base = get_write_ptr(cb_out);

    for (uint32_t r = 0; r < N; ++r) {
        const uint32_t token = get_arg_val<uint32_t>(4 + r);
        const uint64_t src   = emb_noc + static_cast<uint64_t>(token) * D_bytes;
        const uint32_t dst   = l1_base + r * D_bytes;
        noc_async_read(src, dst, D_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
