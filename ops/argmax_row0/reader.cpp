// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC reader for ArgmaxRow0. Scans row 0 of every column-tile of a
// BF16 tile-format buffer with shape [Mt=1, Vt], finds the index of the
// maximum BF16 value, and stages a 4-byte uint32_t into cb_out (CB 16)
// for NCRISC to drain to DRAM.
//
// Tile layout (32×32 BF16, 4 faces of 16×16): row 0 of the tile spans
//   - face 0 row 0: bytes [0..31]    (cols 0..15)
//   - face 1 row 0: bytes [512..543] (cols 16..31)
// Faces 2/3 are rows 16..31; we ignore them for row-0 argmax.
//
// We read the full 2048-byte tile per iteration (one NOC transaction) and
// index into face0/face1's row 0 in L1. Simpler than two 32-byte reads
// and avoids any per-read pipelining concerns.
//
// Runtime args:
//   arg[0..1] = logits_dram NOC (lo, hi)
//   arg[2]    = Vt   (number of column tiles)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

// BF16 "greater than" using sign-magnitude rules. Both ±0 compare equal.
static inline bool bf16_gt(uint16_t a, uint16_t b) {
    const uint32_t a_neg = a >> 15;
    const uint32_t b_neg = b >> 15;
    if (a_neg != b_neg) return a_neg == 0;
    if (a_neg) return a < b;
    return a > b;
}

void kernel_main() {
    const uint64_t logits_noc = join64(get_arg_val<uint32_t>(0),
                                       get_arg_val<uint32_t>(1));
    const uint32_t Vt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out         = 16;
    constexpr uint32_t kTileBytes     = 2048;
    constexpr uint32_t kFaceBytes     = 512;

    // Reserve one page in cb_out. The page is sized to hold a full tile
    // (see make_argmax_row0). We reuse the first 4 bytes for the final
    // uint32 result after the scan.
    cb_reserve_back(cb_out, 1);
    const uint32_t l1_scratch = get_write_ptr(cb_out);

    uint16_t best_val = 0xFF7F;  // BF16 ≈ -3.4e38
    uint32_t best_idx = 0;

    for (uint32_t t = 0; t < Vt; ++t) {
        // Read one full tile (face0..face3) into L1 scratch.
        noc_async_read(logits_noc + static_cast<uint64_t>(t) * kTileBytes,
                       l1_scratch, kTileBytes);
        noc_async_read_barrier();

        // face 0 row 0 → first 16 BF16 of face0 region (offset 0)
        // face 1 row 0 → first 16 BF16 of face1 region (offset kFaceBytes)
        volatile uint16_t* face0 =
            reinterpret_cast<volatile uint16_t*>(l1_scratch);
        volatile uint16_t* face1 =
            reinterpret_cast<volatile uint16_t*>(l1_scratch + kFaceBytes);
        for (uint32_t j = 0; j < 16; ++j) {
            const uint16_t v = face0[j];
            if (bf16_gt(v, best_val)) {
                best_val = v;
                best_idx = t * 32 + j;
            }
        }
        for (uint32_t j = 0; j < 16; ++j) {
            const uint16_t v = face1[j];
            if (bf16_gt(v, best_val)) {
                best_val = v;
                best_idx = t * 32 + 16 + j;
            }
        }
    }

    *reinterpret_cast<volatile uint32_t*>(l1_scratch) = best_idx;
    cb_push_back(cb_out, 1);
}
