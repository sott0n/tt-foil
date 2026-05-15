// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side helpers for the bf16 32×32 tile layout that tt-metal's
// matmul/copy LLK expects:
//
//   face 0 = rows[ 0..16), cols[ 0..16)
//   face 1 = rows[ 0..16), cols[16..32)
//   face 2 = rows[16..32), cols[ 0..16)
//   face 3 = rows[16..32), cols[16..32)
//
// Within a face, elements are row-major. Each tile is 32*32*2 B = 2 KB.
//
// Header-only. Shared by test_matmul_1tile, test_matmul_kt, and future
// compute tests.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace tt::foil::test {

inline constexpr uint32_t kTileH      = 32;
inline constexpr uint32_t kTileW      = 32;
inline constexpr uint32_t kFaceH      = 16;
inline constexpr uint32_t kFaceW      = 16;
inline constexpr uint32_t kTileBytes  = kTileH * kTileW * 2;   // bf16
inline constexpr uint32_t kTileWords  = kTileBytes / sizeof(uint16_t);

// bf16 = top 16 bits of float32, round-to-nearest-even.
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb      = (u >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    u += rounding;
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Convert a 32×32 row-major bf16 block (laid out as row*32 + col) into the
// 4-face tile layout above.  Output is appended to `tile_out`.
inline void row_major_to_tile(const uint16_t* rm32x32, std::vector<uint16_t>& tile_out) {
    size_t base = tile_out.size();
    tile_out.resize(base + kTileWords);
    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t row_off = (face / 2) * kFaceH;
        uint32_t col_off = (face % 2) * kFaceW;
        for (uint32_t r = 0; r < kFaceH; ++r) {
            for (uint32_t c = 0; c < kFaceW; ++c) {
                uint32_t src = (row_off + r) * kTileW + (col_off + c);
                uint32_t dst = face * (kFaceH * kFaceW) + r * kFaceW + c;
                tile_out[base + dst] = rm32x32[src];
            }
        }
    }
}

// Inverse: extract a 32×32 row-major block from a single 4-face tile.
inline void tile_to_row_major(const uint16_t* tile, uint16_t* rm_out) {
    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t row_off = (face / 2) * kFaceH;
        uint32_t col_off = (face % 2) * kFaceW;
        for (uint32_t r = 0; r < kFaceH; ++r) {
            for (uint32_t c = 0; c < kFaceW; ++c) {
                uint32_t src = face * (kFaceH * kFaceW) + r * kFaceW + c;
                uint32_t dst = (row_off + r) * kTileW + (col_off + c);
                rm_out[dst] = tile[src];
            }
        }
    }
}

}  // namespace tt::foil::test
