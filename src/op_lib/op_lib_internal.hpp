// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Shared internals for op_lib implementations (not exported).

#pragma once

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

namespace tt::foil::op_lib::detail {

constexpr uint32_t kTileH     = 32;
constexpr uint32_t kTileW     = 32;
constexpr uint32_t kTileBytes = kTileH * kTileW * 2;  // BF16
constexpr uint32_t kFaceH     = 16;
constexpr uint32_t kFaceW     = 16;
constexpr uint32_t kTileWords = kTileBytes / 2;

// bf16 = top 16 bits of float32, round-to-nearest-even.
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

// 32×32 row-major BF16 block → 4-face tile layout (appended to `out`).
inline void row_major_to_tile(const uint16_t* rm, std::vector<uint16_t>& out) {
    const size_t base = out.size();
    out.resize(base + kTileWords);
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t roff = (face / 2) * kFaceH;
        const uint32_t coff = (face % 2) * kFaceW;
        for (uint32_t r = 0; r < kFaceH; ++r)
            for (uint32_t c = 0; c < kFaceW; ++c) {
                const uint32_t src = (roff + r) * kTileW + (coff + c);
                const uint32_t dst = face * (kFaceH * kFaceW) + r * kFaceW + c;
                out[base + dst] = rm[src];
            }
    }
}

// Tile filled with a single value (32×32 of bf16(val)).
inline std::vector<uint16_t> make_const_tile(float val) {
    std::vector<uint16_t> rm(kTileH * kTileW, f32_to_bf16(val));
    std::vector<uint16_t> out;
    row_major_to_tile(rm.data(), out);
    return out;
}

// Resolve kernel directory: explicit arg → $TT_FOIL_OPS_DIR/<op>/prebuilt
// → $TT_FOIL_KERNEL_DIR (used by the existing hw_test helper)
// → "ops/<op>/prebuilt".
inline std::string resolve_kernel_dir(const std::string& explicit_dir,
                                      const char* op_subdir) {
    if (!explicit_dir.empty()) return explicit_dir;
    if (const char* e = std::getenv("TT_FOIL_OPS_DIR"))
        return std::string(e) + "/" + op_subdir + "/prebuilt";
    if (const char* e = std::getenv("TT_FOIL_KERNEL_DIR")) return e;
    return std::string("ops/") + op_subdir + "/prebuilt";
}

}  // namespace tt::foil::op_lib::detail
