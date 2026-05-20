// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: RoPE (Rotary Position Embedding) via the op_lib host abstraction.
//
// Configuration:
//   St = 1  (32 sequence positions per tile)
//   num_heads = 2
//   Dt_half = 1  (head_dim = 64, so 2 tiles per head)
//
// Layout:
//   x    : [St, num_heads * Dt] = [1, 4] tiles
//   cos  : [St, Dt_half]        = [1, 1] tile  (shared across heads)
//   sin  : [St, Dt_half]        = [1, 1] tile
//   out  : [St, num_heads * Dt] = [1, 4] tiles
//
// Reference rotation (split-half, Llama/Qwen3 style):
//   out_first  = x_first * cos - x_second * sin
//   out_second = x_second * cos + x_first * sin

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kSt        = 1;
constexpr uint32_t kNumHeads  = 2;
constexpr uint32_t kDtHalf    = 1;
constexpr uint32_t kDt        = 2 * kDtHalf;
constexpr uint32_t kTotalDt   = kNumHeads * kDt;
constexpr uint32_t kS         = kSt * kTileH;           // 32
constexpr uint32_t kHeadDim   = kDt * kTileW;           // 64
constexpr uint32_t kHeadHalf  = kDtHalf * kTileW;       // 32

// Deterministic RNG.
float rng(uint32_t seed, float lo, float hi) {
    uint32_t h = seed * 1664525u + 1013904223u;
    return lo + (hi - lo) * (h % 65536u) / 65535.0f;
}

// Convert row-major [Rows, Cols] BF16 → tile stream.
std::vector<uint16_t> tile2d(const std::vector<uint16_t>& rm,
                             uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> out;
    out.reserve(static_cast<size_t>(Rt) * Ct * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < Rt; ++rt)
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = rm[(rt * kTileH + r) * Cols + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    return out;
}

std::vector<uint16_t> untile2d(const std::vector<uint16_t>& tiles,
                               uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> rm(Rows * Cols, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < Rt; ++rt)
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            tt::foil::test::tile_to_row_major(tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    rm[(rt * kTileH + r) * Cols + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }
    return rm;
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // -----------------------------------------------------------------
    // 1. Build inputs: x [S, num_heads*head_dim], cos/sin [S, head_dim/2]
    // -----------------------------------------------------------------
    // x: row-major [S, num_heads * head_dim]
    const uint32_t x_cols   = kNumHeads * kHeadDim;
    const uint32_t cos_cols = kHeadHalf;

    std::vector<float> x_f(kS * x_cols), cos_f(kS * cos_cols), sin_f(kS * cos_cols);
    for (uint32_t i = 0; i < kS * x_cols;   ++i) x_f[i]   = rng(i,       -1.0f, 1.0f);
    for (uint32_t i = 0; i < kS * cos_cols; ++i) cos_f[i]  = rng(i + 100,  0.5f, 1.0f);
    for (uint32_t i = 0; i < kS * cos_cols; ++i) sin_f[i]  = rng(i + 200, -0.5f, 0.5f);

    // -----------------------------------------------------------------
    // 2. Host reference: rotate each head independently.
    // -----------------------------------------------------------------
    std::vector<float> ref_f(kS * x_cols, 0.0f);
    for (uint32_t s = 0; s < kS; ++s) {
        for (uint32_t h = 0; h < kNumHeads; ++h) {
            for (uint32_t d = 0; d < kHeadHalf; ++d) {
                float x0 = x_f[s * x_cols + h * kHeadDim + d];
                float x1 = x_f[s * x_cols + h * kHeadDim + kHeadHalf + d];
                float c  = cos_f[s * cos_cols + d];
                float sn = sin_f[s * cos_cols + d];
                ref_f[s * x_cols + h * kHeadDim + d]            = x0 * c - x1 * sn;
                ref_f[s * x_cols + h * kHeadDim + kHeadHalf + d] = x1 * c + x0 * sn;
            }
        }
    }

    // -----------------------------------------------------------------
    // 3. Tile-format payloads.
    // -----------------------------------------------------------------
    auto to_bf16 = [](const std::vector<float>& f) {
        std::vector<uint16_t> b(f.size());
        for (size_t i = 0; i < f.size(); ++i) b[i] = f32_to_bf16(f[i]);
        return b;
    };
    auto x_tiles   = tile2d(to_bf16(x_f),   kS, x_cols);
    auto cos_tiles = tile2d(to_bf16(cos_f),  kS, cos_cols);
    auto sin_tiles = tile2d(to_bf16(sin_f),  kS, cos_cols);

    // -----------------------------------------------------------------
    // 4. Device run.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto T_x   = ol::allocate_tensor_dram(*dev, kSt * kTotalDt);
    auto T_cos = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_sin = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    ol::TensorDesc T_out;

    tt::foil::write_buffer(*dev, *T_x.buf,   x_tiles.data(),   x_tiles.size()   * 2);
    tt::foil::write_buffer(*dev, *T_cos.buf,  cos_tiles.data(), cos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_sin.buf,  sin_tiles.data(), sin_tiles.size() * 2);

    auto op = ol::make_rope(*dev, T_x, T_cos, T_sin, T_out,
                            kSt, kNumHeads, kDtHalf);
    ol::execute(*dev, op);

    // -----------------------------------------------------------------
    // 5. Read back and compare.
    // -----------------------------------------------------------------
    std::vector<uint16_t> out_tiles(kSt * kTotalDt * kTileWords);
    tt::foil::read_buffer(*dev, *T_out.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, x_cols);

    const float kAbsTol = 0.02f;
    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t i = 0; i < kS * x_cols; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = ref_f[i];
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) ++bad;
    }

    tt::foil::close_device(std::move(dev));

    if (bad != 0) {
        std::fprintf(stderr,
            "test_rope_op: %u/%u mismatches, worst abs diff=%.5f (tol=%.5f)\n",
            bad, kS * x_cols, worst, kAbsTol);
        return 1;
    }
    std::printf("test_rope_op: PASS  (S=%u H=%u Dt_half=%u num_heads=%u, worst=%.5f)\n",
                kS, kHeadDim, kDtHalf, kNumHeads, worst);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_rope_op: FAIL — %s\n", e.what());
    return 1;
}
