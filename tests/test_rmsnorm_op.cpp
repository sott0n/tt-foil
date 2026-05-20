// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: RMSNorm via op_lib.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef RMS_OP_NCHT
#define RMS_OP_NCHT 2
#endif
#ifndef RMS_OP_WT
#define RMS_OP_WT 4
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kNCHt = RMS_OP_NCHT;
constexpr uint32_t kWt   = RMS_OP_WT;
constexpr uint32_t kH    = kWt * kTileW;
constexpr uint32_t kNr   = kNCHt * kTileH;
constexpr uint32_t kElems = kNr * kH;
constexpr float kEps = 1e-5f;
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<uint16_t> x_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i)
        x_rm[i] = f32_to_bf16(-2.0f + 4.0f * (i % 256) / 255.0f);

    std::vector<uint16_t> gamma_rm(kNr * kH);
    for (uint32_t r = 0; r < kNr; ++r)
        for (uint32_t c = 0; c < kH; ++c)
            gamma_rm[r * kH + c] = f32_to_bf16(1.0f);  // gamma = 1 across all 32-row tiles

    std::vector<uint16_t> ref_rm(kElems);
    for (uint32_t t = 0; t < kNr; ++t) {
        float ss = 0.0f;
        for (uint32_t c = 0; c < kH; ++c) {
            float x = bf16_to_f32(x_rm[t * kH + c]);
            ss += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(ss / kH + kEps);
        for (uint32_t c = 0; c < kH; ++c)
            ref_rm[t * kH + c] = f32_to_bf16(bf16_to_f32(x_rm[t * kH + c]) * inv_rms);
    }

    auto tile2d = [](const std::vector<uint16_t>& rm, uint32_t RowT, uint32_t ColT) {
        std::vector<uint16_t> out;
        out.reserve(static_cast<size_t>(RowT) * ColT * kTileWords);
        std::vector<uint16_t> block(kTileH * kTileW);
        for (uint32_t rt = 0; rt < RowT; ++rt)
            for (uint32_t ct = 0; ct < ColT; ++ct) {
                for (uint32_t r = 0; r < kTileH; ++r)
                    for (uint32_t c = 0; c < kTileW; ++c)
                        block[r * kTileW + c] =
                            rm[(rt * kTileH + r) * (ColT * kTileW) + ct * kTileW + c];
                tt::foil::test::row_major_to_tile(block.data(), out);
            }
        return out;
    };
    auto x_tiles     = tile2d(x_rm,     kNCHt, kWt);
    auto gamma_tiles = tile2d(gamma_rm, kNCHt, kWt);  // replicated across token rows

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto x     = ol::allocate_tensor_dram(*dev, kNCHt * kWt);
    auto gamma = ol::allocate_tensor_dram(*dev, kWt);  // one row of gamma tiles
    ol::TensorDesc out;
    tt::foil::write_buffer(*dev, *x.buf,     x_tiles.data(),     kNCHt * kWt * kTileBytes);
    // Gamma in the kernel is 1 row of Wt tiles; replicate first tile-row from gamma_tiles.
    tt::foil::write_buffer(*dev, *gamma.buf, gamma_tiles.data(), kWt * kTileBytes);

    auto op = ol::make_rmsnorm(*dev, x, gamma, out, kNCHt, kWt, kEps);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(static_cast<size_t>(kNCHt) * kWt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), kNCHt * kWt * kTileBytes);

    std::vector<uint16_t> out_rm(kElems);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < kNCHt; ++rt)
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            tt::foil::test::tile_to_row_major(out_tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    out_rm[(rt * kTileH + r) * kH + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }

    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float d = std::fabs(bf16_to_f32(out_rm[i]) - bf16_to_f32(ref_rm[i]));
        if (d > worst) worst = d;
        if (d > 0.02f) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "test_rmsnorm_op: %u bad, worst=%.5f\n", bad, worst);
        std::fprintf(stderr, "test_rmsnorm_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_rmsnorm_op: PASS  (NCHt=%u Wt=%u worst=%.5f)\n", kNCHt, kWt, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_rmsnorm_op: FAIL — %s\n", e.what());
    return 1;
}
