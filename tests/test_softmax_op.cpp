// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Softmax via op_lib.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef SOFTMAX_OP_NCHT
#define SOFTMAX_OP_NCHT 2
#endif
#ifndef SOFTMAX_OP_WT
#define SOFTMAX_OP_WT 4
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kNCHt = SOFTMAX_OP_NCHT;
constexpr uint32_t kWt   = SOFTMAX_OP_WT;
constexpr uint32_t kW    = kWt * kTileW;
constexpr uint32_t kNr   = kNCHt * kTileH;
constexpr uint32_t kElems = kNr * kW;
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<uint16_t> x_rm(kElems);
    for (uint32_t r = 0; r < kNr; ++r)
        for (uint32_t c = 0; c < kW; ++c) {
            float v = -1.0f + 2.0f * c / (kW - 1.0f) + 0.05f * (r % 16);
            x_rm[r * kW + c] = f32_to_bf16(v);
        }

    std::vector<uint16_t> ref_rm(kElems);
    for (uint32_t r = 0; r < kNr; ++r) {
        double s = 0.0;
        for (uint32_t c = 0; c < kW; ++c) s += std::exp(bf16_to_f32(x_rm[r * kW + c]));
        for (uint32_t c = 0; c < kW; ++c)
            ref_rm[r * kW + c] = f32_to_bf16(static_cast<float>(std::exp(bf16_to_f32(x_rm[r * kW + c])) / s));
    }

    // Tile-format x.
    std::vector<uint16_t> x_tiles;
    x_tiles.reserve(static_cast<size_t>(kNCHt) * kWt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < kNCHt; ++rt)
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = x_rm[(rt * kTileH + r) * kW + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), x_tiles);
        }

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto x = ol::allocate_tensor_dram(*dev, kNCHt * kWt);
    ol::TensorDesc out;
    tt::foil::write_buffer(*dev, *x.buf, x_tiles.data(), kNCHt * kWt * kTileBytes);

    auto op = ol::make_softmax(*dev, x, out, kNCHt, kWt);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(static_cast<size_t>(kNCHt) * kWt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), kNCHt * kWt * kTileBytes);

    std::vector<uint16_t> out_rm(kElems);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < kNCHt; ++rt)
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            tt::foil::test::tile_to_row_major(out_tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    out_rm[(rt * kTileH + r) * kW + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }

    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float d = std::fabs(bf16_to_f32(out_rm[i]) - bf16_to_f32(ref_rm[i]));
        if (d > worst) worst = d;
        if (d > 0.02f) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "test_softmax_op: %u bad, worst=%.5f\n", bad, worst);
        std::fprintf(stderr, "test_softmax_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_softmax_op: PASS  (NCHt=%u Wt=%u worst=%.5f)\n", kNCHt, kWt, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_softmax_op: FAIL — %s\n", e.what());
    return 1;
}
