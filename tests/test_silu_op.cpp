// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: SiLU via the op_lib host abstraction.
//
// Demonstrates the minimum surface of include/tt_foil/ops.hpp: allocate two
// TensorDesc handles, call make_silu, fire execute, read back. No
// load_kernel / register_cbs / set_runtime_args boilerplate in the test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef SILU_OP_NT
#define SILU_OP_NT 4
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNumTiles = SILU_OP_NT;
constexpr uint32_t kElems    = kNumTiles * kTileH * kTileW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

int main() try {
    // Op library default: $TT_FOIL_OPS_DIR/<op>/prebuilt. Tests set
    // TT_FOIL_OPS_DIR; existing TT_FOIL_KERNEL_DIR is ignored by op_lib.
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Build deterministic input in row-major then tile.
    std::vector<uint16_t> x_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        float v = -2.0f + 4.0f * static_cast<float>(i % 256) / 255.0f;
        x_rm[i] = f32_to_bf16(v);
    }
    std::vector<uint16_t> x_tiles;
    x_tiles.reserve(static_cast<size_t>(kNumTiles) * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNumTiles; ++t) {
        for (uint32_t i = 0; i < kTileH * kTileW; ++i)
            block[i] = x_rm[t * kTileH * kTileW + i];
        tt::foil::test::row_major_to_tile(block.data(), x_tiles);
    }

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});

    // --- op_lib API only from here ---
    namespace ol = tt::foil::op_lib;
    auto x = ol::allocate_tensor_dram(*dev, kNumTiles);
    ol::TensorDesc out;  // auto-allocated by make_silu

    tt::foil::write_buffer(*dev, *x.buf, x_tiles.data(),
                           static_cast<std::size_t>(kNumTiles) * kTileBytes);

    auto op = ol::make_silu(*dev, x, out);
    ol::execute(*dev, op);
    // ---------------------------------

    std::vector<uint16_t> out_tiles(static_cast<size_t>(kNumTiles) * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(),
                          static_cast<std::size_t>(kNumTiles) * kTileBytes);

    // Untile to row-major for comparison.
    std::vector<uint16_t> out_rm(kElems);
    for (uint32_t t = 0; t < kNumTiles; ++t) {
        std::vector<uint16_t> tmp(kTileH * kTileW);
        tt::foil::test::tile_to_row_major(out_tiles.data() + t * kTileWords, tmp.data());
        for (uint32_t i = 0; i < kTileH * kTileW; ++i)
            out_rm[t * kTileH * kTileW + i] = tmp[i];
    }

    const float kAbsTol = 0.02f;
    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = silu(bf16_to_f32(x_rm[i]));
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) ++bad;
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_silu_op: %u/%u mismatches; worst abs diff=%.5f\n",
            bad, kElems, worst);
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_silu_op: FAIL\n");
        return 1;
    }
    std::printf("test_silu_op: PASS  (num_tiles=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kNumTiles, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_silu_op: FAIL — %s\n", e.what());
    return 1;
}
