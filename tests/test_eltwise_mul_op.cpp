// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: ElementwiseMul via op_lib host abstraction.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef EMUL_OP_NT
#define EMUL_OP_NT 4
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kNT    = EMUL_OP_NT;
constexpr uint32_t kElems = kNT * kTileH * kTileW;

std::vector<uint16_t> tile_block(std::vector<uint16_t> const& rm) {
    std::vector<uint16_t> out;
    out.reserve(static_cast<size_t>(kNT) * kTileWords);
    for (uint32_t t = 0; t < kNT; ++t)
        tt::foil::test::row_major_to_tile(rm.data() + t * kTileH * kTileW, out);
    return out;
}
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<uint16_t> a_rm(kElems), b_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        a_rm[i] = f32_to_bf16(-1.0f + 2.0f * (i % 256) / 255.0f);
        b_rm[i] = f32_to_bf16( 0.5f + 0.01f * (i % 100));
    }
    auto a_tiles = tile_block(a_rm);
    auto b_tiles = tile_block(b_rm);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto a = ol::allocate_tensor_dram(*dev, kNT);
    auto b = ol::allocate_tensor_dram(*dev, kNT);
    ol::TensorDesc out;

    tt::foil::write_buffer(*dev, *a.buf, a_tiles.data(), kNT * kTileBytes);
    tt::foil::write_buffer(*dev, *b.buf, b_tiles.data(), kNT * kTileBytes);

    auto op = ol::make_eltwise_mul(*dev, a, b, out);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(kNT * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), kNT * kTileBytes);

    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t t = 0; t < kNT; ++t) {
        std::vector<uint16_t> block(kTileH * kTileW);
        tt::foil::test::tile_to_row_major(out_tiles.data() + t * kTileWords, block.data());
        for (uint32_t i = 0; i < kTileH * kTileW; ++i) {
            float got = bf16_to_f32(block[i]);
            float exp = bf16_to_f32(a_rm[t * kTileH * kTileW + i]) *
                        bf16_to_f32(b_rm[t * kTileH * kTileW + i]);
            float d = std::fabs(got - exp);
            if (d > worst) worst = d;
            if (d > 0.02f) ++bad;
        }
    }
    if (bad != 0) {
        std::fprintf(stderr, "test_eltwise_mul_op: %u bad, worst=%.5f\n", bad, worst);
        std::fprintf(stderr, "test_eltwise_mul_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_eltwise_mul_op: PASS  (NT=%u, worst=%.5f)\n", kNT, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_eltwise_mul_op: FAIL — %s\n", e.what());
    return 1;
}
