// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: MatMul via op_lib.
//   C[Mt × Nt] = A[Mt × Kt] · B[Kt × Nt]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef MM_OP_MT
#define MM_OP_MT 2
#endif
#ifndef MM_OP_KT
#define MM_OP_KT 4
#endif
#ifndef MM_OP_NT
#define MM_OP_NT 2
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kMt = MM_OP_MT;
constexpr uint32_t kKt = MM_OP_KT;
constexpr uint32_t kNt = MM_OP_NT;
constexpr uint32_t kM = kMt * kTileH;
constexpr uint32_t kK = kKt * kTileW;
constexpr uint32_t kN = kNt * kTileW;

auto tile2d = [](const std::vector<uint16_t>& rm, uint32_t Rows, uint32_t Cols) {
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
};
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<float> A(kM * kK), B(kK * kN), C(kM * kN, 0.0f);
    auto rand_val = [](uint32_t i) {
        uint32_t h = i * 1664525u + 1013904223u;
        return -0.5f + 1.0f * (h % 1024u) / 1023.0f;
    };
    for (uint32_t i = 0; i < kM * kK; ++i) A[i] = 0.5f * rand_val(i);
    for (uint32_t i = 0; i < kK * kN; ++i) B[i] = 0.5f * rand_val(i + 9999u);
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t n = 0; n < kN; ++n) {
            float s = 0.0f;
            for (uint32_t k = 0; k < kK; ++k) s += A[m * kK + k] * B[k * kN + n];
            C[m * kN + n] = s;
        }

    std::vector<uint16_t> A_rm(kM * kK), B_rm(kK * kN), ref_rm(kM * kN);
    for (uint32_t i = 0; i < kM * kK; ++i) A_rm[i] = f32_to_bf16(A[i]);
    for (uint32_t i = 0; i < kK * kN; ++i) B_rm[i] = f32_to_bf16(B[i]);
    for (uint32_t i = 0; i < kM * kN; ++i) ref_rm[i] = f32_to_bf16(C[i]);

    auto a_tiles = tile2d(A_rm, kM, kK);
    auto b_tiles = tile2d(B_rm, kK, kN);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto a = ol::allocate_tensor_dram(*dev, kMt * kKt);
    auto b = ol::allocate_tensor_dram(*dev, kKt * kNt);
    ol::TensorDesc out;

    tt::foil::write_buffer(*dev, *a.buf, a_tiles.data(), kMt * kKt * kTileBytes);
    tt::foil::write_buffer(*dev, *b.buf, b_tiles.data(), kKt * kNt * kTileBytes);

    auto op = ol::make_matmul(*dev, a, b, out, kMt, kKt, kNt);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), kMt * kNt * kTileBytes);

    std::vector<uint16_t> out_rm(kM * kN);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < kMt; ++rt)
        for (uint32_t ct = 0; ct < kNt; ++ct) {
            tt::foil::test::tile_to_row_major(out_tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    out_rm[(rt * kTileH + r) * kN + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }

    uint32_t bad = 0; float worst = 0.0f;
    // BF16 matmul accumulates Kt partials; tolerance scales roughly with K.
    const float kAbsTol = 0.05f * std::sqrt(static_cast<float>(kK));
    for (uint32_t i = 0; i < kM * kN; ++i) {
        float d = std::fabs(bf16_to_f32(out_rm[i]) - bf16_to_f32(ref_rm[i]));
        if (d > worst) worst = d;
        if (d > kAbsTol) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "test_matmul_op: %u bad, worst=%.5f (tol=%.5f)\n",
                     bad, worst, kAbsTol);
        std::fprintf(stderr, "test_matmul_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_matmul_op: PASS  (Mt=%u Kt=%u Nt=%u worst=%.5f tol=%.5f)\n",
                kMt, kKt, kNt, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_op: FAIL — %s\n", e.what());
    return 1;
}
