// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal reproducer for the iter3 chip-state corruption: running an
// extra single-core kernel dispatch after a 1×4 matmul_grid call broke
// subsequent decode steps in qwen3_run. The reduction:
//
//   Phase 1 (baseline):
//     - boot 1×4 grid on (0,0..0,3)
//     - loop kSteps:
//         exec matmul_grid(A,B,C) ; read C
//     - assert all step's C identical (this should pass — it's what
//       qwen3_run does today)
//
//   Phase 2 (the iter3 scenario):
//     - loop kSteps:
//         exec matmul_grid(A,B,C) ; exec silu(SI,SO) ; read both
//     - assert all step's (C, SO) identical AND step-0 C matches phase-1
//     - if iter3 bug reproduces here, we get a small, fast, repeatable
//       test we can use to verify hypotheses against dispatch.cpp.
//
// Requires TT_FOIL_OPS_DIR pointing at a tree with both matmul/prebuilt
// and eltwise_unary/prebuilt under it.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

constexpr uint32_t kMt   = 1;
constexpr uint32_t kKt   = 2;
constexpr uint32_t kNt   = 4;   // Nt_per_core = 1
constexpr uint32_t kSilT = 1;   // silu IO size in tiles
constexpr uint32_t kSteps = 4;

std::vector<uint16_t> tile2d(const std::vector<uint16_t>& rm, uint32_t Rows, uint32_t Cols) {
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
}  // namespace

int main() try {
    namespace ol = tt::foil::op_lib;
    int pcie_index = 0;
    if (const char* e = std::getenv("TT_FOIL_DEVICE")) pcie_index = std::stoi(e);
    if (!std::getenv("TT_FOIL_OPS_DIR")) {
        std::fprintf(stderr, "test_post_matmul_dispatch: set TT_FOIL_OPS_DIR\n");
        return 2;
    }

    const std::vector<tt::foil::CoreCoord> grid = {{0,0}, {0,1}, {0,2}, {0,3}};
    auto dev = tt::foil::open_device(pcie_index, "", grid);

    const uint32_t M = kMt * kTileH, K = kKt * kTileW, N = kNt * kTileW;
    std::vector<uint16_t> A_rm(M * K), B_rm(K * N);
    for (uint32_t i = 0; i < M * K; ++i)
        A_rm[i] = f32_to_bf16(0.01f * (static_cast<int>(i % 64) - 32));
    for (uint32_t i = 0; i < K * N; ++i)
        B_rm[i] = f32_to_bf16(0.01f * (static_cast<int>(i % 96) - 48));
    auto a_tiles = tile2d(A_rm, M, K);
    auto b_tiles = tile2d(B_rm, K, N);

    std::vector<uint16_t> si_tile(kSilT * kTileWords);
    for (uint32_t i = 0; i < si_tile.size(); ++i)
        si_tile[i] = f32_to_bf16(0.01f * (static_cast<int>(i % 200) - 100));

    auto A = ol::allocate_tensor_dram(*dev, kMt * kKt);
    auto B = ol::allocate_tensor_dram(*dev, kKt * kNt);
    auto SI = ol::allocate_tensor_dram(*dev, kSilT);
    tt::foil::write_buffer(*dev, *A.buf, a_tiles.data(), kMt * kKt * kTileBytes);
    tt::foil::write_buffer(*dev, *B.buf, b_tiles.data(), kKt * kNt * kTileBytes);
    tt::foil::write_buffer(*dev, *SI.buf, si_tile.data(), kSilT * kTileBytes);

    auto exec_matmul = [&](ol::TensorDesc& out) {
        auto op = ol::make_matmul_grid(*dev, A, B, out, kMt, kKt, kNt, grid);
        ol::execute(*dev, op);
        for (const auto& c : grid) {
            tt::foil::release_kernels(*dev, c);
            tt::foil::reset_l1(*dev, c);
        }
    };
    tt::foil::CoreCoord core00{0, 0};
    auto exec_silu = [&](ol::TensorDesc& out) {
        auto op = ol::make_silu(*dev, SI, out, core00);
        ol::execute(*dev, op);
        tt::foil::release_kernels(*dev, core00);
        tt::foil::reset_l1(*dev, core00);
    };

    // --- Phase 1: matmul alone ---
    std::vector<std::vector<uint16_t>> c_alone(kSteps);
    for (uint32_t s = 0; s < kSteps; ++s) {
        ol::TensorDesc C;
        exec_matmul(C);
        c_alone[s].resize(kMt * kNt * kTileWords);
        tt::foil::read_buffer(*dev, *C.buf, c_alone[s].data(),
                              kMt * kNt * kTileBytes);
    }
    bool phase1_stable = true;
    for (uint32_t s = 1; s < kSteps; ++s)
        if (c_alone[s] != c_alone[0]) { phase1_stable = false; break; }
    if (!phase1_stable) {
        std::fprintf(stderr,
            "test_post_matmul_dispatch: BASELINE FAIL — matmul-alone unstable\n");
        return 1;
    }
    std::fprintf(stderr,
        "Phase 1 OK: matmul_grid stable across %u steps\n", kSteps);

    // --- Phase 2: matmul + silu ---
    std::vector<std::vector<uint16_t>> c_mix(kSteps), s_mix(kSteps);
    for (uint32_t s = 0; s < kSteps; ++s) {
        ol::TensorDesc C, SO;
        exec_matmul(C);
        exec_silu(SO);
        c_mix[s].resize(kMt * kNt * kTileWords);
        s_mix[s].resize(kSilT * kTileWords);
        tt::foil::read_buffer(*dev, *C.buf, c_mix[s].data(),
                              kMt * kNt * kTileBytes);
        tt::foil::read_buffer(*dev, *SO.buf, s_mix[s].data(),
                              kSilT * kTileBytes);
    }
    int bad_c = 0, bad_s = 0;
    for (uint32_t s = 0; s < kSteps; ++s) {
        if (c_mix[s] != c_alone[0]) ++bad_c;
        if (s_mix[s] != s_mix[0]) ++bad_s;
    }
    // iter3 saw silu writing 0x35858A86 (in 16-bit halves 0x8A86, 0x3585)
    // to its output slot. Flag that pattern too.
    bool silu_garbage = false;
    for (uint32_t i = 0; i < kTileWords; ++i)
        if (s_mix[0][i] == 0x8A86 || s_mix[0][i] == 0x3585) silu_garbage = true;

    if (bad_c == 0 && bad_s == 0 && !silu_garbage) {
        std::printf("test_post_matmul_dispatch: PASS — matmul+silu stable\n");
        std::fprintf(stderr,
            "(iter3 corruption did NOT reproduce in this minimal setup)\n");
        return 0;
    }
    std::fprintf(stderr,
        "test_post_matmul_dispatch: FAIL — bad_c=%d bad_s=%d garbage=%d\n",
        bad_c, bad_s, silu_garbage);
    std::fprintf(stderr, "silu_step0 first 16 words: ");
    for (uint32_t i = 0; i < 16 && i < s_mix[0].size(); ++i)
        std::fprintf(stderr, "%04x ", s_mix[0][i]);
    std::fprintf(stderr, "\n");
    if (bad_c) {
        std::fprintf(stderr, "matmul_step0 first 8 words: ");
        for (uint32_t i = 0; i < 8; ++i) std::fprintf(stderr, "%04x ", c_alone[0][i]);
        std::fprintf(stderr, "\nmatmul_step1 first 8 words: ");
        for (uint32_t i = 0; i < 8; ++i) std::fprintf(stderr, "%04x ", c_mix[1][i]);
        std::fprintf(stderr, "\n");
    }
    return 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_post_matmul_dispatch: EXCEPTION — %s\n", e.what());
    return 1;
}
