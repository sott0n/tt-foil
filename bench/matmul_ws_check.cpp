// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Validation + timing for the weight-stationary matmul path now folded into
// ops/matmul via op_lib's mb_max argument. For each shape it runs op_lib
// make_matmul twice on a fresh device — mb_max=1 (stock per-mt-row) and
// mb_max=auto (cache mb A-rows) — verifies the WS output against a host
// reference, and reports WS-vs-stock wall time. Prefill (Mt>1) is the target.
//
// Run (or bench/run.sh ws):
//   TT_VISIBLE_DEVICES=0 TT_FOIL_DEVICE=0 ./build/bench/matmul_ws_check

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

using namespace tt::foil;
namespace ol = tt::foil::op_lib;
using Clock = std::chrono::steady_clock;
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

namespace {

// Largest mb_max whose cb_a (mb*Kt) + cb_b (2*Kt) + cb_out (2) fits ~855 KB
// (≈427 tiles). Kt=64 → 4, Kt=192 → 1.
uint32_t pick_mb(uint32_t Kt) {
    uint32_t budget = 427u;
    if (2 * Kt + 2 >= budget) return 1u;
    uint32_t mb = (budget - 2 * Kt - 2) / Kt;
    return mb < 1 ? 1u : mb;
}

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

// Run op_lib make_matmul at the given mb_max on a fresh device; optional readback.
double run(int pcie, uint32_t Mt, uint32_t Kt, uint32_t Nt, uint32_t mb_max,
           const std::vector<uint16_t>& a_tiles, const std::vector<uint16_t>& b_tiles,
           std::vector<uint16_t>* c_out, int iters) {
    auto dev = open_device(pcie, "", {{0, 0}});
    auto a = ol::allocate_tensor_dram(*dev, Mt * Kt);
    auto b = ol::allocate_tensor_dram(*dev, Kt * Nt);
    ol::TensorDesc c; c.num_tiles = 0;
    write_buffer(*dev, *a.buf, a_tiles.data(), (size_t)Mt * Kt * kTileBytes);
    write_buffer(*dev, *b.buf, b_tiles.data(), (size_t)Kt * Nt * kTileBytes);

    auto op = ol::make_matmul(*dev, a, b, c, Mt, Kt, Nt, {0, 0}, "", mb_max);
    for (int i = 0; i < 3; ++i) ol::execute(*dev, op);
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) ol::execute(*dev, op);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    if (c_out) {
        c_out->assign((size_t)Mt * Nt * kTileWords, 0);
        read_buffer(*dev, *c.buf, c_out->data(), (size_t)Mt * Nt * kTileBytes);
    }
    close_device(std::move(dev));
    return us;
}

struct Shape { const char* tag; uint32_t Mt, Kt, Nt; };

}  // namespace

int main() {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie = dev_env ? std::atoi(dev_env) : 0;
    const std::vector<Shape> shapes = {
        {"verify_m1",  1,  4,   2},
        {"verify_m3",  3,  4,   2},
        {"qkv_m8",     8,  64,  32},
        {"qkv_m16",    16, 64,  32},
        {"ffn_up_m16", 16, 64,  96},
        {"ffn_dn_m16", 16, 192, 16},
    };
    const int iters = 30;

    std::printf("%-11s Mt   Kt   Nt   Mb  WS(us)   stock(us)  speedup  maxerr\n", "shape");
    std::printf("---------------------------------------------------------------------\n");
    bool all_ok = true;
    for (const auto& s : shapes) {
        const uint32_t M = s.Mt * kTileH, K = s.Kt * kTileW, N = s.Nt * kTileW;
        std::vector<float> A(M * K), B(K * N), Cref(M * N, 0.0f);
        auto rvf = [](uint32_t i){ uint32_t h=i*1664525u+1013904223u; return -0.5f+1.0f*(h%1024u)/1023.0f; };
        for (uint32_t i = 0; i < M * K; ++i) A[i] = 0.5f * rvf(i);
        for (uint32_t i = 0; i < K * N; ++i) B[i] = 0.5f * rvf(i + 9999u);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
                Cref[m * N + n] = acc;
            }
        std::vector<uint16_t> A_rm(M * K), B_rm(K * N);
        for (uint32_t i = 0; i < M * K; ++i) A_rm[i] = f32_to_bf16(A[i]);
        for (uint32_t i = 0; i < K * N; ++i) B_rm[i] = f32_to_bf16(B[i]);
        auto a_tiles = tile2d(A_rm, M, K);
        auto b_tiles = tile2d(B_rm, K, N);

        const uint32_t mb = std::min(pick_mb(s.Kt), s.Mt);
        std::vector<uint16_t> c_ws;
        double tws    = run(pcie, s.Mt, s.Kt, s.Nt, mb, a_tiles, b_tiles, &c_ws, iters);
        double tstock = run(pcie, s.Mt, s.Kt, s.Nt, 1,  a_tiles, b_tiles, nullptr, iters);

        std::vector<uint16_t> c_rm(M * N);
        std::vector<uint16_t> block(kTileH * kTileW);
        uint32_t idx = 0;
        for (uint32_t rt = 0; rt < s.Mt; ++rt)
            for (uint32_t ct = 0; ct < s.Nt; ++ct) {
                tt::foil::test::tile_to_row_major(c_ws.data() + idx * kTileWords, block.data());
                for (uint32_t r = 0; r < kTileH; ++r)
                    for (uint32_t cc = 0; cc < kTileW; ++cc)
                        c_rm[(rt * kTileH + r) * N + ct * kTileW + cc] = block[r * kTileW + cc];
                ++idx;
            }
        float maxerr = 0.0f;
        for (uint32_t i = 0; i < M * N; ++i)
            maxerr = std::fmax(maxerr, std::fabs(bf16_to_f32(c_rm[i]) - Cref[i]));
        const float tol = 0.05f * std::sqrt((float)K);
        bool ok = maxerr <= tol;
        all_ok = all_ok && ok;
        std::printf("%-11s %3u %4u %4u %3u %8.2f %9.2f   %5.2fx  %.4f %s\n",
                    s.tag, s.Mt, s.Kt, s.Nt, mb, tws, tstock,
                    tws > 0 ? tstock / tws : 0.0, maxerr, ok ? "" : "FAIL");
        std::fflush(stdout);
    }
    std::printf("%s\n", all_ok ? "ALL VERIFY OK" : "VERIFY FAILED");
    return all_ok ? 0 : 1;
}
