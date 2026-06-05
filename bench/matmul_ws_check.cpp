// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Validation + timing for the weight-stationary matmul (ops/matmul_ws) vs the
// stock matmul (ops/matmul). Uses the low-level runtime API directly (no
// op_lib changes): cb_a is sized Mb*Kt and the reader/compute/writer take an
// extra Mb runtime arg. Confirms WS produces bit-comparable output to a host
// reference, then reports WS vs stock wall time at prefill (Mt>1) shapes.
//
// Run (or via bench/run.sh ws):
//   TT_VISIBLE_DEVICES=0 TT_FOIL_DEVICE=0 ./build/bench/matmul_ws_check \
//     ops/matmul/prebuilt ops/matmul_ws/prebuilt

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"
#include "tile_utils.hpp"

using namespace tt::foil;
using Clock = std::chrono::steady_clock;
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

namespace {

// Budget ~855 KB user L1 ≈ 427 tiles. WS layout: cb_a=Mb*Kt, cb_b=2*Kt
// (double-buffer to keep read/compute overlap), cb_out=2. Fall back to a
// single cb_b when even Mb=1 + 2*Kt won't fit.
uint32_t ws_cb_b_tiles(uint32_t Kt) { return (3 * Kt + 2 <= 425) ? (2 * Kt) : Kt; }
uint32_t pick_mb(uint32_t Kt, uint32_t Mt) {
    const uint32_t cbb = ws_cb_b_tiles(Kt);
    uint32_t mb = (cbb + 3 >= 425) ? 1u : (425u - cbb - 2u) / Kt;
    if (mb < 1) mb = 1;
    if (mb > Mt) mb = Mt;
    return mb;
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

struct Bins {
    std::array<RiscBinary, 5> b;
    Bins(const std::string& d, bool ws) {
        b = {{
            {RiscBinary::RiscId::BRISC,  d + "/reader.brisc.elf"},
            {RiscBinary::RiscId::NCRISC, d + "/writer.ncrisc.elf"},
            {RiscBinary::RiscId::TRISC0, d + "/matmul.trisc0.elf"},
            {RiscBinary::RiscId::TRISC1, d + "/matmul.trisc1.elf"},
            {RiscBinary::RiscId::TRISC2, d + "/matmul.trisc2.elf"},
        }};
        (void)ws;
    }
};

// Build + run one matmul (ws or stock) on a fresh device; optionally read C.
double run_matmul(int pcie, const std::string& dir, bool ws,
                  uint32_t Mt, uint32_t Kt, uint32_t Nt,
                  const std::vector<uint16_t>& a_tiles,
                  const std::vector<uint16_t>& b_tiles,
                  std::vector<uint16_t>* c_out, int iters) {
    auto dev = open_device(pcie, "", {{0, 0}});
    CoreCoord core{0, 0};
    auto a = allocate_buffer(*dev, BufferLocation::DRAM, (size_t)Mt * Kt * kTileBytes);
    auto b = allocate_buffer(*dev, BufferLocation::DRAM, (size_t)Kt * Nt * kTileBytes);
    auto c = allocate_buffer(*dev, BufferLocation::DRAM, (size_t)Mt * Nt * kTileBytes);
    write_buffer(*dev, *a, a_tiles.data(), (size_t)Mt * Kt * kTileBytes);
    write_buffer(*dev, *b, b_tiles.data(), (size_t)Kt * Nt * kTileBytes);

    const uint32_t Mb = ws ? pick_mb(Kt, Mt) : 1;
    const uint32_t cb_a_tiles = ws ? (Mb * Kt) : Kt;
    const uint32_t cb_b_tiles = ws ? ws_cb_b_tiles(Kt)
                                   : (((6 * Kt + 2) <= 855) ? (2 * Kt) : Kt);
    auto l1_a   = allocate_buffer(*dev, BufferLocation::L1, (size_t)cb_a_tiles * kTileBytes, core);
    auto l1_b   = allocate_buffer(*dev, BufferLocation::L1, (size_t)cb_b_tiles * kTileBytes, core);
    auto l1_out = allocate_buffer(*dev, BufferLocation::L1, 2 * kTileBytes, core);

    Bins bins(dir, ws);
    auto kernel = load_kernel(*dev, bins.b, core);
    std::array<CbConfig, 3> cbs = {{
        {0,  l1_a->device_addr,   cb_a_tiles * kTileBytes, cb_a_tiles, kTileBytes},
        {1,  l1_b->device_addr,   cb_b_tiles * kTileBytes, cb_b_tiles, kTileBytes},
        {16, l1_out->device_addr, 2 * kTileBytes,          2,          kTileBytes},
    }};
    register_cbs(*dev, *kernel, cbs);

    const uint64_t a_noc = make_noc_dram_addr(*dev, a->device_addr);
    const uint64_t b_noc = make_noc_dram_addr(*dev, b->device_addr);
    const uint64_t d_noc = make_noc_dram_addr(*dev, c->device_addr);
    using R = RiscBinary;
    if (ws) {
        std::array<uint32_t, 9> rb = {(uint32_t)a_noc,(uint32_t)(a_noc>>32),
            (uint32_t)b_noc,(uint32_t)(b_noc>>32), Mt,Kt,Nt,/*stride*/Nt, Mb};
        std::array<uint32_t, 4> rt = {Mt,Kt,Nt,Mb};
        std::array<uint32_t, 6> rn = {(uint32_t)d_noc,(uint32_t)(d_noc>>32), Mt,Nt,/*stride*/Nt,Mb};
        set_runtime_args(*dev, *kernel, R::RiscId::BRISC, rb);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, rn);
    } else {
        std::array<uint32_t, 8> rb = {(uint32_t)a_noc,(uint32_t)(a_noc>>32),
            (uint32_t)b_noc,(uint32_t)(b_noc>>32), Mt,Kt,Nt,/*stride*/Nt};
        std::array<uint32_t, 3> rt = {Mt,Kt,Nt};
        std::array<uint32_t, 5> rn = {(uint32_t)d_noc,(uint32_t)(d_noc>>32), Mt,Nt,/*stride*/Nt};
        set_runtime_args(*dev, *kernel, R::RiscId::BRISC, rb);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, rt);
        set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, rn);
    }

    for (int i = 0; i < 3; ++i) execute(*dev, *kernel);  // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) execute(*dev, *kernel);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    if (c_out) {
        c_out->assign((size_t)Mt * Nt * kTileWords, 0);
        read_buffer(*dev, *c, c_out->data(), (size_t)Mt * Nt * kTileBytes);
    }
    close_device(std::move(dev));
    return us;
}

struct Shape { const char* tag; uint32_t Mt, Kt, Nt; };

}  // namespace

int main(int argc, char** argv) {
    const std::string stock_dir = argc > 1 ? argv[1] : "ops/matmul/prebuilt";
    const std::string ws_dir    = argc > 2 ? argv[2] : "ops/matmul_ws/prebuilt";
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie = dev_env ? std::atoi(dev_env) : 0;

    const std::vector<Shape> shapes = {
        {"verify_m1",  1,  4,  2},
        {"verify_m3",  3,  4,  2},
        {"qkv_m8",     8,  64, 32},
        {"qkv_m16",    16, 64, 32},
        {"ffn_up_m16", 16, 64, 96},
        {"ffn_dn_m16", 16, 192, 16},
    };
    const int iters = 30;

    std::printf("%-11s Mt   Kt   Nt   Mb  WS(us)   stock(us)  speedup  maxerr\n", "shape");
    std::printf("---------------------------------------------------------------------\n");
    bool all_ok = true;
    for (const auto& s : shapes) {
        const uint32_t M = s.Mt * kTileH, K = s.Kt * kTileW, N = s.Nt * kTileW;
        std::vector<float> A(M * K), B(K * N), Cref(M * N, 0.0f);
        auto rv = [](uint32_t i){ uint32_t h=i*1664525u+1013904223u; return -0.5f+1.0f*(h%1024u)/1023.0f; };
        for (uint32_t i = 0; i < M * K; ++i) A[i] = 0.5f * rv(i);
        for (uint32_t i = 0; i < K * N; ++i) B[i] = 0.5f * rv(i + 9999u);
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

        std::vector<uint16_t> c_ws;
        double tws    = run_matmul(pcie, ws_dir,    true,  s.Mt, s.Kt, s.Nt, a_tiles, b_tiles, &c_ws, iters);
        double tstock = run_matmul(pcie, stock_dir, false, s.Mt, s.Kt, s.Nt, a_tiles, b_tiles, nullptr, iters);

        // De-tile c_ws and compare to Cref.
        std::vector<uint16_t> c_rm(M * N);
        std::vector<uint16_t> block(kTileH * kTileW);
        uint32_t idx = 0;
        for (uint32_t rt = 0; rt < s.Mt; ++rt)
            for (uint32_t ct = 0; ct < s.Nt; ++ct) {
                tt::foil::test::tile_to_row_major(c_ws.data() + idx * kTileWords, block.data());
                for (uint32_t r = 0; r < kTileH; ++r)
                    for (uint32_t c = 0; c < kTileW; ++c)
                        c_rm[(rt * kTileH + r) * N + ct * kTileW + c] = block[r * kTileW + c];
                ++idx;
            }
        float maxerr = 0.0f;
        for (uint32_t i = 0; i < M * N; ++i)
            maxerr = std::fmax(maxerr, std::fabs(bf16_to_f32(c_rm[i]) - Cref[i]));
        const float tol = 0.05f * std::sqrt((float)K);
        bool ok = maxerr <= tol;
        all_ok = all_ok && ok;
        uint32_t Mb = pick_mb(s.Kt, s.Mt);
        std::printf("%-11s %3u %4u %4u %3u %8.2f %9.2f   %5.2fx  %.4f %s\n",
                    s.tag, s.Mt, s.Kt, s.Nt, Mb, tws, tstock,
                    tws > 0 ? tstock / tws : 0.0, maxerr, ok ? "" : "FAIL");
        std::fflush(stdout);
    }
    std::printf("%s\n", all_ok ? "ALL VERIFY OK" : "VERIFY FAILED");
    return all_ok ? 0 : 1;
}
