// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ResNet layer = downsample_block followed by basic_block, chained
// end-to-end on a single Tensix core. This is the first tt-foil test
// that runs more than one ResNet block in sequence and verifies the
// composition against a pure-host fp32→bf16 reference.
//
// Block 1 (downsample, spatial 16×16 → 8×8):
//   main:  Conv₁(3×3, s=2) + b + ReLU  →  Conv₂(3×3, s=1) + b
//   skip:  Conv_s(1×1, s=2) + b
//   y₁ = ReLU(main + skip)
//
// Block 2 (basic, spatial 8×8 → 8×8):
//   main:  Conv₁(3×3, s=1) + b + ReLU  →  Conv₂(3×3, s=1) + b
//   skip:  identity (y₁)
//   y₂ = ReLU(main + y₁)
//
// We keep C = 32 throughout so the prebuilt kernels in
// models/downsample_block/prebuilt/{conv_s2,conv,conv1x1,residual_add}
// (all shaped for Mt=1, Kt={9,9,1}, Nt=2) can be reused as-is — no new
// kernel binaries needed for this test.
//
// Eight bf16 rounding stages stack (4 per block: conv₁, conv₂, skip,
// post-add+ReLU); tolerance is loosened accordingly.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"
#include "tile_utils.hpp"

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kC    = 32;
constexpr uint32_t kHin  = 16;
constexpr uint32_t kWin  = 16;
constexpr uint32_t kHmid = 8;     // after downsample
constexpr uint32_t kWmid = 8;
constexpr uint32_t kHout = kHmid; // basic block preserves spatial dims
constexpr uint32_t kWout = kWmid;

constexpr uint32_t kKH = 3, kKW = 3, kPad = 1;

// Shared matmul shape after downsampling: M = C → Mt = 1, N = HW_mid =
// 64 → Nt = 2. Same Mt/Nt all the way through the layer because C
// stays at 32 and spatial dims stay at 8×8 after block 1.
constexpr uint32_t kMt    = kC / kTileH;            // 1
constexpr uint32_t kNt    = (kHmid * kWmid) / kTileW; // 2
constexpr uint32_t kKt_3  = (kC * kKH * kKW) / kTileW; // 9
constexpr uint32_t kKt_1  =  kC / kTileW;              // 1
constexpr uint32_t kKdim3 = kKt_3 * kTileW;            // 288
constexpr uint32_t kKdim1 = kKt_1 * kTileW;            // 32
constexpr uint32_t kHWmid = kHmid * kWmid;             // 64
constexpr uint32_t kRaNt  = kMt * kNt;                 // 2

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// ---------------------------------------------------------------------------
// Reference math
// ---------------------------------------------------------------------------

void conv3x3_bf16(const std::vector<uint16_t>& x_chw,
                  const std::vector<uint16_t>& w_cchw,
                  uint32_t Hi, uint32_t Wi, uint32_t Ho, uint32_t Wo,
                  uint32_t stride,
                  std::vector<uint16_t>& y_chw) {
    y_chw.assign(kC * Ho * Wo, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ho = 0; ho < Ho; ++ho)
            for (uint32_t wo = 0; wo < Wo; ++wo) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kC; ++ci)
                    for (uint32_t ki = 0; ki < kKH; ++ki)
                        for (uint32_t kj = 0; kj < kKW; ++kj) {
                            int ih = static_cast<int>(ho * stride + ki) - static_cast<int>(kPad);
                            int iw = static_cast<int>(wo * stride + kj) - static_cast<int>(kPad);
                            if (ih < 0 || ih >= static_cast<int>(Hi) ||
                                iw < 0 || iw >= static_cast<int>(Wi)) continue;
                            acc += bf16_to_f32(x_chw[(ci * Hi + ih) * Wi + iw]) *
                                   bf16_to_f32(w_cchw[((co * kC + ci) * kKH + ki) * kKW + kj]);
                        }
                y_chw[(co * Ho + ho) * Wo + wo] = f32_to_bf16(acc);
            }
}

void conv1x1_s2_bf16(const std::vector<uint16_t>& x_chw,
                     const std::vector<uint16_t>& w_2d,
                     std::vector<uint16_t>& y_chw) {
    y_chw.assign(kC * kHmid * kWmid, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ho = 0; ho < kHmid; ++ho)
            for (uint32_t wo = 0; wo < kWmid; ++wo) {
                float acc = 0.0f;
                uint32_t ih = ho * 2, iw = wo * 2;
                for (uint32_t ci = 0; ci < kC; ++ci)
                    acc += bf16_to_f32(x_chw[(ci * kHin + ih) * kWin + iw]) *
                           bf16_to_f32(w_2d[co * kC + ci]);
                y_chw[(co * kHmid + ho) * kWmid + wo] = f32_to_bf16(acc);
            }
}

void bias_and_relu(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t i = 0; i < kHWmid; ++i) {
            float v = bf16_to_f32(y_chw[co * kHWmid + i]) + bias[co];
            if (v < 0.0f) v = 0.0f;
            y_chw[co * kHWmid + i] = f32_to_bf16(v);
        }
}
void bias_only(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t i = 0; i < kHWmid; ++i) {
            float v = bf16_to_f32(y_chw[co * kHWmid + i]) + bias[co];
            y_chw[co * kHWmid + i] = f32_to_bf16(v);
        }
}
void relu_inplace(std::vector<uint16_t>& y) {
    for (auto& v : y) if (bf16_to_f32(v) < 0.f) v = 0;
}
// y = bf16(a + b), element-wise.
void add_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
              std::vector<uint16_t>& y) {
    y.assign(a.size(), 0);
    for (size_t i = 0; i < a.size(); ++i)
        y[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

// ---------------------------------------------------------------------------
// Layout helpers (3×3 im2col with arbitrary stride; 1×1 stride-2 subsample;
// weight reshape; tile / untile).
// ---------------------------------------------------------------------------

void im2col_3x3(const std::vector<uint16_t>& x_chw,
                uint32_t Hi, uint32_t Wi, uint32_t Ho, uint32_t Wo,
                uint32_t stride, std::vector<uint16_t>& a) {
    a.assign(kKdim3 * (Ho * Wo), 0);
    for (uint32_t ci = 0; ci < kC; ++ci)
        for (uint32_t ki = 0; ki < kKH; ++ki)
            for (uint32_t kj = 0; kj < kKW; ++kj) {
                uint32_t row = ci * (kKH * kKW) + ki * kKW + kj;
                for (uint32_t ho = 0; ho < Ho; ++ho) {
                    int ih = static_cast<int>(ho * stride + ki) - static_cast<int>(kPad);
                    if (ih < 0 || ih >= static_cast<int>(Hi)) continue;
                    for (uint32_t wo = 0; wo < Wo; ++wo) {
                        int iw = static_cast<int>(wo * stride + kj) - static_cast<int>(kPad);
                        if (iw < 0 || iw >= static_cast<int>(Wi)) continue;
                        uint32_t col = ho * Wo + wo;
                        a[row * (Ho * Wo) + col] = x_chw[(ci * Hi + ih) * Wi + iw];
                    }
                }
            }
}

void weight_3x3_reshape(const std::vector<uint16_t>& w, std::vector<uint16_t>& m) {
    m.assign(kC * kKdim3, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ci = 0; ci < kC; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj)
                    m[co * kKdim3 + ci * (kKH * kKW) + ki * kKW + kj] =
                        w[((co * kC + ci) * kKH + ki) * kKW + kj];
}
void weight_1x1_reshape(const std::vector<uint16_t>& w, std::vector<uint16_t>& m) {
    m.assign(kC * kKdim1, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ci = 0; ci < kC; ++ci)
            m[co * kKdim1 + ci] = w[co * kC + ci];
}
void subsample_s2_to_2d(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& b) {
    b.assign(kC * kHWmid, 0);
    for (uint32_t ci = 0; ci < kC; ++ci)
        for (uint32_t ho = 0; ho < kHmid; ++ho) {
            uint32_t ih = ho * 2;
            for (uint32_t wo = 0; wo < kWmid; ++wo) {
                uint32_t iw = wo * 2;
                b[ci * kHWmid + ho * kWmid + wo] = x_chw[(ci * kHin + ih) * kWin + iw];
            }
        }
}

void tile_matrix(const std::vector<uint16_t>& m, uint32_t rows_t, uint32_t cols_t,
                 uint32_t col_dim, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(rows_t) * cols_t * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] =
                        m[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}
void untile_matrix(const std::vector<uint16_t>& t, uint32_t rows_t, uint32_t cols_t,
                   uint32_t col_dim, std::vector<uint16_t>& m) {
    m.assign(rows_t * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = t.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    m[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
}
void chw_to_tile_stream(const std::vector<uint16_t>& chw, std::vector<uint16_t>& t) {
    tile_matrix(chw, kMt, kNt, kHWmid, t);
}
void tile_stream_to_chw(const std::vector<uint16_t>& t, std::vector<uint16_t>& chw) {
    untile_matrix(t, kMt, kNt, kHWmid, chw);
}

}  // namespace

int main() try {
    const std::string kernel_root = required_env("TT_FOIL_KERNEL_DIR");
    const std::string conv_s2_dir = kernel_root + "/conv_s2";
    const std::string conv_dir    = kernel_root + "/conv";
    const std::string conv1x1_dir = kernel_root + "/conv1x1";
    const std::string add_dir     = kernel_root + "/residual_add";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Random tensors / BN-folded params --------------------------
    std::mt19937 rng(0x1a1e5c);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x(kC * kHin * kWin);
    for (auto& v : x) v = f32_to_bf16(u(rng));

    auto rand_w3 = [&]() {
        std::vector<uint16_t> w(kC * kC * kKH * kKW);
        for (auto& v : w) v = f32_to_bf16(0.05f * u(rng));
        return w;
    };
    auto rand_w1 = [&]() {
        std::vector<uint16_t> w(kC * kC);
        for (auto& v : w) v = f32_to_bf16(0.1f * u(rng));
        return w;
    };
    auto rand_bias = [&]() {
        std::vector<float> b(kC);
        for (auto& v : b) v = 0.1f * u(rng);
        return b;
    };

    // Block 1 (downsample) params
    auto W1d = rand_w3(); auto W2d = rand_w3(); auto Wsd = rand_w1();
    auto b1d = rand_bias(); auto b2d = rand_bias(); auto bsd = rand_bias();
    // Block 2 (basic) params
    auto W1b = rand_w3(); auto W2b = rand_w3();
    auto b1b = rand_bias(); auto b2b = rand_bias();

    // ---- Reference: chain both blocks in fp32 with one bf16 round per
    // device-equivalent stage. ----------------------------------------
    std::vector<uint16_t> y_ref;
    {
        // Block 1 — downsample
        std::vector<uint16_t> t1, t2, sk, y1;
        conv3x3_bf16(x,  W1d, kHin, kWin, kHmid, kWmid, /*stride=*/2, t1);
        bias_and_relu(t1, b1d);
        conv3x3_bf16(t1, W2d, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, t2);
        bias_only(t2, b2d);
        conv1x1_s2_bf16(x, Wsd, sk);
        bias_only(sk, bsd);
        add_bf16(t2, sk, y1);
        relu_inplace(y1);

        // Block 2 — basic (skip = y1)
        std::vector<uint16_t> t3, t4;
        conv3x3_bf16(y1, W1b, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, t3);
        bias_and_relu(t3, b1b);
        conv3x3_bf16(t3, W2b, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, t4);
        bias_only(t4, b2b);
        add_bf16(t4, y1, y_ref);
        relu_inplace(y_ref);
    }

    // ---- Device ----------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // Matmul-shaped DRAM scratch (reused across every conv invocation).
    const uint32_t w3_bytes = kMt * kKt_3 * kTileBytes;
    const uint32_t a3_bytes = kKt_3 * kNt * kTileBytes;
    const uint32_t w1_bytes = kMt * kKt_1 * kTileBytes;
    const uint32_t a1_bytes = kKt_1 * kNt * kTileBytes;
    const uint32_t y_bytes  = kMt * kNt * kTileBytes;
    const uint32_t add_bytes = kRaNt * kTileBytes;
    static_assert(y_bytes == add_bytes, "matmul output and add tile streams match");

    auto buf_W3   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w3_bytes,  core);
    auto buf_A3   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a3_bytes,  core);
    auto buf_W1   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w1_bytes,  core);
    auto buf_A1   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a1_bytes,  core);
    // Four DRAM result buffers so we can hold {y1, sk-or-skip-input, main-of-block2, final-out}
    // without overwriting them prematurely (the residual_add reads two of them).
    auto buf_Ymain = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);
    auto buf_Yskip = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);
    auto buf_Y1    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);
    auto buf_Yout  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    auto noc_of = [&](auto& buf) { return tt::foil::make_noc_dram_addr(*dev, buf->device_addr); };
    uint64_t W3_noc    = noc_of(buf_W3);
    uint64_t A3_noc    = noc_of(buf_A3);
    uint64_t W1_noc    = noc_of(buf_W1);
    uint64_t A1_noc    = noc_of(buf_A1);
    uint64_t Ymain_noc = noc_of(buf_Ymain);
    uint64_t Yskip_noc = noc_of(buf_Yskip);
    uint64_t Y1_noc    = noc_of(buf_Y1);
    uint64_t Yout_noc  = noc_of(buf_Yout);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    using R = tt::foil::RiscBinary;

    auto load = [&](const std::string& dir) {
        std::array<R, 5> bins = {{
            {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
            {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
            {R::RiscId::TRISC0, dir + "/compute.trisc0.elf"},
            {R::RiscId::TRISC1, dir + "/compute.trisc1.elf"},
            {R::RiscId::TRISC2, dir + "/compute.trisc2.elf"},
        }};
        return tt::foil::load_kernel(*dev, bins, core);
    };
    auto k_conv_s2 = load(conv_s2_dir);
    auto k_conv    = load(conv_dir);
    auto k_conv1x1 = load(conv1x1_dir);
    auto k_add     = load(add_dir);

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};

    // Run an arbitrary matmul-shaped conv kernel.
    auto run_matmul = [&](tt::foil::Kernel& k,
                          tt::foil::Buffer& buf_W, tt::foil::Buffer& buf_A,
                          tt::foil::Buffer& buf_Y,
                          uint64_t W_noc, uint64_t A_noc, uint64_t Y_noc,
                          const std::vector<uint16_t>& w_tiles,
                          const std::vector<uint16_t>& a_tiles,
                          uint32_t Kt,
                          std::vector<uint16_t>& y_chw_out) {
        tt::foil::write_buffer(*dev, buf_W, w_tiles.data(), w_tiles.size() * sizeof(uint16_t));
        tt::foil::write_buffer(*dev, buf_A, a_tiles.data(), a_tiles.size() * sizeof(uint16_t));
        std::vector<uint8_t> zero(y_bytes, 0);
        tt::foil::write_buffer(*dev, buf_Y, zero.data(), y_bytes);

        std::array<uint32_t, 7> ra_brisc = {
            lo(W_noc), hi(W_noc),
            lo(A_noc), hi(A_noc),
            kMt, Kt, kNt,
        };
        std::array<uint32_t, 3> ra_ncrisc = { lo(Y_noc), hi(Y_noc), kMt * kNt };
        tt::foil::set_runtime_args(*dev, k, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::NCRISC, ra_ncrisc);
        tt::foil::register_cbs(*dev, k, cbs);
        tt::foil::execute(*dev, k);

        std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, buf_Y, y_tiles.data(), y_bytes);
        tile_stream_to_chw(y_tiles, y_chw_out);
    };

    // Run residual_add: A := buf_main, B := buf_skip, OUT := buf_out.
    auto run_add = [&](tt::foil::Buffer& buf_a_dram,
                       tt::foil::Buffer& buf_b_dram,
                       tt::foil::Buffer& buf_out_dram,
                       uint64_t a_noc, uint64_t b_noc, uint64_t out_noc,
                       const std::vector<uint16_t>& a_tiles,
                       const std::vector<uint16_t>& b_tiles,
                       std::vector<uint16_t>& y_chw_out) {
        tt::foil::write_buffer(*dev, buf_a_dram, a_tiles.data(), add_bytes);
        tt::foil::write_buffer(*dev, buf_b_dram, b_tiles.data(), add_bytes);
        std::vector<uint8_t> zero(add_bytes, 0);
        tt::foil::write_buffer(*dev, buf_out_dram, zero.data(), add_bytes);

        std::array<uint32_t, 4> ra_brisc = {
            lo(a_noc), hi(a_noc), lo(b_noc), hi(b_noc),
        };
        std::array<uint32_t, 2> ra_ncrisc = { lo(out_noc), hi(out_noc) };
        tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::NCRISC, ra_ncrisc);
        tt::foil::register_cbs(*dev, *k_add, cbs);
        tt::foil::execute(*dev, *k_add);

        std::vector<uint16_t> y_tiles(kRaNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, buf_out_dram, y_tiles.data(), add_bytes);
        tile_stream_to_chw(y_tiles, y_chw_out);
    };

    // ================ BLOCK 1: downsample ============================
    std::vector<uint16_t> t1, t2, sk, y1;
    {
        // Stage 1a — Conv₁ (3×3 s=2)
        std::vector<uint16_t> wmat, wt, amat, at;
        weight_3x3_reshape(W1d, wmat);
        tile_matrix(wmat, kMt, kKt_3, kKdim3, wt);
        im2col_3x3(x, kHin, kWin, kHmid, kWmid, /*stride=*/2, amat);
        tile_matrix(amat, kKt_3, kNt, kHWmid, at);
        run_matmul(*k_conv_s2, *buf_W3, *buf_A3, *buf_Ymain,
                   W3_noc, A3_noc, Ymain_noc, wt, at, kKt_3, t1);
        bias_and_relu(t1, b1d);

        // Stage 1b — Conv₂ (3×3 s=1) on t1
        weight_3x3_reshape(W2d, wmat);
        tile_matrix(wmat, kMt, kKt_3, kKdim3, wt);
        im2col_3x3(t1, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, amat);
        tile_matrix(amat, kKt_3, kNt, kHWmid, at);
        run_matmul(*k_conv, *buf_W3, *buf_A3, *buf_Ymain,
                   W3_noc, A3_noc, Ymain_noc, wt, at, kKt_3, t2);
        bias_only(t2, b2d);

        // Stage 1c — Skip Conv_s (1×1 s=2) on original x
        std::vector<uint16_t> swmat, swt, samat, sat;
        weight_1x1_reshape(Wsd, swmat);
        tile_matrix(swmat, kMt, kKt_1, kKdim1, swt);
        subsample_s2_to_2d(x, samat);
        tile_matrix(samat, kKt_1, kNt, kHWmid, sat);
        run_matmul(*k_conv1x1, *buf_W1, *buf_A1, *buf_Yskip,
                   W1_noc, A1_noc, Yskip_noc, swt, sat, kKt_1, sk);
        bias_only(sk, bsd);

        // Stage 1d — residual add + ReLU
        std::vector<uint16_t> t2_t, sk_t;
        chw_to_tile_stream(t2, t2_t);
        chw_to_tile_stream(sk, sk_t);
        run_add(*buf_Ymain, *buf_Yskip, *buf_Y1,
                Ymain_noc, Yskip_noc, Y1_noc, t2_t, sk_t, y1);
        relu_inplace(y1);
    }

    // ================ BLOCK 2: basic (skip = y1) =====================
    std::vector<uint16_t> y_dev;
    {
        // Stage 2a — Conv₁ (3×3 s=1) on y1
        std::vector<uint16_t> wmat, wt, amat, at, t3;
        weight_3x3_reshape(W1b, wmat);
        tile_matrix(wmat, kMt, kKt_3, kKdim3, wt);
        im2col_3x3(y1, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, amat);
        tile_matrix(amat, kKt_3, kNt, kHWmid, at);
        run_matmul(*k_conv, *buf_W3, *buf_A3, *buf_Ymain,
                   W3_noc, A3_noc, Ymain_noc, wt, at, kKt_3, t3);
        bias_and_relu(t3, b1b);

        // Stage 2b — Conv₂ (3×3 s=1)
        std::vector<uint16_t> t4;
        weight_3x3_reshape(W2b, wmat);
        tile_matrix(wmat, kMt, kKt_3, kKdim3, wt);
        im2col_3x3(t3, kHmid, kWmid, kHmid, kWmid, /*stride=*/1, amat);
        tile_matrix(amat, kKt_3, kNt, kHWmid, at);
        run_matmul(*k_conv, *buf_W3, *buf_A3, *buf_Ymain,
                   W3_noc, A3_noc, Ymain_noc, wt, at, kKt_3, t4);
        bias_only(t4, b2b);

        // Stage 2c — skip + ReLU. Skip is y1, sitting in buf_Y1.
        std::vector<uint16_t> t4_t, y1_t;
        chw_to_tile_stream(t4, t4_t);
        chw_to_tile_stream(y1, y1_t);
        // Re-write y1's tiles into buf_Yskip so run_add can use it as B.
        run_add(*buf_Ymain, *buf_Yskip, *buf_Yout,
                Ymain_noc, Yskip_noc, Yout_noc, t4_t, y1_t, y_dev);
        relu_inplace(y_dev);
    }

    // ---- Compare ---------------------------------------------------
    // Eight bf16 rounding stages stack across two blocks; allow ~0.3
    // absolute / 6 % relative. The signals here sit in roughly [-1, 1.5].
    const float kAbsTol = 0.30f;
    const float kRelTol = 0.06f;
    uint32_t bad = 0, first_bad = static_cast<uint32_t>(y_dev.size());
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (uint32_t i = 0; i < y_dev.size(); ++i) {
        float got = bf16_to_f32(y_dev[i]);
        float exp = bf16_to_f32(y_ref[i]);
        float d = std::fabs(got - exp);
        float ref = std::fabs(exp);
        float tol = std::max(kAbsTol, kRelTol * ref);
        if (d > worst_abs) worst_abs = d;
        if (ref > 0.f && (d / ref) > worst_rel) worst_rel = d / ref;
        if (d > tol) {
            if (first_bad == y_dev.size()) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_layer: %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, y_dev.size(), first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(y_ref[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_layer: FAIL");
        return 1;
    }

    std::printf("test_layer: PASS  "
                "(downsample %ux%u→%ux%u then basic at %ux%u, C=%u, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kHin, kWin, kHmid, kWmid, kHmid, kWmid, kC,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_layer: FAIL — %s\n", e.what());
    return 1;
}
