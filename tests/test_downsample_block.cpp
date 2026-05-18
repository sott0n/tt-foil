// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ResNet downsample block — the block at every stage boundary where the
// spatial dims halve and (typically) the channel count doubles. tt-foil
// composes it host-orchestrated on top of three device kernels:
//
//   main path:
//     Conv₁  (3×3, stride 2, pad 1, Cin→Cout)  + b₁'  + ReLU
//     Conv₂  (3×3, stride 1, pad 1, Cout→Cout) + b₂'
//   skip path:
//     Conv_s (1×1, stride 2, pad 0, Cin→Cout)  + bₛ'
//   y = ReLU( main + skip )
//
// (W, b) for each conv come from BN folded into the conv — exactly the
// algebra exercised by tests/test_bn_fold.cpp.
//
// For test simplicity Cin == Cout == 32 here, but the layout matches the
// real torchvision BasicBlock-with-downsample.
//
// Tile mapping for this shape (Cin=Cout=32, H_in=W_in=16, H_out=W_out=8):
//   Conv₁ (3×3 s=2)  M=Cout=32 →Mt=1, K=Cin*9=288 →Kt=9, N=H_out*W_out=64 →Nt=2
//   Conv₂ (3×3 s=1)  M=32 →Mt=1, K=288 →Kt=9, N=64 →Nt=2
//   Conv_s (1×1)     M=32 →Mt=1, K=Cin=32 →Kt=1, N=64 →Nt=2
//   skip add          RA_NT = Mt*Nt = 2
//
// Kernel dirs (built by models/downsample_block/build_kernels.sh):
//   $TT_FOIL_KERNEL_DIR/conv_s2/        (conv_3x3_s2 prebuilt, Mt=1 Kt=9 Nt=2)
//   $TT_FOIL_KERNEL_DIR/conv/           (conv_3x3 prebuilt,    Mt=1 Kt=9 Nt=2)
//   $TT_FOIL_KERNEL_DIR/conv1x1/        (conv_1x1 fresh,       Mt=1 Kt=1 Nt=2)
//   $TT_FOIL_KERNEL_DIR/residual_add/   (residual_add fresh,   RA_NT=2)

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

constexpr uint32_t kCin   = 32;
constexpr uint32_t kCout  = 32;
constexpr uint32_t kHin   = 16;
constexpr uint32_t kWin   = 16;
constexpr uint32_t kHout  = 8;
constexpr uint32_t kWout  = 8;

constexpr uint32_t kKH = 3, kKW = 3, kPad = 1, kStride = 2;

// Shared matmul shape: M=Cout=32→Mt=1, N=HW_out=64→Nt=2. K differs per
// conv (Kt=9 for 3×3, Kt=1 for 1×1).
constexpr uint32_t kMt    = kCout / kTileH;           // 1
constexpr uint32_t kNt    = (kHout * kWout) / kTileW; // 2
constexpr uint32_t kKt_3  = (kCin * kKH * kKW) / kTileW;     // 9
constexpr uint32_t kKt_1  =  kCin / kTileW;                  // 1
constexpr uint32_t kKdim3 = kKt_3 * kTileW;           // 288
constexpr uint32_t kKdim1 = kKt_1 * kTileW;           // 32
constexpr uint32_t kHWout = kHout * kWout;            // 64
constexpr uint32_t kRaNt  = kMt * kNt;                // 2

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// ---------------------------------------------------------------------------
// Reference math (fp32 inner, single bf16 round per device-equivalent stage)
// ---------------------------------------------------------------------------

void conv3x3_bf16(const std::vector<uint16_t>& x_chw,
                  const std::vector<uint16_t>& w_cchw,
                  uint32_t Hi, uint32_t Wi, uint32_t Ho, uint32_t Wo,
                  uint32_t stride,
                  std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * Ho * Wo, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ho = 0; ho < Ho; ++ho)
            for (uint32_t wo = 0; wo < Wo; ++wo) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kCin; ++ci)
                    for (uint32_t ki = 0; ki < kKH; ++ki)
                        for (uint32_t kj = 0; kj < kKW; ++kj) {
                            int ih = static_cast<int>(ho * stride + ki) - static_cast<int>(kPad);
                            int iw = static_cast<int>(wo * stride + kj) - static_cast<int>(kPad);
                            if (ih < 0 || ih >= static_cast<int>(Hi) ||
                                iw < 0 || iw >= static_cast<int>(Wi)) continue;
                            acc += bf16_to_f32(x_chw[(ci * Hi + ih) * Wi + iw]) *
                                   bf16_to_f32(w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj]);
                        }
                y_chw[(co * Ho + ho) * Wo + wo] = f32_to_bf16(acc);
            }
}

void conv1x1_s2_bf16(const std::vector<uint16_t>& x_chw,
                     const std::vector<uint16_t>& w_2d,    // (Cout, Cin)
                     std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * kHout * kWout, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ho = 0; ho < kHout; ++ho)
            for (uint32_t wo = 0; wo < kWout; ++wo) {
                float acc = 0.0f;
                uint32_t ih = ho * kStride;
                uint32_t iw = wo * kStride;
                for (uint32_t ci = 0; ci < kCin; ++ci) {
                    acc += bf16_to_f32(x_chw[(ci * kHin + ih) * kWin + iw]) *
                           bf16_to_f32(w_2d[co * kCin + ci]);
                }
                y_chw[(co * kHout + ho) * kWout + wo] = f32_to_bf16(acc);
            }
}

void bias_and_relu(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t i = 0; i < kHWout; ++i) {
            float v = bf16_to_f32(y_chw[co * kHWout + i]) + bias[co];
            if (v < 0.0f) v = 0.0f;
            y_chw[co * kHWout + i] = f32_to_bf16(v);
        }
}
void bias_only(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t i = 0; i < kHWout; ++i) {
            float v = bf16_to_f32(y_chw[co * kHWout + i]) + bias[co];
            y_chw[co * kHWout + i] = f32_to_bf16(v);
        }
}
void relu_inplace(std::vector<uint16_t>& y) {
    for (auto& v : y) if ((bf16_to_f32(v)) < 0.f) v = 0;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

// Generic im2col for (3×3, pad=1, stride=s) over input (Cin, Hi, Wi).
// Produces row-major (Cin*9, Ho*Wo) matrix.
void im2col_3x3(const std::vector<uint16_t>& x_chw,
                uint32_t Hi, uint32_t Wi,
                uint32_t Ho, uint32_t Wo, uint32_t stride,
                std::vector<uint16_t>& a) {
    a.assign(kKdim3 * (Ho * Wo), 0);
    for (uint32_t ci = 0; ci < kCin; ++ci)
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

void weight_3x3_reshape(const std::vector<uint16_t>& w_cchw,
                        std::vector<uint16_t>& w_mat) {
    w_mat.assign(kCout * kKdim3, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj) {
                    uint32_t col = ci * (kKH * kKW) + ki * kKW + kj;
                    w_mat[co * kKdim3 + col] =
                        w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj];
                }
}

// 1×1 conv reshape: weight is just (Cout, Cin) — no spatial axes.
void weight_1x1_reshape(const std::vector<uint16_t>& w_2d,
                        std::vector<uint16_t>& w_mat) {
    w_mat.assign(kCout * kKdim1, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            w_mat[co * kKdim1 + ci] = w_2d[co * kCin + ci];
}

// For 1×1 stride=2 skip: just take every other (h, w) from the input and
// flatten to (Cin, HW_out). This is the matmul's B operand (no kernel
// dimension to unroll into rows because K=1×1=1).
void subsample_s2_to_2d(const std::vector<uint16_t>& x_chw,
                        std::vector<uint16_t>& b) {
    b.assign(kCin * kHWout, 0);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t ho = 0; ho < kHout; ++ho) {
            uint32_t ih = ho * kStride;
            for (uint32_t wo = 0; wo < kWout; ++wo) {
                uint32_t iw = wo * kStride;
                b[ci * kHWout + ho * kWout + wo] =
                    x_chw[(ci * kHin + ih) * kWin + iw];
            }
        }
}

void tile_matrix(const std::vector<uint16_t>& m_rm,
                 uint32_t rows_t, uint32_t cols_t, uint32_t col_dim,
                 std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(rows_t) * cols_t * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] =
                        m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}

void untile_matrix(const std::vector<uint16_t>& tiles,
                   uint32_t rows_t, uint32_t cols_t, uint32_t col_dim,
                   std::vector<uint16_t>& m_rm) {
    m_rm.assign(rows_t * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = tiles.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
}

void chw_out_to_tile_stream(const std::vector<uint16_t>& chw,
                            std::vector<uint16_t>& tiles) {
    tile_matrix(chw, kMt, kNt, kHWout, tiles);
}

void tile_stream_to_chw_out(const std::vector<uint16_t>& tiles,
                            std::vector<uint16_t>& chw) {
    untile_matrix(tiles, kMt, kNt, kHWout, chw);
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

    // ---- Inputs / BN-folded params ----------------------------------
    std::mt19937 rng(0xd0a5e5);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x_chw(kCin * kHin * kWin);
    for (auto& v : x_chw) v = f32_to_bf16(u(rng));

    auto rand_3x3 = [&](std::vector<uint16_t>& w) {
        w.assign(kCout * kCin * kKH * kKW, 0);
        for (auto& v : w) v = f32_to_bf16(0.05f * u(rng));
    };
    std::vector<uint16_t> w1, w2;
    rand_3x3(w1);
    rand_3x3(w2);

    std::vector<uint16_t> ws(kCout * kCin);  // 1×1 skip weights, (Cout, Cin)
    for (auto& v : ws) v = f32_to_bf16(0.1f * u(rng));

    auto rand_bias = [&]() {
        std::vector<float> b(kCout);
        for (auto& v : b) v = 0.1f * u(rng);
        return b;
    };
    auto b1 = rand_bias();
    auto b2 = rand_bias();
    auto bs = rand_bias();

    // ---- Reference -------------------------------------------------
    std::vector<uint16_t> y_ref;
    {
        std::vector<uint16_t> t1, t2, sk;
        conv3x3_bf16(x_chw, w1, kHin, kWin, kHout, kWout, kStride, t1);
        bias_and_relu(t1, b1);
        conv3x3_bf16(t1, w2, kHout, kWout, kHout, kWout, /*stride=*/1, t2);
        bias_only(t2, b2);
        conv1x1_s2_bf16(x_chw, ws, sk);
        bias_only(sk, bs);
        y_ref.assign(kCout * kHWout, 0);
        for (uint32_t i = 0; i < y_ref.size(); ++i)
            y_ref[i] = f32_to_bf16(bf16_to_f32(t2[i]) + bf16_to_f32(sk[i]));
        relu_inplace(y_ref);
    }

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // Matmul scratch sizing — keep two sets:
    //   3×3 conv: K dim is kKdim3 (=288 = 9*32)
    //   1×1 conv: K dim is kKdim1 (=32 = 1*32). Smaller, fits the same
    //             buffer if we reuse buf_W3 / buf_A3 since their byte
    //             sizes are larger. To keep ranges precise we allocate
    //             dedicated buffers for the 1×1 stage.
    const uint32_t w3_bytes = kMt * kKt_3 * kTileBytes;
    const uint32_t a3_bytes = kKt_3 * kNt * kTileBytes;
    const uint32_t y_bytes  = kMt * kNt * kTileBytes;
    const uint32_t w1_bytes = kMt * kKt_1 * kTileBytes;
    const uint32_t a1_bytes = kKt_1 * kNt * kTileBytes;
    const uint32_t add_bytes = kRaNt * kTileBytes;

    // Conv (main path) buffers — written fresh each conv call.
    auto buf_W3a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w3_bytes,  core);
    auto buf_A3a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a3_bytes,  core);
    auto buf_Ymain = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes,   core);

    // 1×1 skip path buffers.
    auto buf_Ws    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w1_bytes,  core);
    auto buf_As    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a1_bytes,  core);
    auto buf_Yskip = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes,   core);

    // Residual add output (= block output).
    auto buf_Yout  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, add_bytes, core);

    // One set of CB scratch in L1 — same shape for every kernel here
    // (one bf16 tile per CB slot).
    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    auto noc_of = [&](const auto& buf) {
        return tt::foil::make_noc_dram_addr(*dev, buf->device_addr);
    };
    uint64_t W3_noc    = noc_of(buf_W3a);
    uint64_t A3_noc    = noc_of(buf_A3a);
    uint64_t Ymain_noc = noc_of(buf_Ymain);
    uint64_t Ws_noc    = noc_of(buf_Ws);
    uint64_t As_noc    = noc_of(buf_As);
    uint64_t Yskip_noc = noc_of(buf_Yskip);
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

    // Runs a matmul-shaped conv: writes weight + im2col tiles into buf_W /
    // buf_A, fires the kernel, reads the (Mt, Nt) output tile stream
    // back, untiles into row-major (Cout, HW_out).
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
        std::array<uint32_t, 3> ra_ncrisc = {
            lo(Y_noc), hi(Y_noc), kMt * kNt,
        };
        tt::foil::set_runtime_args(*dev, k, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::NCRISC, ra_ncrisc);
        tt::foil::register_cbs(*dev, k, cbs);
        tt::foil::execute(*dev, k);

        std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, buf_Y, y_tiles.data(), y_bytes);
        tile_stream_to_chw_out(y_tiles, y_chw_out);
    };

    // ---- Stage 1 — Conv₁ (3×3 s=2) ----------------------------------
    std::vector<uint16_t> w1_mat, w1_tiles, a1_mat, a1_tiles;
    weight_3x3_reshape(w1, w1_mat);
    tile_matrix(w1_mat, kMt, kKt_3, kKdim3, w1_tiles);
    im2col_3x3(x_chw, kHin, kWin, kHout, kWout, kStride, a1_mat);
    tile_matrix(a1_mat, kKt_3, kNt, kHWout, a1_tiles);

    std::vector<uint16_t> t1;
    run_matmul(*k_conv_s2, *buf_W3a, *buf_A3a, *buf_Ymain,
               W3_noc, A3_noc, Ymain_noc,
               w1_tiles, a1_tiles, kKt_3, t1);
    bias_and_relu(t1, b1);

    // ---- Stage 2 — Conv₂ (3×3 s=1) on the downsampled feature map --
    std::vector<uint16_t> w2_mat, w2_tiles, a2_mat, a2_tiles;
    weight_3x3_reshape(w2, w2_mat);
    tile_matrix(w2_mat, kMt, kKt_3, kKdim3, w2_tiles);
    im2col_3x3(t1, kHout, kWout, kHout, kWout, /*stride=*/1, a2_mat);
    tile_matrix(a2_mat, kKt_3, kNt, kHWout, a2_tiles);

    std::vector<uint16_t> t2;
    run_matmul(*k_conv, *buf_W3a, *buf_A3a, *buf_Ymain,
               W3_noc, A3_noc, Ymain_noc,
               w2_tiles, a2_tiles, kKt_3, t2);
    bias_only(t2, b2);

    // ---- Stage 3 — Conv_s (1×1 s=2) on the original input ----------
    // Host does the s=2 subsampling so the device-side matmul is a plain
    // 1×1 conv (no kernel unroll).
    std::vector<uint16_t> ws_mat, ws_tiles, as_mat, as_tiles;
    weight_1x1_reshape(ws, ws_mat);
    tile_matrix(ws_mat, kMt, kKt_1, kKdim1, ws_tiles);
    subsample_s2_to_2d(x_chw, as_mat);
    tile_matrix(as_mat, kKt_1, kNt, kHWout, as_tiles);

    std::vector<uint16_t> sk;
    run_matmul(*k_conv1x1, *buf_Ws, *buf_As, *buf_Yskip,
               Ws_noc, As_noc, Yskip_noc,
               ws_tiles, as_tiles, kKt_1, sk);
    bias_only(sk, bs);

    // ---- Stage 4 — device residual_add: t2 + sk ---------------------
    std::vector<uint16_t> t2_tiles, sk_tiles;
    chw_out_to_tile_stream(t2, t2_tiles);
    chw_out_to_tile_stream(sk, sk_tiles);

    // Reuse buf_Ymain and buf_Yskip as the add inputs (NOC-addressable
    // already), reuse buf_Yout for the sum.
    tt::foil::write_buffer(*dev, *buf_Ymain, t2_tiles.data(), add_bytes);
    tt::foil::write_buffer(*dev, *buf_Yskip, sk_tiles.data(), add_bytes);
    std::vector<uint8_t> zero(add_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_Yout, zero.data(), add_bytes);

    std::array<uint32_t, 4> ra_add_brisc = {
        lo(Ymain_noc), hi(Ymain_noc),
        lo(Yskip_noc), hi(Yskip_noc),
    };
    std::array<uint32_t, 2> ra_add_ncrisc = {
        lo(Yout_noc), hi(Yout_noc),
    };
    tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::BRISC,  ra_add_brisc);
    tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::NCRISC, ra_add_ncrisc);
    tt::foil::register_cbs(*dev, *k_add, cbs);
    tt::foil::execute(*dev, *k_add);

    std::vector<uint16_t> y_tiles(kRaNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_Yout, y_tiles.data(), add_bytes);
    std::vector<uint16_t> y_dev;
    tile_stream_to_chw_out(y_tiles, y_dev);
    relu_inplace(y_dev);

    // ---- Compare ----------------------------------------------------
    const float kAbsTol = 0.20f;     // a bit looser than basic_block:
    const float kRelTol = 0.04f;     // five bf16 rounds instead of four.
    uint32_t bad = 0, first_bad = static_cast<uint32_t>(y_dev.size());
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (uint32_t i = 0; i < y_dev.size(); ++i) {
        float got = bf16_to_f32(y_dev[i]);
        float exp = bf16_to_f32(y_ref[i]);
        float d   = std::fabs(got - exp);
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
            "test_downsample_block: %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, y_dev.size(), first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(y_ref[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_downsample_block: FAIL");
        return 1;
    }

    std::printf("test_downsample_block: PASS  "
                "(Cin=%u Cout=%u %ux%u→%ux%u, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kCin, kCout, kHin, kWin, kHout, kWout,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_downsample_block: FAIL — %s\n", e.what());
    return 1;
}
