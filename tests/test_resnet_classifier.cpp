// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Full ResNet classifier — every arithmetic op runs on device. The host
// only handles layout (im2col, tile/untile, chw↔hwc, window gather) and
// reference comparison. The "image in, logits out" test:
//
//   x        : (C=32, 32, 32)
//   stem     : Conv₇ₓ₇ s=2 pad=3 → bias + ReLU → Maxpool₃ₓ₃ s=2 pad=1
//   block₁   : basic_block (3×3 s=1 ×2 + identity skip)  on (32, 8, 8)
//   block₂   : basic_block (3×3 s=1 ×2 + identity skip)  on (32, 8, 8)
//   gap      : global avg pool over (h, w)               → (32,)   device
//   fc       : W · gap + bias                            → 32-way logits  device
//
// Seven distinct kernel programs across the chain (Conv₇ₓ₇, Maxpool₃ₓ₃,
// Conv₃ₓ₃, residual_add, bias_relu_post, global_avg_pool, FC matmul).
// They don't all fit in Blackhole's per-core KERNEL_CONFIG region
// (~69 KB) simultaneously, so the test loads three phase-specific
// kernel sets in turn with release_kernels() between phases:
//   • stem  : conv_7x7 + maxpool_3x3 + bias_relu_post
//   • blocks: conv_3x3 + residual_add + bias_relu_post
//   • tail  : global_avg_pool + fc + bias_relu_post
//
// bias_relu_post is reused in every phase (Nt is a runtime arg). The
// residual-add ReLU is also delivered by bias_relu_post with bias=0,
// relu=1.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

// ---- Shape ----
constexpr uint32_t kC      = 32;
constexpr uint32_t kHin    = 32;
constexpr uint32_t kWin    = 32;
constexpr uint32_t kHmid   = 16;     // after Conv₇ₓ₇
constexpr uint32_t kWmid   = 16;
constexpr uint32_t kHpost  = 8;      // after Maxpool / feeds blocks
constexpr uint32_t kWpost  = 8;

constexpr uint32_t kK7      = 7, kPad7 = 3, kStride7 = 2;
constexpr uint32_t kK3      = 3, kPad3 = 1;   // 3×3 s=1 (blocks)
constexpr uint32_t kPadP    = 1, kStrideP = 2;

// Conv₇ₓ₇ matmul (Mt=1, Kt=49, Nt=8)
constexpr uint32_t kMt_7 = kC / kTileH;
constexpr uint32_t kKt_7 = (kC * kK7 * kK7) / kTileW;
constexpr uint32_t kNt_7 = (kHmid * kWmid) / kTileW;
constexpr uint32_t kKdim7 = kKt_7 * kTileW;
constexpr uint32_t kHWmid = kHmid * kWmid;

// 3×3 conv matmul for blocks (Mt=1, Kt=9, Nt=2)
constexpr uint32_t kMt_3 = kC / kTileH;
constexpr uint32_t kKt_3 = (kC * kK3 * kK3) / kTileW;
constexpr uint32_t kNt_3 = (kHpost * kWpost) / kTileW;
constexpr uint32_t kKdim3 = kKt_3 * kTileW;
constexpr uint32_t kHWpost = kHpost * kWpost;

// residual_add (RA_NT=2)
constexpr uint32_t kRaNt = kMt_3 * kNt_3;       // 2

// maxpool stream tile count (RA_NT-equivalent)
constexpr uint32_t kNtPool = (kHpost * kWpost) / kTileH;   // 2

// Classifier-tail FC matmul: M=Ncl=32, K=C=32, N=1 tile (padded).
constexpr uint32_t kNcl    = 32;
constexpr uint32_t kMt_fc  = kNcl / kTileH;     // 1
constexpr uint32_t kKt_fc  = kC   / kTileW;     // 1
constexpr uint32_t kNt_fc  = 1;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// ---------------------------------------------------------------------------
// Reference math (one bf16 round per device-equivalent stage).
// ---------------------------------------------------------------------------

void conv7x7_s2_bf16(const std::vector<uint16_t>& x_chw,
                     const std::vector<uint16_t>& w_cchw,
                     std::vector<uint16_t>& y_chw) {
    y_chw.assign(kC * kHmid * kWmid, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ho = 0; ho < kHmid; ++ho)
            for (uint32_t wo = 0; wo < kWmid; ++wo) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kC; ++ci)
                    for (uint32_t ki = 0; ki < kK7; ++ki)
                        for (uint32_t kj = 0; kj < kK7; ++kj) {
                            int ih = static_cast<int>(ho * kStride7 + ki) - static_cast<int>(kPad7);
                            int iw = static_cast<int>(wo * kStride7 + kj) - static_cast<int>(kPad7);
                            if (ih < 0 || ih >= static_cast<int>(kHin) ||
                                iw < 0 || iw >= static_cast<int>(kWin)) continue;
                            acc += bf16_to_f32(x_chw[(ci * kHin + ih) * kWin + iw]) *
                                   bf16_to_f32(w_cchw[((co * kC + ci) * kK7 + ki) * kK7 + kj]);
                        }
                y_chw[(co * kHmid + ho) * kWmid + wo] = f32_to_bf16(acc);
            }
}

void conv3x3_s1_bf16(const std::vector<uint16_t>& x_chw,
                     const std::vector<uint16_t>& w_cchw,
                     std::vector<uint16_t>& y_chw) {
    y_chw.assign(kC * kHpost * kWpost, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ho = 0; ho < kHpost; ++ho)
            for (uint32_t wo = 0; wo < kWpost; ++wo) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kC; ++ci)
                    for (uint32_t ki = 0; ki < kK3; ++ki)
                        for (uint32_t kj = 0; kj < kK3; ++kj) {
                            int ih = static_cast<int>(ho + ki) - static_cast<int>(kPad3);
                            int iw = static_cast<int>(wo + kj) - static_cast<int>(kPad3);
                            if (ih < 0 || ih >= static_cast<int>(kHpost) ||
                                iw < 0 || iw >= static_cast<int>(kWpost)) continue;
                            acc += bf16_to_f32(x_chw[(ci * kHpost + ih) * kWpost + iw]) *
                                   bf16_to_f32(w_cchw[((co * kC + ci) * kK3 + ki) * kK3 + kj]);
                        }
                y_chw[(co * kHpost + ho) * kWpost + wo] = f32_to_bf16(acc);
            }
}

void bias_and_relu_chw_h(std::vector<uint16_t>& y_chw, uint32_t H, uint32_t W,
                         const std::vector<float>& bias) {
    const uint32_t hw = H * W;
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t i = 0; i < hw; ++i) {
            float v = bf16_to_f32(y_chw[co * hw + i]) + bias[co];
            if (v < 0.0f) v = 0.0f;
            y_chw[co * hw + i] = f32_to_bf16(v);
        }
}
void bias_only_chw_h(std::vector<uint16_t>& y_chw, uint32_t H, uint32_t W,
                     const std::vector<float>& bias) {
    const uint32_t hw = H * W;
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t i = 0; i < hw; ++i) {
            float v = bf16_to_f32(y_chw[co * hw + i]) + bias[co];
            y_chw[co * hw + i] = f32_to_bf16(v);
        }
}
void add_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
              std::vector<uint16_t>& y) {
    y.assign(a.size(), 0);
    for (size_t i = 0; i < a.size(); ++i)
        y[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}
void relu_inplace(std::vector<uint16_t>& y) {
    for (auto& v : y) if (bf16_to_f32(v) < 0.f) v = 0;
}

// CHW maxpool over (kHmid, kWmid, C) → (kHpost, kWpost, C). Result is
// in HWC layout because that's how the device maxpool wants its input/
// output too.
void maxpool3x3_chw_to_hwc_ref(const std::vector<uint16_t>& x_chw,
                               std::vector<uint16_t>& y_hwc) {
    y_hwc.assign(kHpost * kWpost * kC, 0);
    for (uint32_t oh = 0; oh < kHpost; ++oh)
        for (uint32_t ow = 0; ow < kWpost; ++ow)
            for (uint32_t c = 0; c < kC; ++c) {
                float m = -std::numeric_limits<float>::infinity();
                for (int di = 0; di < 3; ++di)
                    for (int dj = 0; dj < 3; ++dj) {
                        int ih = static_cast<int>(oh) * static_cast<int>(kStrideP) +
                                 di - static_cast<int>(kPadP);
                        int iw = static_cast<int>(ow) * static_cast<int>(kStrideP) +
                                 dj - static_cast<int>(kPadP);
                        if (ih < 0 || ih >= static_cast<int>(kHmid) ||
                            iw < 0 || iw >= static_cast<int>(kWmid)) continue;
                        float v = bf16_to_f32(x_chw[(c * kHmid + static_cast<uint32_t>(ih)) *
                                                    kWmid + static_cast<uint32_t>(iw)]);
                        if (v > m) m = v;
                    }
                y_hwc[(oh * kWpost + ow) * kC + c] = f32_to_bf16(m);
            }
}

// ---------------------------------------------------------------------------
// Layout helpers.
// ---------------------------------------------------------------------------

void im2col_7x7(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& a) {
    a.assign(kKdim7 * kHWmid, 0);
    for (uint32_t ci = 0; ci < kC; ++ci)
        for (uint32_t ki = 0; ki < kK7; ++ki)
            for (uint32_t kj = 0; kj < kK7; ++kj) {
                uint32_t row = ci * (kK7 * kK7) + ki * kK7 + kj;
                for (uint32_t ho = 0; ho < kHmid; ++ho) {
                    int ih = static_cast<int>(ho * kStride7 + ki) - static_cast<int>(kPad7);
                    if (ih < 0 || ih >= static_cast<int>(kHin)) continue;
                    for (uint32_t wo = 0; wo < kWmid; ++wo) {
                        int iw = static_cast<int>(wo * kStride7 + kj) - static_cast<int>(kPad7);
                        if (iw < 0 || iw >= static_cast<int>(kWin)) continue;
                        a[row * kHWmid + ho * kWmid + wo] =
                            x_chw[(ci * kHin + ih) * kWin + iw];
                    }
                }
            }
}
void weight_7x7_reshape(const std::vector<uint16_t>& w_cchw,
                        std::vector<uint16_t>& w_mat) {
    w_mat.assign(kC * kKdim7, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ci = 0; ci < kC; ++ci)
            for (uint32_t ki = 0; ki < kK7; ++ki)
                for (uint32_t kj = 0; kj < kK7; ++kj)
                    w_mat[co * kKdim7 + ci * (kK7 * kK7) + ki * kK7 + kj] =
                        w_cchw[((co * kC + ci) * kK7 + ki) * kK7 + kj];
}
void im2col_3x3_s1(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& a) {
    a.assign(kKdim3 * kHWpost, 0);
    for (uint32_t ci = 0; ci < kC; ++ci)
        for (uint32_t ki = 0; ki < kK3; ++ki)
            for (uint32_t kj = 0; kj < kK3; ++kj) {
                uint32_t row = ci * (kK3 * kK3) + ki * kK3 + kj;
                for (uint32_t ho = 0; ho < kHpost; ++ho) {
                    int ih = static_cast<int>(ho + ki) - static_cast<int>(kPad3);
                    if (ih < 0 || ih >= static_cast<int>(kHpost)) continue;
                    for (uint32_t wo = 0; wo < kWpost; ++wo) {
                        int iw = static_cast<int>(wo + kj) - static_cast<int>(kPad3);
                        if (iw < 0 || iw >= static_cast<int>(kWpost)) continue;
                        a[row * kHWpost + ho * kWpost + wo] =
                            x_chw[(ci * kHpost + ih) * kWpost + iw];
                    }
                }
            }
}
void weight_3x3_reshape(const std::vector<uint16_t>& w_cchw,
                        std::vector<uint16_t>& w_mat) {
    w_mat.assign(kC * kKdim3, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ci = 0; ci < kC; ++ci)
            for (uint32_t ki = 0; ki < kK3; ++ki)
                for (uint32_t kj = 0; kj < kK3; ++kj)
                    w_mat[co * kKdim3 + ci * (kK3 * kK3) + ki * kK3 + kj] =
                        w_cchw[((co * kC + ci) * kK3 + ki) * kK3 + kj];
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

// (H, W, C) HWC ↔ (C, H, W) CHW
void hwc_to_chw(const std::vector<uint16_t>& hwc, uint32_t H, uint32_t W,
                std::vector<uint16_t>& chw) {
    chw.assign(kC * H * W, 0);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < kC; ++c)
                chw[(c * H + h) * W + w] = hwc[(h * W + w) * kC + c];
}
void chw_to_hwc(const std::vector<uint16_t>& chw, uint32_t H, uint32_t W,
                std::vector<uint16_t>& hwc) {
    hwc.assign(H * W * kC, 0);
    for (uint32_t c = 0; c < kC; ++c)
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t w = 0; w < W; ++w)
                hwc[(h * W + w) * kC + c] = chw[(c * H + h) * W + w];
}

// Same row-major-block to tile-face helper used by maxpool.
void rm_matrix_to_tiles(const std::vector<uint16_t>& rm, uint32_t rows,
                        std::vector<uint16_t>& out) {
    if (rows % kTileH != 0) throw std::runtime_error("rows % 32 != 0");
    const uint32_t nt = rows / kTileH;
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                block[r * kTileW + c] = rm[(t * kTileH + r) * kTileW + c];
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

// (C, HW_post) tile stream ↔ (C, H_post, W_post) CHW (just an N-tile
// repack — same data, different grouping).
void chw_post_to_tile_stream(const std::vector<uint16_t>& chw, std::vector<uint16_t>& t) {
    tile_matrix(chw, kMt_3, kNt_3, kHWpost, t);
}

}  // namespace

int main() try {
    static_assert(kC == kTileW, "this test assumes C fits in one channel tile");

    const std::string kernel_root = required_env("TT_FOIL_KERNEL_DIR");
    const std::string conv7_dir   = kernel_root + "/conv_7x7";
    const std::string pool_dir    = kernel_root + "/maxpool_3x3";
    const std::string conv_dir    = kernel_root + "/conv";
    const std::string add_dir     = kernel_root + "/residual_add";
    const std::string bias_dir    = kernel_root + "/bias_relu_post";
    const std::string gap_dir     = kernel_root + "/global_avg_pool";
    const std::string fc_dir      = kernel_root + "/fc";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Random inputs / weights / biases --------------------------
    std::mt19937 rng(0xb0a5ed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x(kC * kHin * kWin);
    for (auto& v : x) v = f32_to_bf16(u(rng));

    auto rand_w7 = [&]() {
        std::vector<uint16_t> w(kC * kC * kK7 * kK7);
        for (auto& v : w) v = f32_to_bf16(0.02f * u(rng));
        return w;
    };
    auto rand_w3 = [&]() {
        std::vector<uint16_t> w(kC * kC * kK3 * kK3);
        for (auto& v : w) v = f32_to_bf16(0.05f * u(rng));
        return w;
    };
    auto rand_bias = [&]() {
        std::vector<float> b(kC);
        for (auto& v : b) v = 0.1f * u(rng);
        return b;
    };

    auto W7   = rand_w7();   auto b7   = rand_bias();
    auto W11  = rand_w3();   auto b11  = rand_bias();   // block₁ conv1
    auto W12  = rand_w3();   auto b12  = rand_bias();   // block₁ conv2
    auto W21  = rand_w3();   auto b21  = rand_bias();   // block₂ conv1
    auto W22  = rand_w3();   auto b22  = rand_bias();   // block₂ conv2

    // FC weights and bias (num_classes × C).
    std::vector<uint16_t> Wfc(kNcl * kC);
    for (auto& v : Wfc) v = f32_to_bf16(0.1f * u(rng));
    std::vector<float> bfc(kNcl);
    for (auto& v : bfc) v = 0.1f * u(rng);

    // ---- Reference -------------------------------------------------
    std::vector<uint16_t> y_ref;             // logits (kNcl,)
    std::vector<uint16_t> feature_ref;       // final feature map (kept for debug)
    {
        std::vector<uint16_t> conv_out, pool_hwc, stem_chw;
        conv7x7_s2_bf16(x, W7, conv_out);
        bias_and_relu_chw_h(conv_out, kHmid, kWmid, b7);
        maxpool3x3_chw_to_hwc_ref(conv_out, pool_hwc);
        hwc_to_chw(pool_hwc, kHpost, kWpost, stem_chw);

        // Block 1
        std::vector<uint16_t> t1, t2, y1;
        conv3x3_s1_bf16(stem_chw, W11, t1);
        bias_and_relu_chw_h(t1, kHpost, kWpost, b11);
        conv3x3_s1_bf16(t1, W12, t2);
        bias_only_chw_h(t2, kHpost, kWpost, b12);
        add_bf16(t2, stem_chw, y1);
        relu_inplace(y1);

        // Block 2
        std::vector<uint16_t> t3, t4;
        conv3x3_s1_bf16(y1, W21, t3);
        bias_and_relu_chw_h(t3, kHpost, kWpost, b21);
        conv3x3_s1_bf16(t3, W22, t4);
        bias_only_chw_h(t4, kHpost, kWpost, b22);
        std::vector<uint16_t> feat;
        add_bf16(t4, y1, feat);
        relu_inplace(feat);
        feature_ref = feat;

        // Classifier tail
        std::vector<float> gap(kC, 0.0f);
        for (uint32_t c = 0; c < kC; ++c) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < kHpost * kWpost; ++i)
                acc += bf16_to_f32(feat[c * (kHpost * kWpost) + i]);
            gap[c] = acc / static_cast<float>(kHpost * kWpost);
        }
        std::vector<uint16_t> gap_bf16(kC);
        for (uint32_t c = 0; c < kC; ++c) gap_bf16[c] = f32_to_bf16(gap[c]);

        y_ref.assign(kNcl, 0);
        for (uint32_t k = 0; k < kNcl; ++k) {
            float acc = 0.0f;
            for (uint32_t c = 0; c < kC; ++c)
                acc += bf16_to_f32(Wfc[k * kC + c]) * bf16_to_f32(gap_bf16[c]);
            y_ref[k] = f32_to_bf16(acc + bfc[k]);
        }
    }

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // Conv stage DRAM scratch. Two K-sizes (7×7 and 3×3) get separate
    // weight/im2col buffers to keep extents explicit and avoid having
    // to track which size last lived there.
    const uint32_t w7_bytes   = kMt_7 * kKt_7 * kTileBytes;
    const uint32_t a7_bytes   = kKt_7 * kNt_7 * kTileBytes;
    const uint32_t y7_bytes   = kMt_7 * kNt_7 * kTileBytes;   // 8 tiles
    const uint32_t w3_bytes   = kMt_3 * kKt_3 * kTileBytes;
    const uint32_t a3_bytes   = kKt_3 * kNt_3 * kTileBytes;
    const uint32_t y3_bytes   = kMt_3 * kNt_3 * kTileBytes;   // 2 tiles
    const uint32_t add_bytes  = kRaNt  * kTileBytes;          // 2 tiles
    static_assert(y3_bytes == add_bytes, "matmul out and add stream sizes match");

    auto buf_W7      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w7_bytes,   core);
    auto buf_A7      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a7_bytes,   core);
    auto buf_Y7      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y7_bytes,   core);
    auto buf_W3      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w3_bytes,   core);
    auto buf_A3      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a3_bytes,   core);
    auto buf_Ym      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y3_bytes,   core);  // conv₂ output
    auto buf_Ys      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y3_bytes,   core);  // skip-path side
    auto buf_Yo      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y3_bytes,   core);  // block output
    // bias_relu_post output (sized for stem = 8 tiles, blocks use only 2):
    auto buf_post    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y7_bytes,   core);
    // bias tile (1 tile reused across every bias_relu_post call):
    auto buf_bias_d  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    // GAP scaler tile (1/HW filled, written once before the tail phase):
    auto buf_scaler  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    // Maxpool stage (9 input streams + 1 output, all L1)
    const uint32_t stream_bytes = kNtPool * kTileBytes;
    std::array<std::shared_ptr<tt::foil::Buffer>, 9> buf_pool_in;
    for (uint32_t s = 0; s < 9; ++s)
        buf_pool_in[s] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    auto buf_pool_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    std::array<std::shared_ptr<tt::foil::Buffer>, 10> pool_cb_bufs;
    for (uint32_t i = 0; i < 10; ++i)
        pool_cb_bufs[i] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    auto noc_of = [&](auto& buf) { return tt::foil::make_noc_dram_addr(*dev, buf->device_addr); };
    uint64_t W7_noc      = noc_of(buf_W7);
    uint64_t A7_noc      = noc_of(buf_A7);
    uint64_t Y7_noc      = noc_of(buf_Y7);
    uint64_t W3_noc      = noc_of(buf_W3);
    uint64_t A3_noc      = noc_of(buf_A3);
    uint64_t Ym_noc      = noc_of(buf_Ym);
    uint64_t Ys_noc      = noc_of(buf_Ys);
    uint64_t Yo_noc      = noc_of(buf_Yo);
    uint64_t post_noc    = noc_of(buf_post);
    uint64_t bias_d_noc  = noc_of(buf_bias_d);
    uint64_t scaler_noc  = noc_of(buf_scaler);

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

    std::array<tt::foil::CbConfig, 3> matmul_cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    std::array<tt::foil::CbConfig, 10> pool_cbs = {{
        {0,  pool_cb_bufs[0]->device_addr, kTileBytes, 1, kTileBytes},
        {1,  pool_cb_bufs[1]->device_addr, kTileBytes, 1, kTileBytes},
        {2,  pool_cb_bufs[2]->device_addr, kTileBytes, 1, kTileBytes},
        {3,  pool_cb_bufs[3]->device_addr, kTileBytes, 1, kTileBytes},
        {4,  pool_cb_bufs[4]->device_addr, kTileBytes, 1, kTileBytes},
        {5,  pool_cb_bufs[5]->device_addr, kTileBytes, 1, kTileBytes},
        {6,  pool_cb_bufs[6]->device_addr, kTileBytes, 1, kTileBytes},
        {7,  pool_cb_bufs[7]->device_addr, kTileBytes, 1, kTileBytes},
        {8,  pool_cb_bufs[8]->device_addr, kTileBytes, 1, kTileBytes},
        {16, pool_cb_bufs[9]->device_addr, kTileBytes, 1, kTileBytes},
    }};

    // ---- Helper: pack a per-channel bias vector into one (32, 32) tile,
    //              with channel biases in column 0 and zeros elsewhere.
    auto pack_bias_tile = [&](const std::vector<float>& bias,
                              std::vector<uint16_t>& tiles_out) {
        std::vector<uint16_t> rm(kTileH * kTileW, 0);
        for (uint32_t r = 0; r < kC; ++r) rm[r * kTileW + 0] = f32_to_bf16(bias[r]);
        tiles_out.clear();
        tt::foil::test::row_major_to_tile(rm.data(), tiles_out);
    };
    // Zero-bias tile, used when bias_relu_post is asked to apply ReLU only
    // (e.g. after a residual add).
    auto zero_bias = std::vector<float>(kC, 0.0f);

    // Helper: write a freshly-packed bias tile into buf_bias_d. The reader
    // re-reads the same DRAM address every tile so this only runs once
    // per bias_relu_post call.
    auto stage_bias = [&](const std::vector<float>& bias) {
        std::vector<uint16_t> bt;
        pack_bias_tile(bias, bt);
        tt::foil::write_buffer(*dev, *buf_bias_d, bt.data(), kTileBytes);
    };

    // Helper: run bias_relu_post on `n_tiles` tiles already laid out in
    // `buf_in_dram` (NOC addr `in_noc`), reading bias from buf_bias_d,
    // writing to `buf_out_dram` (NOC addr `out_noc`).
    auto run_bias_relu = [&](tt::foil::Kernel& k,
                             tt::foil::Buffer& buf_in_dram, uint64_t in_noc,
                             tt::foil::Buffer& buf_out_dram, uint64_t out_noc,
                             uint32_t n_tiles, uint32_t relu_enable) {
        const uint32_t bytes = n_tiles * kTileBytes;
        std::vector<uint8_t> zero(bytes, 0);
        tt::foil::write_buffer(*dev, buf_out_dram, zero.data(), bytes);

        std::array<uint32_t, 5> rab = {
            lo(in_noc),     hi(in_noc),
            lo(bias_d_noc), hi(bias_d_noc),
            n_tiles,
        };
        std::array<uint32_t, 3> ran = { lo(out_noc), hi(out_noc), n_tiles };
        std::array<uint32_t, 2> rac = { n_tiles, relu_enable };
        tt::foil::set_runtime_args(*dev, k, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::NCRISC, ran);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC0, rac);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC1, rac);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC2, rac);
        tt::foil::register_cbs(*dev, k, matmul_cbs);
        tt::foil::execute(*dev, k);
    };

    // Helper: pack (C, HW) row-major chw → tile stream, write to DRAM
    // buffer, then run bias+ReLU on device, read back tiles, untile
    // back to (C, HW) chw. Bias staged via stage_bias() before calling.
    auto bias_relu_chw = [&](tt::foil::Kernel& k,
                             std::vector<uint16_t>& chw_inout,
                             uint32_t H, uint32_t W,
                             const std::vector<float>& bias,
                             uint32_t relu_enable,
                             tt::foil::Buffer& buf_io_dram,
                             uint64_t io_noc) {
        const uint32_t hw = H * W;
        const uint32_t nt = hw / kTileW;
        const uint32_t bytes = nt * kTileBytes;
        std::vector<uint16_t> in_tiles;
        tile_matrix(chw_inout, kMt_3 /*=1*/, nt, hw, in_tiles);
        tt::foil::write_buffer(*dev, buf_io_dram, in_tiles.data(), bytes);

        stage_bias(bias);
        run_bias_relu(k, buf_io_dram, io_noc, *buf_post, post_noc, nt, relu_enable);

        std::vector<uint16_t> out_tiles(nt * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, out_tiles.data(), bytes);
        untile_matrix(out_tiles, 1, nt, hw, chw_inout);
    };

    // Kernel handles: rebound per phase via release_kernels(). The
    // shared_ptr is captured by reference by the lambdas below, so they
    // resolve to whichever program is currently loaded.
    std::shared_ptr<tt::foil::Kernel> k_conv7, k_pool, k_conv, k_add,
                                       k_bias, k_gap, k_fc;

    // ================ Phase 1 — STEM ================================
    // Kernel slots: conv_7x7 + maxpool_3x3 + bias_relu_post.
    k_conv7 = load(conv7_dir);
    k_pool  = load(pool_dir);
    k_bias  = load(bias_dir);

    std::vector<uint16_t> stem_chw;
    {
        // ---- Conv₇ₓ₇ → buf_Y7 (8 tiles, C × HW_mid) ----------------
        std::vector<uint16_t> w_mat, w_tiles, a_mat, a_tiles;
        weight_7x7_reshape(W7, w_mat);
        tile_matrix(w_mat, kMt_7, kKt_7, kKdim7, w_tiles);
        im2col_7x7(x, a_mat);
        tile_matrix(a_mat, kKt_7, kNt_7, kHWmid, a_tiles);

        tt::foil::write_buffer(*dev, *buf_W7, w_tiles.data(), w7_bytes);
        tt::foil::write_buffer(*dev, *buf_A7, a_tiles.data(), a7_bytes);
        std::vector<uint8_t> zero(y7_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_Y7, zero.data(), y7_bytes);

        std::array<uint32_t, 7> ra = {
            lo(W7_noc), hi(W7_noc), lo(A7_noc), hi(A7_noc),
            kMt_7, kKt_7, kNt_7,
        };
        std::array<uint32_t, 3> rn = { lo(Y7_noc), hi(Y7_noc), kMt_7 * kNt_7 };
        tt::foil::set_runtime_args(*dev, *k_conv7, R::RiscId::BRISC,  ra);
        tt::foil::set_runtime_args(*dev, *k_conv7, R::RiscId::NCRISC, rn);
        tt::foil::register_cbs(*dev, *k_conv7, matmul_cbs);
        tt::foil::execute(*dev, *k_conv7);

        // ---- bias + ReLU on device (buf_Y7 → buf_post) --------------
        stage_bias(b7);
        run_bias_relu(*k_bias, *buf_Y7, Y7_noc, *buf_post, post_noc,
                      /*n_tiles=*/kMt_7 * kNt_7, /*relu_enable=*/1);

        // Read the post-op tiles back and untile to (C, HW_mid).
        std::vector<uint16_t> y_tiles(kMt_7 * kNt_7 * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, y_tiles.data(), y7_bytes);
        std::vector<uint16_t> conv_chw;
        untile_matrix(y_tiles, kMt_7, kNt_7, kHWmid, conv_chw);   // (C, HW_mid)

        // ---- Stem stage 2: Maxpool₃ₓ₃ (HWC layout in/out) ----------
        std::vector<uint16_t> conv_hwc;
        chw_to_hwc(conv_chw, kHmid, kWmid, conv_hwc);

        const uint16_t kNegInf = f32_to_bf16(-1.0e30f);
        auto idx = [&](uint32_t h, uint32_t w, uint32_t c) {
            return (h * kWmid + w) * kC + c;
        };
        std::array<std::vector<uint16_t>, 9> streams_rm;
        for (auto& s : streams_rm) s.assign(kHpost * kWpost * kC, kNegInf);
        for (uint32_t s = 0; s < 9; ++s) {
            int di = static_cast<int>(s / 3);
            int dj = static_cast<int>(s % 3);
            for (uint32_t oh = 0; oh < kHpost; ++oh) {
                int ih = static_cast<int>(oh) * static_cast<int>(kStrideP) + di -
                         static_cast<int>(kPadP);
                if (ih < 0 || ih >= static_cast<int>(kHmid)) continue;
                for (uint32_t ow = 0; ow < kWpost; ++ow) {
                    int iw = static_cast<int>(ow) * static_cast<int>(kStrideP) + dj -
                             static_cast<int>(kPadP);
                    if (iw < 0 || iw >= static_cast<int>(kWmid)) continue;
                    for (uint32_t c = 0; c < kC; ++c)
                        streams_rm[s][(oh * kWpost + ow) * kC + c] =
                            conv_hwc[idx(static_cast<uint32_t>(ih),
                                         static_cast<uint32_t>(iw), c)];
                }
            }
        }
        std::array<std::vector<uint16_t>, 9> streams_tiles;
        for (uint32_t s = 0; s < 9; ++s)
            rm_matrix_to_tiles(streams_rm[s], kHpost * kWpost, streams_tiles[s]);
        for (uint32_t s = 0; s < 9; ++s)
            tt::foil::write_buffer(*dev, *buf_pool_in[s],
                                   streams_tiles[s].data(), stream_bytes);
        std::vector<uint16_t> zerop(stream_bytes / 2, 0);
        tt::foil::write_buffer(*dev, *buf_pool_out, zerop.data(), stream_bytes);

        std::array<uint32_t, 10> rab = {
            static_cast<uint32_t>(buf_pool_in[0]->device_addr),
            static_cast<uint32_t>(buf_pool_in[1]->device_addr),
            static_cast<uint32_t>(buf_pool_in[2]->device_addr),
            static_cast<uint32_t>(buf_pool_in[3]->device_addr),
            static_cast<uint32_t>(buf_pool_in[4]->device_addr),
            static_cast<uint32_t>(buf_pool_in[5]->device_addr),
            static_cast<uint32_t>(buf_pool_in[6]->device_addr),
            static_cast<uint32_t>(buf_pool_in[7]->device_addr),
            static_cast<uint32_t>(buf_pool_in[8]->device_addr),
            kNtPool,
        };
        std::array<uint32_t, 2> ran = {
            static_cast<uint32_t>(buf_pool_out->device_addr), kNtPool };
        std::array<uint32_t, 1> rac = { kNtPool };
        tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::NCRISC, ran);
        tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC0, rac);
        tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC1, rac);
        tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC2, rac);
        tt::foil::register_cbs(*dev, *k_pool, pool_cbs);
        tt::foil::execute(*dev, *k_pool);

        std::vector<uint16_t> pool_tiles(stream_bytes / 2, 0);
        tt::foil::read_buffer(*dev, *buf_pool_out, pool_tiles.data(), stream_bytes);
        std::vector<uint16_t> pool_hwc(kHpost * kWpost * kC);
        std::vector<uint16_t> block(kTileH * kTileW);
        for (uint32_t t = 0; t < kNtPool; ++t) {
            tt::foil::test::tile_to_row_major(pool_tiles.data() + t * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    pool_hwc[(t * kTileH + r) * kTileW + c] = block[r * kTileW + c];
        }
        hwc_to_chw(pool_hwc, kHpost, kWpost, stem_chw);
    }

    // ---- Phase 1 → Phase 2 transition: free the stem kernels and reload
    // the block kernels into the freed KERNEL_CONFIG slot.
    k_conv7.reset();
    k_pool.reset();
    k_bias.reset();
    tt::foil::release_kernels(*dev, core);
    k_conv = load(conv_dir);
    k_add  = load(add_dir);
    k_bias = load(bias_dir);

    // Helpers for the block-stage matmul (Mt=1 Kt=9 Nt=2) and residual_add.
    auto run_conv3 = [&](const std::vector<uint16_t>& x_chw,
                         const std::vector<uint16_t>& w_cchw,
                         tt::foil::Buffer& buf_Y, uint64_t Y_noc,
                         std::vector<uint16_t>& y_chw_out) {
        std::vector<uint16_t> w_mat, w_tiles, a_mat, a_tiles;
        weight_3x3_reshape(w_cchw, w_mat);
        tile_matrix(w_mat, kMt_3, kKt_3, kKdim3, w_tiles);
        im2col_3x3_s1(x_chw, a_mat);
        tile_matrix(a_mat, kKt_3, kNt_3, kHWpost, a_tiles);

        tt::foil::write_buffer(*dev, *buf_W3, w_tiles.data(), w3_bytes);
        tt::foil::write_buffer(*dev, *buf_A3, a_tiles.data(), a3_bytes);
        std::vector<uint8_t> zero(y3_bytes, 0);
        tt::foil::write_buffer(*dev, buf_Y, zero.data(), y3_bytes);

        std::array<uint32_t, 7> ra = {
            lo(W3_noc), hi(W3_noc), lo(A3_noc), hi(A3_noc),
            kMt_3, kKt_3, kNt_3,
        };
        std::array<uint32_t, 3> rn = { lo(Y_noc), hi(Y_noc), kMt_3 * kNt_3 };
        tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::BRISC,  ra);
        tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::NCRISC, rn);
        tt::foil::register_cbs(*dev, *k_conv, matmul_cbs);
        tt::foil::execute(*dev, *k_conv);

        std::vector<uint16_t> y_tiles(kMt_3 * kNt_3 * kTileWords, 0);
        tt::foil::read_buffer(*dev, buf_Y, y_tiles.data(), y3_bytes);
        untile_matrix(y_tiles, kMt_3, kNt_3, kHWpost, y_chw_out);
    };

    auto run_add = [&](tt::foil::Buffer& buf_a_dram, tt::foil::Buffer& buf_b_dram,
                       tt::foil::Buffer& buf_out_dram,
                       uint64_t a_noc, uint64_t b_noc, uint64_t out_noc,
                       const std::vector<uint16_t>& a_tiles,
                       const std::vector<uint16_t>& b_tiles,
                       std::vector<uint16_t>& y_chw_out) {
        tt::foil::write_buffer(*dev, buf_a_dram, a_tiles.data(), add_bytes);
        tt::foil::write_buffer(*dev, buf_b_dram, b_tiles.data(), add_bytes);
        std::vector<uint8_t> zero(add_bytes, 0);
        tt::foil::write_buffer(*dev, buf_out_dram, zero.data(), add_bytes);
        std::array<uint32_t, 4> ra = {
            lo(a_noc), hi(a_noc), lo(b_noc), hi(b_noc),
        };
        std::array<uint32_t, 2> rn = { lo(out_noc), hi(out_noc) };
        tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::BRISC,  ra);
        tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::NCRISC, rn);
        tt::foil::register_cbs(*dev, *k_add, matmul_cbs);
        tt::foil::execute(*dev, *k_add);
        std::vector<uint16_t> y_tiles(kRaNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, buf_out_dram, y_tiles.data(), add_bytes);
        untile_matrix(y_tiles, kMt_3, kNt_3, kHWpost, y_chw_out);
    };

    // ================ Block 1 — basic, skip = stem_chw ==============
    std::vector<uint16_t> y1_chw;
    {
        std::vector<uint16_t> t1, t2;
        run_conv3(stem_chw, W11, *buf_Ym, Ym_noc, t1);
        bias_relu_chw(*k_bias, t1, kHpost, kWpost, b11, /*relu=*/1, *buf_Ym, Ym_noc);
        run_conv3(t1, W12, *buf_Ym, Ym_noc, t2);
        bias_relu_chw(*k_bias, t2, kHpost, kWpost, b12, /*relu=*/0, *buf_Ym, Ym_noc);

        std::vector<uint16_t> t2_tiles, sk_tiles;
        chw_post_to_tile_stream(t2,       t2_tiles);
        chw_post_to_tile_stream(stem_chw, sk_tiles);
        run_add(*buf_Ym, *buf_Ys, *buf_Yo, Ym_noc, Ys_noc, Yo_noc,
                t2_tiles, sk_tiles, y1_chw);
        // ReLU after residual: bias_relu_post with bias = 0.
        bias_relu_chw(*k_bias, y1_chw, kHpost, kWpost, zero_bias, /*relu=*/1, *buf_Ym, Ym_noc);
    }

    // ================ Block 2 — basic, skip = y1_chw ================
    std::vector<uint16_t> feat_dev;
    {
        std::vector<uint16_t> t3, t4;
        run_conv3(y1_chw, W21, *buf_Ym, Ym_noc, t3);
        bias_relu_chw(*k_bias, t3, kHpost, kWpost, b21, /*relu=*/1, *buf_Ym, Ym_noc);
        run_conv3(t3, W22, *buf_Ym, Ym_noc, t4);
        bias_relu_chw(*k_bias, t4, kHpost, kWpost, b22, /*relu=*/0, *buf_Ym, Ym_noc);

        std::vector<uint16_t> t4_tiles, y1_tiles;
        chw_post_to_tile_stream(t4,     t4_tiles);
        chw_post_to_tile_stream(y1_chw, y1_tiles);
        run_add(*buf_Ym, *buf_Ys, *buf_Yo, Ym_noc, Ys_noc, Yo_noc,
                t4_tiles, y1_tiles, feat_dev);
        bias_relu_chw(*k_bias, feat_dev, kHpost, kWpost, zero_bias, /*relu=*/1, *buf_Ym, Ym_noc);
    }

    // ---- Phase 2 → Phase 3 transition: free block kernels, load tail. -
    k_conv.reset();
    k_add.reset();
    k_bias.reset();
    tt::foil::release_kernels(*dev, core);
    k_gap  = load(gap_dir);
    k_fc   = load(fc_dir);
    k_bias = load(bias_dir);

    // ================ Classifier tail: GAP + FC + bias (all device) =
    std::vector<uint16_t> y_dev(kNcl);
    {
        // ---- Stage A: device global_avg_pool -----------------------
        // Pre-fill the scaler tile (1/HW) once for this stage.
        std::vector<uint16_t> scaler_rm(kTileH * kTileW,
                                        f32_to_bf16(1.0f / static_cast<float>(kHpost * kWpost)));
        std::vector<uint16_t> scaler_tiles;
        tt::foil::test::row_major_to_tile(scaler_rm.data(), scaler_tiles);
        tt::foil::write_buffer(*dev, *buf_scaler, scaler_tiles.data(), kTileBytes);

        // Tile feat_dev (C, HW_post) and write to buf_Ym (2 tiles).
        std::vector<uint16_t> feat_tiles;
        chw_post_to_tile_stream(feat_dev, feat_tiles);
        const uint32_t feat_bytes = kRaNt * kTileBytes;
        tt::foil::write_buffer(*dev, *buf_Ym, feat_tiles.data(), feat_bytes);

        // GAP output (1 tile, col 0 = per-channel mean) → buf_Ys.
        std::vector<uint8_t> zero(kTileBytes, 0);
        tt::foil::write_buffer(*dev, *buf_Ys, zero.data(), kTileBytes);

        std::array<uint32_t, 5> ga_rab = {
            lo(Ym_noc),     hi(Ym_noc),
            lo(scaler_noc), hi(scaler_noc),
            kRaNt,
        };
        std::array<uint32_t, 2> ga_ran = { lo(Ys_noc), hi(Ys_noc) };
        std::array<uint32_t, 1> ga_rac = { kRaNt };
        tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::BRISC,  ga_rab);
        tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::NCRISC, ga_ran);
        tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC0, ga_rac);
        tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC1, ga_rac);
        tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC2, ga_rac);
        tt::foil::register_cbs(*dev, *k_gap, matmul_cbs);
        tt::foil::execute(*dev, *k_gap);

        // ---- Stage B: device FC matmul -----------------------------
        // W: (Ncl, C) tiled. X comes straight from GAP output (already
        // in buf_Ys at the FC's expected layout — channel value in col 0
        // of a 32×32 tile). Output → buf_Yo (1 tile).
        std::vector<uint16_t> Wmat(kNcl * kC);
        for (uint32_t r = 0; r < kNcl; ++r)
            for (uint32_t c = 0; c < kC; ++c)
                Wmat[r * kC + c] = Wfc[r * kC + c];
        std::vector<uint16_t> w_tiles;
        tile_matrix(Wmat, kMt_fc, kKt_fc, kC, w_tiles);

        tt::foil::write_buffer(*dev, *buf_Ym, w_tiles.data(), kTileBytes);
        tt::foil::write_buffer(*dev, *buf_Yo, zero.data(), kTileBytes);

        std::array<uint32_t, 7> fc_ra = {
            lo(Ym_noc), hi(Ym_noc), lo(Ys_noc), hi(Ys_noc),
            kMt_fc, kKt_fc, kNt_fc,
        };
        std::array<uint32_t, 3> fc_rn = { lo(Yo_noc), hi(Yo_noc), kMt_fc * kNt_fc };
        tt::foil::set_runtime_args(*dev, *k_fc, R::RiscId::BRISC,  fc_ra);
        tt::foil::set_runtime_args(*dev, *k_fc, R::RiscId::NCRISC, fc_rn);
        tt::foil::register_cbs(*dev, *k_fc, matmul_cbs);
        tt::foil::execute(*dev, *k_fc);

        // ---- Stage C: device FC bias add ---------------------------
        stage_bias(bfc);
        run_bias_relu(*k_bias, *buf_Yo, Yo_noc, *buf_post, post_noc,
                      /*n_tiles=*/1, /*relu_enable=*/0);

        // Read logits: col 0 of the single output tile.
        std::vector<uint16_t> y_tiles(kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, y_tiles.data(), kTileBytes);
        std::vector<uint16_t> y_block(kTileH * kTileW);
        tt::foil::test::tile_to_row_major(y_tiles.data(), y_block.data());
        for (uint32_t k = 0; k < kNcl; ++k)
            y_dev[k] = y_block[k * kTileW + 0];
    }

    // ---- Compare ---------------------------------------------------
    // The chain stacks ~12 bf16 rounding stages (feature path + GAP +
    // FC + bias). Logit magnitudes are small (the FC sees gap values
    // bounded by the bf16 of channel means), so allow a generous abs
    // floor.
    const float kAbsTol = 0.10f;
    const float kRelTol = 0.05f;
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
            "test_resnet_classifier: %u/%zu mismatches; first at class %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, y_dev.size(), first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(y_ref[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_resnet_classifier: FAIL");
        return 1;
    }

    // argmax for a "real classifier" feel.
    uint32_t argmax_dev = 0, argmax_ref = 0;
    float    max_dev = bf16_to_f32(y_dev[0]), max_ref = bf16_to_f32(y_ref[0]);
    for (uint32_t k = 1; k < kNcl; ++k) {
        float vd = bf16_to_f32(y_dev[k]);
        float vr = bf16_to_f32(y_ref[k]);
        if (vd > max_dev) { max_dev = vd; argmax_dev = k; }
        if (vr > max_ref) { max_ref = vr; argmax_ref = k; }
    }
    const char* agree = (argmax_dev == argmax_ref) ? "✓" : "✗";

    std::printf("test_resnet_classifier: PASS  "
                "(image %ux%u→stem→2× basic→GAP→FC %u-way; "
                "argmax dev=%u ref=%u %s, worst abs=%.5f, worst rel=%.4f%%)\n",
                kHin, kWin, kNcl, argmax_dev, argmax_ref, agree,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_resnet_classifier: FAIL — %s\n", e.what());
    return 1;
}
