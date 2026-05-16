// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v7-2: bf16 3×3 convolution with padding=1, stride=1, N=1.
//
// Implementation strategy is the classic im2col → matmul:
//   1. Host generates X (C_in, H, W) and W (C_out, C_in, 3, 3).
//   2. Host computes a reference conv directly (the ground truth).
//   3. Host im2col's X into a (C_in*9, H*W) matrix; reshapes W into
//      (C_out, C_in*9).
//   4. Device runs Y = W @ X_im2col via the matmul_dram kernel.
//   5. Host reshapes Y (C_out, H*W) back to (C_out, H, W) and compares.
//
// Tile-count requirements:
//   M = C_out          must be a multiple of 32
//   K = C_in * 9       (C_in % 32 == 0 ⇒ K % 32 == 0 since 9 is odd
//                       and 32 = 2^5)
//   N = H * W          must be a multiple of 32

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"
#include "tile_utils.hpp"

#ifndef MM_MT
#define MM_MT 1
#endif
#ifndef MM_KT
#define MM_KT 9
#endif
#ifndef MM_NT
#define MM_NT 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kMt   = MM_MT;
constexpr uint32_t kKt   = MM_KT;
constexpr uint32_t kNt   = MM_NT;

constexpr uint32_t kCout = kMt * kTileH;             // M
constexpr uint32_t kKdim = kKt * kTileW;             // K = C_in * 9
constexpr uint32_t kHW   = kNt * kTileW;             // N = H * W

constexpr uint32_t kKH = 3;
constexpr uint32_t kKW = 3;
constexpr uint32_t kPad = 1;
constexpr uint32_t kCin  = kKdim / (kKH * kKW);      // recover C_in
static_assert(kCin * kKH * kKW == kKdim,
              "MM_KT*32 must be divisible by 9 (i.e. C_in*9 == MM_KT*32)");

constexpr uint32_t kH = 8;
constexpr uint32_t kW = kHW / kH;
static_assert(kH * kW == kHW, "H must divide MM_NT*32");

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Reference 3×3 conv (pad=1, stride=1): direct CHW computation.
void conv3x3_reference(const std::vector<uint16_t>& x_chw,
                       const std::vector<uint16_t>& w_cchw,
                       std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * kH * kW, 0);
    for (uint32_t co = 0; co < kCout; ++co) {
        for (uint32_t h = 0; h < kH; ++h) {
            for (uint32_t w_ = 0; w_ < kW; ++w_) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kCin; ++ci) {
                    for (uint32_t ki = 0; ki < kKH; ++ki) {
                        for (uint32_t kj = 0; kj < kKW; ++kj) {
                            int ih = static_cast<int>(h + ki) - static_cast<int>(kPad);
                            int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                            if (ih < 0 || ih >= static_cast<int>(kH) ||
                                iw < 0 || iw >= static_cast<int>(kW)) continue;
                            float xv = bf16_to_f32(x_chw[(ci * kH + ih) * kW + iw]);
                            float wv = bf16_to_f32(
                                w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj]);
                            acc += xv * wv;
                        }
                    }
                }
                y_chw[(co * kH + h) * kW + w_] = f32_to_bf16(acc);
            }
        }
    }
}

// im2col: X (C_in, H, W) → A (C_in*9, H*W), row-major.
// Row r = ci * 9 + ki * 3 + kj; column c = h * W + w_.
void im2col(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& a) {
    a.assign(kKdim * kHW, 0);   // zero-init handles padding
    for (uint32_t ci = 0; ci < kCin; ++ci) {
        for (uint32_t ki = 0; ki < kKH; ++ki) {
            for (uint32_t kj = 0; kj < kKW; ++kj) {
                uint32_t row = ci * (kKH * kKW) + ki * kKW + kj;
                for (uint32_t h = 0; h < kH; ++h) {
                    int ih = static_cast<int>(h + ki) - static_cast<int>(kPad);
                    if (ih < 0 || ih >= static_cast<int>(kH)) continue;
                    for (uint32_t w_ = 0; w_ < kW; ++w_) {
                        int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                        if (iw < 0 || iw >= static_cast<int>(kW)) continue;
                        uint32_t col = h * kW + w_;
                        a[row * kHW + col] = x_chw[(ci * kH + ih) * kW + iw];
                    }
                }
            }
        }
    }
}

// Weight reshape: W (C_out, C_in, 3, 3) → (C_out, C_in*9), row-major.
void weight_reshape(const std::vector<uint16_t>& w_cchw,
                    std::vector<uint16_t>& w_mat) {
    w_mat.assign(kCout * kKdim, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj) {
                    uint32_t col = ci * (kKH * kKW) + ki * kKW + kj;
                    w_mat[co * kKdim + col] =
                        w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj];
                }
}

// Tile (rows_t × cols_t) row-major matrix → tile stream (rt, ct) row-major.
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

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t w_bytes   = kMt * kKt * kTileBytes;
    const uint32_t a_bytes   = kKt * kNt * kTileBytes;
    const uint32_t y_bytes   = kMt * kNt * kTileBytes;

    // ---- Host tensors -----------------------------------------------
    std::vector<uint16_t> x_chw(kCin * kH * kW);
    std::vector<uint16_t> w_cchw(kCout * kCin * kKH * kKW);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t h = 0; h < kH; ++h)
            for (uint32_t w_ = 0; w_ < kW; ++w_)
                x_chw[(ci * kH + h) * kW + w_] = f32_to_bf16(0.005f *
                    static_cast<float>((ci + 1) + ((h * kW + w_) % 7)));

    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj)
                    w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj] =
                        f32_to_bf16(0.005f *
                            static_cast<float>((co + 1) + (ci % 5) + (ki * 3 + kj)));

    // ---- Reference --------------------------------------------------
    std::vector<uint16_t> y_ref_chw;
    conv3x3_reference(x_chw, w_cchw, y_ref_chw);

    // ---- im2col + weight reshape -----------------------------------
    std::vector<uint16_t> a_mat;        // (C_in*9, H*W)
    std::vector<uint16_t> w_mat;        // (C_out, C_in*9)
    im2col(x_chw, a_mat);
    weight_reshape(w_cchw, w_mat);

    // Matmul: Y = W @ A, where:
    //   M = C_out   → W is (M, K)
    //   K = C_in*9  → A is (K, N)
    //   N = H*W     → Y is (M, N)
    // The matmul kernel takes the first input as the "A" CB and the
    // second as "B", so we feed W as A and im2col as B.
    std::vector<uint16_t> w_tiles, a_tiles;
    tile_matrix(w_mat, /*rows_t=*/kMt, /*cols_t=*/kKt, /*col_dim=*/kKdim, w_tiles);
    tile_matrix(a_mat, /*rows_t=*/kKt, /*cols_t=*/kNt, /*col_dim=*/kHW,   a_tiles);

    // ---- Device -----------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes, core);
    auto buf_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes, core);
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, w_tiles.data(), w_bytes);
    tt::foil::write_buffer(*dev, *buf_b, a_tiles.data(), a_bytes);
    std::vector<uint8_t> zero(y_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), y_bytes);

    uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, buf_a  ->device_addr);
    uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, buf_b  ->device_addr);
    uint64_t out_noc = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    std::array<uint32_t, 7> ra_brisc = {
        lo(a_noc), hi(a_noc),
        lo(b_noc), hi(b_noc),
        kMt, kKt, kNt,
    };
    std::array<uint32_t, 3> ra_ncrisc = {
        lo(out_noc), hi(out_noc), kMt * kNt,
    };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), y_bytes);

    std::vector<uint16_t> y_mat;     // (C_out, H*W)
    untile_matrix(y_tiles, kMt, kNt, kHW, y_mat);

    // (C_out, H*W) → CHW is an identity reinterpretation.
    const auto& y_dev_chw = y_mat;

    // ---- Compare ----------------------------------------------------
    const float kAbsTol = 0.05f;
    const float kRelTol = 0.01f;
    uint32_t bad = 0;
    uint32_t first_bad = static_cast<uint32_t>(y_dev_chw.size());
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (uint32_t i = 0; i < y_dev_chw.size(); ++i) {
        float got = bf16_to_f32(y_dev_chw[i]);
        float exp = bf16_to_f32(y_ref_chw[i]);
        float d   = std::fabs(got - exp);
        float ref = std::fabs(exp);
        float tol = std::max(kAbsTol, kRelTol * ref);
        if (d > worst_abs) worst_abs = d;
        if (ref > 0.f && (d / ref) > worst_rel) worst_rel = d / ref;
        if (d > tol) {
            if (first_bad == y_dev_chw.size()) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_conv_3x3(C_in=%u C_out=%u H=%u W=%u): %u/%zu mismatches; "
            "first at idx %u: got=%.5f expected=%.5f, worst abs=%.5f, "
            "worst rel=%.4f%%\n",
            kCin, kCout, kH, kW, bad, y_dev_chw.size(), first_bad,
            bf16_to_f32(y_dev_chw[first_bad]),
            bf16_to_f32(y_ref_chw[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_conv_3x3: FAIL");
        return 1;
    }

    std::printf("test_conv_3x3: PASS  (C_in=%u C_out=%u H=%u W=%u, pad=1 stride=1, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kCin, kCout, kH, kW, worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_conv_3x3: FAIL — %s\n", e.what());
    return 1;
}
