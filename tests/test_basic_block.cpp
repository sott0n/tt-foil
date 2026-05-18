// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ResNet basic block, host-orchestrated on top of tt-foil primitives.
//
//   y = ReLU( x + Conv₂( ReLU( Conv₁(x; W₁') + b₁' ); W₂') + b₂' )
//
// where (W_i', b_i') = BatchNorm folded into the i-th conv's weights and
// bias.  Both convs are 3×3, pad=1, stride=1 (no downsample), so the
// spatial shape and channel count are preserved end-to-end and the skip
// connection is identity.
//
// Device work:
//   1. Conv₁  — examples/conv_3x3 kernel (matmul over im2col).
//   2. Conv₂  — same kernel, different weights / input.
//   3. Skip add  — examples/residual_add kernel (RA_NT=2 build under
//      examples/basic_block/prebuilt/).
//
// Host work in between each device call:
//   - im2col of the running activation,
//   - addition of the BN-folded bias broadcast over (H, W),
//   - ReLU after Conv₁,
//   - ReLU after the residual add.
//
// The reference path computes the same composition in pure fp32 →
// bf16 (one round per device-equivalent stage) and compares element-
// wise. Tolerance is wider than the single-conv test because four
// bf16 rounding stages stack up.
//
// Two kernel dirs are needed because the residual_add kernels must be
// compiled with RA_NT matching this test's tile count, while the conv
// example's prebuilt is shaped for Mt=1, Kt=9, Nt=2. See
// CMakeLists.txt for how both are passed via env.

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

// Shape — chosen to match the existing conv_3x3 example's prebuilt
// kernels (MM_MT=1, MM_KT=9, MM_NT=2) so we can reuse them as-is.
constexpr uint32_t kCin  = 32;
constexpr uint32_t kCout = 32;     // residual add requires Cin == Cout
constexpr uint32_t kH    = 8;
constexpr uint32_t kW    = 8;

constexpr uint32_t kKH = 3, kKW = 3, kPad = 1;

constexpr uint32_t kMt = 1;        // C_out / 32
constexpr uint32_t kKt = 9;        // C_in * 9 / 32  (32*9 / 32)
constexpr uint32_t kNt = 2;        // H*W / 32       (64 / 32)
constexpr uint32_t kKdim = kCin * kKH * kKW;  // 288
constexpr uint32_t kHW   = kH * kW;           // 64

// residual_add tile count: one bf16 32×32 tile per Mt*Nt slot.
constexpr uint32_t kRaNt = kMt * kNt;         // 2

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// ---------------------------------------------------------------------------
// Reference math (fp32, single bf16 rounding per "stage" to match device
// rounding behaviour).
// ---------------------------------------------------------------------------

void conv3x3_bf16(const std::vector<uint16_t>& x_chw,
                  const std::vector<uint16_t>& w_cchw,
                  std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * kH * kW, 0);
    for (uint32_t co = 0; co < kCout; ++co) {
        for (uint32_t h = 0; h < kH; ++h) {
            for (uint32_t w_ = 0; w_ < kW; ++w_) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kCin; ++ci)
                    for (uint32_t ki = 0; ki < kKH; ++ki)
                        for (uint32_t kj = 0; kj < kKW; ++kj) {
                            int ih = static_cast<int>(h + ki) - static_cast<int>(kPad);
                            int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                            if (ih < 0 || ih >= static_cast<int>(kH) ||
                                iw < 0 || iw >= static_cast<int>(kW)) continue;
                            acc += bf16_to_f32(x_chw[(ci * kH + ih) * kW + iw]) *
                                   bf16_to_f32(w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj]);
                        }
                y_chw[(co * kH + h) * kW + w_] = f32_to_bf16(acc);
            }
        }
    }
}

// y[co, h, w] = ReLU(y[co, h, w] + bias[co]); one bf16 round.
void bias_and_relu(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t i = 0; i < kHW; ++i) {
            float v = bf16_to_f32(y_chw[co * kHW + i]) + bias[co];
            if (v < 0.0f) v = 0.0f;
            y_chw[co * kHW + i] = f32_to_bf16(v);
        }
}

// y += bias broadcast (no ReLU).
void bias_only(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t i = 0; i < kHW; ++i) {
            float v = bf16_to_f32(y_chw[co * kHW + i]) + bias[co];
            y_chw[co * kHW + i] = f32_to_bf16(v);
        }
}

void relu_inplace(std::vector<uint16_t>& y) {
    for (auto& v : y) {
        float f = bf16_to_f32(v);
        if (f < 0.0f) v = 0;
    }
}

// ---------------------------------------------------------------------------
// Tile <-> row-major helpers (mirror test_conv_3x3 / test_residual_add).
// ---------------------------------------------------------------------------

// X (C_in, H, W)  →  A (C_in*9, H*W)  row-major.
void im2col(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& a) {
    a.assign(kKdim * kHW, 0);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t ki = 0; ki < kKH; ++ki)
            for (uint32_t kj = 0; kj < kKW; ++kj) {
                uint32_t row = ci * (kKH * kKW) + ki * kKW + kj;
                for (uint32_t h = 0; h < kH; ++h) {
                    int ih = static_cast<int>(h + ki) - static_cast<int>(kPad);
                    if (ih < 0 || ih >= static_cast<int>(kH)) continue;
                    for (uint32_t w_ = 0; w_ < kW; ++w_) {
                        int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                        if (iw < 0 || iw >= static_cast<int>(kW)) continue;
                        a[row * kHW + h * kW + w_] = x_chw[(ci * kH + ih) * kW + iw];
                    }
                }
            }
}

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

// CHW activations are laid out as (Cout, H*W) row-major when used as
// either A or B of the next conv's im2col. The bf16 stream of RA_NT
// tiles for residual_add is the same (Cout, H*W) tiled stream.
void chw_to_tile_stream(const std::vector<uint16_t>& chw,
                        std::vector<uint16_t>& tiles) {
    tile_matrix(chw, kMt, kNt, kHW, tiles);
}

void tile_stream_to_chw(const std::vector<uint16_t>& tiles,
                        std::vector<uint16_t>& chw) {
    untile_matrix(tiles, kMt, kNt, kHW, chw);
}

}  // namespace

int main() try {
    const std::string conv_dir = required_env("TT_FOIL_KERNEL_DIR");
    const std::string add_dir  = required_env("TT_FOIL_KERNEL_DIR_ADD");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Host-side inputs / BN-folded params -----------------------
    std::mt19937 rng(0xb16ca5e);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x_chw(kCin * kHW);
    for (auto& v : x_chw) v = f32_to_bf16(u(rng));

    auto make_weights = [&](std::vector<uint16_t>& w) {
        w.assign(kCout * kCin * kKH * kKW, 0);
        for (auto& v : w) v = f32_to_bf16(0.05f * u(rng));
    };
    std::vector<uint16_t> w1, w2;
    make_weights(w1);
    make_weights(w2);

    // BN-folded biases are arbitrary fp32 per output channel.
    auto make_bias = [&]() {
        std::vector<float> b(kCout);
        for (auto& v : b) v = 0.1f * u(rng);
        return b;
    };
    std::vector<float> b1 = make_bias();
    std::vector<float> b2 = make_bias();

    // ---- Reference -------------------------------------------------
    std::vector<uint16_t> y_ref_chw;
    {
        std::vector<uint16_t> t1, t2;
        conv3x3_bf16(x_chw, w1, t1);
        bias_and_relu(t1, b1);                    // ReLU(Conv₁(x) + b₁')
        conv3x3_bf16(t1, w2, t2);
        bias_only(t2, b2);                        //   Conv₂(...) + b₂'
        // skip add: y = t2 + x (one bf16 round per output element)
        y_ref_chw.assign(kCin * kHW, 0);
        for (uint32_t i = 0; i < y_ref_chw.size(); ++i)
            y_ref_chw[i] = f32_to_bf16(bf16_to_f32(t2[i]) + bf16_to_f32(x_chw[i]));
        relu_inplace(y_ref_chw);                  // ReLU(skip)
    }

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // Matmul-shaped DRAM scratch + L1 CB buffers.  We reuse the same
    // three DRAM buffers across both conv calls (re-written each time)
    // and a fourth DRAM buffer pair for the residual add stage.
    const uint32_t w_bytes = kMt * kKt * kTileBytes;     // weights tiled (M, K)
    const uint32_t a_bytes = kKt * kNt * kTileBytes;     // im2col tiled (K, N)
    const uint32_t y_bytes = kMt * kNt * kTileBytes;     // (M, N) output tiles

    auto buf_W       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes,    core);
    auto buf_A       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes,    core);
    auto buf_Y       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes,    core);
    // Skip-path bufs: one holds the post-conv₂ tensor, one holds the
    // original x (also re-used as residual_add output).
    auto buf_skip_x  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes,    core);
    auto buf_skip_y  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes,    core);

    auto buf_cb_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);
    auto buf_cb_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);

    uint64_t W_noc      = tt::foil::make_noc_dram_addr(*dev, buf_W      ->device_addr);
    uint64_t A_noc      = tt::foil::make_noc_dram_addr(*dev, buf_A      ->device_addr);
    uint64_t Y_noc      = tt::foil::make_noc_dram_addr(*dev, buf_Y      ->device_addr);
    uint64_t skipx_noc  = tt::foil::make_noc_dram_addr(*dev, buf_skip_x ->device_addr);
    uint64_t skipy_noc  = tt::foil::make_noc_dram_addr(*dev, buf_skip_y ->device_addr);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    using R = tt::foil::RiscBinary;

    // ---- Load both kernel programs.  Each load_kernel rewrites the
    // per-RISC kernel-text region in L1, so we only have ONE active
    // program on the core at a time; switching between them is a
    // matter of which Kernel handle the next execute() call uses.
    std::array<R, 5> conv_bins = {{
        {R::RiscId::BRISC,  conv_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, conv_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, conv_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, conv_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, conv_dir + "/compute.trisc2.elf"},
    }};
    std::array<R, 5> add_bins = {{
        {R::RiscId::BRISC,  add_dir  + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, add_dir  + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, add_dir  + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, add_dir  + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, add_dir  + "/compute.trisc2.elf"},
    }};
    auto k_conv = tt::foil::load_kernel(*dev, conv_bins, core);
    auto k_add  = tt::foil::load_kernel(*dev, add_bins,  core);

    // Identical CB layout for both — three 1-tile bf16 CBs.
    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *k_conv, cbs);
    tt::foil::register_cbs(*dev, *k_add,  cbs);

    // Helper: run a single conv₃ₓ₃ pass. Inputs are CHW; weights are CCHW.
    // Returns the post-conv activation in CHW (no bias, no ReLU).
    auto run_conv = [&](const std::vector<uint16_t>& x_in_chw,
                        const std::vector<uint16_t>& w_cchw,
                        std::vector<uint16_t>& y_out_chw) {
        std::vector<uint16_t> w_mat, w_tiles;
        weight_reshape(w_cchw, w_mat);
        tile_matrix(w_mat, kMt, kKt, kKdim, w_tiles);

        std::vector<uint16_t> a_mat, a_tiles;
        im2col(x_in_chw, a_mat);
        tile_matrix(a_mat, kKt, kNt, kHW, a_tiles);

        tt::foil::write_buffer(*dev, *buf_W, w_tiles.data(), w_bytes);
        tt::foil::write_buffer(*dev, *buf_A, a_tiles.data(), a_bytes);
        std::vector<uint8_t> zero(y_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_Y, zero.data(), y_bytes);

        std::array<uint32_t, 7> ra_brisc = {
            lo(W_noc), hi(W_noc),
            lo(A_noc), hi(A_noc),
            kMt, kKt, kNt,
        };
        std::array<uint32_t, 3> ra_ncrisc = {
            lo(Y_noc), hi(Y_noc), kMt * kNt,
        };
        tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::NCRISC, ra_ncrisc);

        // Re-register CBs is required: register_cbs(k_add, ...) above
        // wrote the launch_msg cb table with k_add's binding. Doing
        // it again before each conv execute keeps the conv kernel's
        // CB descriptors fresh in the launch_msg even after k_add
        // has run.
        tt::foil::register_cbs(*dev, *k_conv, cbs);
        tt::foil::execute(*dev, *k_conv);

        std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_Y, y_tiles.data(), y_bytes);
        tile_stream_to_chw(y_tiles, y_out_chw);
    };

    // ---- Stage 1 — Conv₁ on device, bias + ReLU on host -----------
    std::vector<uint16_t> t1;
    run_conv(x_chw, w1, t1);
    bias_and_relu(t1, b1);

    // ---- Stage 2 — Conv₂ on device, bias on host ------------------
    std::vector<uint16_t> t2;
    run_conv(t1, w2, t2);
    bias_only(t2, b2);

    // ---- Stage 3 — device residual add: t2 + x --------------------
    std::vector<uint16_t> t2_tiles, x_tiles;
    chw_to_tile_stream(t2,    t2_tiles);
    chw_to_tile_stream(x_chw, x_tiles);

    // The residual_add reader expects A and B in DRAM; we re-use
    // buf_skip_x and buf_skip_y for this stage. Output goes into
    // buf_Y.
    const uint32_t add_bytes = kRaNt * kTileBytes;
    tt::foil::write_buffer(*dev, *buf_skip_x, t2_tiles.data(), add_bytes);
    tt::foil::write_buffer(*dev, *buf_skip_y, x_tiles.data(),  add_bytes);
    std::vector<uint8_t> zero(add_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_Y, zero.data(), add_bytes);

    std::array<uint32_t, 4> ra_add_brisc = {
        lo(skipx_noc), hi(skipx_noc),
        lo(skipy_noc), hi(skipy_noc),
    };
    std::array<uint32_t, 2> ra_add_ncrisc = {
        lo(Y_noc), hi(Y_noc),
    };
    tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::BRISC,  ra_add_brisc);
    tt::foil::set_runtime_args(*dev, *k_add, R::RiscId::NCRISC, ra_add_ncrisc);
    tt::foil::register_cbs(*dev, *k_add, cbs);
    tt::foil::execute(*dev, *k_add);

    std::vector<uint16_t> y_tiles(kRaNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_Y, y_tiles.data(), add_bytes);
    std::vector<uint16_t> y_dev_chw;
    tile_stream_to_chw(y_tiles, y_dev_chw);

    relu_inplace(y_dev_chw);

    // ---- Compare ----------------------------------------------------
    // Four bf16 rounding stages (Conv₁, ReLU₁, Conv₂, residual+ReLU)
    // accumulate ~few-percent worst-case relative error. The values
    // in this test sit in roughly [-0.5, 1.5] so an absolute floor of
    // 0.15 plus 3% relative is comfortable.
    const float kAbsTol = 0.15f;
    const float kRelTol = 0.03f;
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
            "test_basic_block: %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, y_dev_chw.size(), first_bad,
            bf16_to_f32(y_dev_chw[first_bad]),
            bf16_to_f32(y_ref_chw[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_basic_block: FAIL");
        return 1;
    }

    std::printf("test_basic_block: PASS  (C=%u H=%u W=%u, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kCin, kH, kW, worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_basic_block: FAIL — %s\n", e.what());
    return 1;
}
