// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ResNet stem — chains the two stem operators end-to-end on one Tensix core:
//
//   y = Maxpool₃ₓ₃, s=2, pad=1 ( ReLU( Conv₇ₓ₇, s=2, pad=3 (x) + b ) )
//
// Shape (a scaled-down mirror of real ResNet's 224 → 112 → 56 pipeline):
//   x        : (C=32, H=W=32)
//   conv-out : (C=32, H=W=16)
//   y        : (C=32, H=W=8)   = the activation that feeds layer1.
//
// Device kernels:
//   • Conv₇ₓ₇  — matmul_dram-shape kernels built fresh for Mt=1, Kt=49,
//                 Nt=8 (the HW_mid = 256 = 8-tile N dimension).
//   • Maxpool₃ₓ₃ — examples/maxpool_3x3 prebuilt, default fixture
//                  (H_mid → H_out = 16 → 8).
//
// Host glue:
//   • im2col(7×7, s=2, pad=3) on the input,
//   • bias-broadcast + ReLU on the conv output,
//   • CHW → (H, W, C) repack + 9-window gather (with -∞ pad sentinels)
//     to feed the maxpool reader,
//   • untile the (H_out * W_out, C) maxpool output.

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

// ---- Shape ---------------------------------------------------------------
constexpr uint32_t kC      = 32;
constexpr uint32_t kHin    = 32;
constexpr uint32_t kWin    = 32;
constexpr uint32_t kHmid   = 16;
constexpr uint32_t kWmid   = 16;
constexpr uint32_t kHout   = 8;
constexpr uint32_t kWout   = 8;

// conv_7x7 params
constexpr uint32_t kK7   = 7;
constexpr uint32_t kPad7 = 3;
constexpr uint32_t kStride7 = 2;

// maxpool_3x3 params
constexpr uint32_t kPadP = 1;
constexpr uint32_t kStrideP = 2;

// Matmul shape for the conv stage.
constexpr uint32_t kMt    = kC / kTileH;             // 1
constexpr uint32_t kKt    = (kC * kK7 * kK7) / kTileW; // 49
constexpr uint32_t kNt    = (kHmid * kWmid) / kTileW;  // 8
constexpr uint32_t kKdim  = kKt * kTileW;              // 1568
constexpr uint32_t kHWmid = kHmid * kWmid;             // 256

// Maxpool output tile count.
constexpr uint32_t kNtPool = (kHout * kWout) / kTileH; // 2

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// ---- Reference -----------------------------------------------------------

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

void bias_and_relu_chw(std::vector<uint16_t>& y_chw, const std::vector<float>& bias) {
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t i = 0; i < kHWmid; ++i) {
            float v = bf16_to_f32(y_chw[co * kHWmid + i]) + bias[co];
            if (v < 0.0f) v = 0.0f;
            y_chw[co * kHWmid + i] = f32_to_bf16(v);
        }
}

void maxpool3x3_bf16_ref(const std::vector<uint16_t>& x_hwc,
                         std::vector<uint16_t>& y_hwc) {
    y_hwc.assign(kHout * kWout * kC, 0);
    for (uint32_t oh = 0; oh < kHout; ++oh)
        for (uint32_t ow = 0; ow < kWout; ++ow)
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
                        float v = bf16_to_f32(x_hwc[(static_cast<uint32_t>(ih) * kWmid +
                                                     static_cast<uint32_t>(iw)) * kC + c]);
                        if (v > m) m = v;
                    }
                y_hwc[(oh * kWout + ow) * kC + c] = f32_to_bf16(m);
            }
}

// ---- Layout helpers ------------------------------------------------------

void im2col_7x7(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& a) {
    a.assign(kKdim * kHWmid, 0);
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
    w_mat.assign(kC * kKdim, 0);
    for (uint32_t co = 0; co < kC; ++co)
        for (uint32_t ci = 0; ci < kC; ++ci)
            for (uint32_t ki = 0; ki < kK7; ++ki)
                for (uint32_t kj = 0; kj < kK7; ++kj)
                    w_mat[co * kKdim + ci * (kK7 * kK7) + ki * kK7 + kj] =
                        w_cchw[((co * kC + ci) * kK7 + ki) * kK7 + kj];
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

// (C, H, W) → (H, W, C) row-major transpose.
void chw_to_hwc(const std::vector<uint16_t>& chw, uint32_t H, uint32_t W,
                std::vector<uint16_t>& hwc) {
    hwc.assign(H * W * kC, 0);
    for (uint32_t c = 0; c < kC; ++c)
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t w = 0; w < W; ++w)
                hwc[(h * W + w) * kC + c] = chw[(c * H + h) * W + w];
}

// Same row-major-block to tile-face helper used by maxpool_2x2 / maxpool_3x3.
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

}  // namespace

int main() try {
    static_assert(kC == kTileW, "stem test assumes C fits in one channel tile");

    const std::string kernel_root = required_env("TT_FOIL_KERNEL_DIR");
    const std::string conv_dir    = kernel_root + "/conv_7x7";
    const std::string pool_dir    = kernel_root + "/maxpool_3x3";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Inputs / weights / bias -----------------------------------
    std::mt19937 rng(0x57ec);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x_chw(kC * kHin * kWin);
    for (auto& v : x_chw) v = f32_to_bf16(u(rng));

    std::vector<uint16_t> w_cchw(kC * kC * kK7 * kK7);
    for (auto& v : w_cchw) v = f32_to_bf16(0.02f * u(rng));

    std::vector<float> bias(kC);
    for (auto& v : bias) v = 0.1f * u(rng);

    // ---- Reference -------------------------------------------------
    std::vector<uint16_t> y_ref;
    {
        std::vector<uint16_t> conv_out, conv_hwc;
        conv7x7_s2_bf16(x_chw, w_cchw, conv_out);
        bias_and_relu_chw(conv_out, bias);
        chw_to_hwc(conv_out, kHmid, kWmid, conv_hwc);
        maxpool3x3_bf16_ref(conv_hwc, y_ref);
    }

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // ---- Conv stage buffers ----------------------------------------
    const uint32_t w_bytes  = kMt * kKt * kTileBytes;
    const uint32_t a_bytes  = kKt * kNt * kTileBytes;
    const uint32_t y_bytes  = kMt * kNt * kTileBytes;     // 1*8 = 8 tiles
    auto buf_W = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes, core);
    auto buf_A = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes, core);
    auto buf_Y = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);
    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    // ---- Maxpool stage buffers (separate from conv to keep L1 layout
    //      stable across kernel switches) -----------------------------
    const uint32_t stream_bytes = kNtPool * kTileBytes;
    std::array<std::shared_ptr<tt::foil::Buffer>, 9> buf_pool_in;
    for (uint32_t s = 0; s < 9; ++s)
        buf_pool_in[s] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    auto buf_pool_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    std::array<std::shared_ptr<tt::foil::Buffer>, 10> pool_cb_bufs;
    for (uint32_t i = 0; i < 10; ++i)
        pool_cb_bufs[i] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    auto noc_of = [&](auto& buf) { return tt::foil::make_noc_dram_addr(*dev, buf->device_addr); };
    uint64_t W_noc = noc_of(buf_W);
    uint64_t A_noc = noc_of(buf_A);
    uint64_t Y_noc = noc_of(buf_Y);

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
    auto k_conv = load(conv_dir);
    auto k_pool = load(pool_dir);

    std::array<tt::foil::CbConfig, 3> conv_cbs = {{
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

    // ================ Stage 1 — Conv₇ₓ₇ ============================
    std::vector<uint16_t> w_mat, w_tiles, a_mat, a_tiles;
    weight_7x7_reshape(w_cchw, w_mat);
    tile_matrix(w_mat, kMt, kKt, kKdim, w_tiles);
    im2col_7x7(x_chw, a_mat);
    tile_matrix(a_mat, kKt, kNt, kHWmid, a_tiles);

    tt::foil::write_buffer(*dev, *buf_W, w_tiles.data(), w_bytes);
    tt::foil::write_buffer(*dev, *buf_A, a_tiles.data(), a_bytes);
    std::vector<uint8_t> zero_y(y_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_Y, zero_y.data(), y_bytes);

    std::array<uint32_t, 7> ra_brisc = {
        lo(W_noc), hi(W_noc), lo(A_noc), hi(A_noc),
        kMt, kKt, kNt,
    };
    std::array<uint32_t, 3> ra_ncrisc = { lo(Y_noc), hi(Y_noc), kMt * kNt };
    tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *k_conv, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::register_cbs(*dev, *k_conv, conv_cbs);
    tt::foil::execute(*dev, *k_conv);

    std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_Y, y_tiles.data(), y_bytes);
    std::vector<uint16_t> conv_out_chw;
    untile_matrix(y_tiles, kMt, kNt, kHWmid, conv_out_chw);   // (C, HW_mid)

    // ---- Stage 2 — bias + ReLU (host) ------------------------------
    bias_and_relu_chw(conv_out_chw, bias);

    // ---- Stage 3 — repack to (H, W, C) + gather 9 maxpool streams --
    std::vector<uint16_t> conv_hwc;
    chw_to_hwc(conv_out_chw, kHmid, kWmid, conv_hwc);

    const uint16_t kNegInf = f32_to_bf16(-1.0e30f);
    auto idx_hwc = [&](uint32_t h, uint32_t w, uint32_t c) {
        return (h * kWmid + w) * kC + c;
    };
    std::array<std::vector<uint16_t>, 9> streams_rm;
    for (auto& s : streams_rm) s.assign(kHout * kWout * kC, kNegInf);

    for (uint32_t s = 0; s < 9; ++s) {
        int di = static_cast<int>(s / 3);
        int dj = static_cast<int>(s % 3);
        for (uint32_t oh = 0; oh < kHout; ++oh) {
            int ih = static_cast<int>(oh) * static_cast<int>(kStrideP) + di -
                     static_cast<int>(kPadP);
            if (ih < 0 || ih >= static_cast<int>(kHmid)) continue;
            for (uint32_t ow = 0; ow < kWout; ++ow) {
                int iw = static_cast<int>(ow) * static_cast<int>(kStrideP) + dj -
                         static_cast<int>(kPadP);
                if (iw < 0 || iw >= static_cast<int>(kWmid)) continue;
                for (uint32_t c = 0; c < kC; ++c)
                    streams_rm[s][(oh * kWout + ow) * kC + c] =
                        conv_hwc[idx_hwc(static_cast<uint32_t>(ih),
                                         static_cast<uint32_t>(iw), c)];
            }
        }
    }

    std::array<std::vector<uint16_t>, 9> streams_tiles;
    for (uint32_t s = 0; s < 9; ++s)
        rm_matrix_to_tiles(streams_rm[s], kHout * kWout, streams_tiles[s]);
    for (uint32_t s = 0; s < 9; ++s)
        tt::foil::write_buffer(*dev, *buf_pool_in[s], streams_tiles[s].data(), stream_bytes);
    std::vector<uint16_t> zero_pool(stream_bytes / 2, 0);
    tt::foil::write_buffer(*dev, *buf_pool_out, zero_pool.data(), stream_bytes);

    // ================ Stage 4 — Maxpool₃ₓ₃ =========================
    std::array<uint32_t, 10> pool_ra_brisc = {
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
    std::array<uint32_t, 2> pool_ra_ncrisc = {
        static_cast<uint32_t>(buf_pool_out->device_addr), kNtPool
    };
    std::array<uint32_t, 1> pool_ra_compute = { kNtPool };
    tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::BRISC,  pool_ra_brisc);
    tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::NCRISC, pool_ra_ncrisc);
    tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC0, pool_ra_compute);
    tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC1, pool_ra_compute);
    tt::foil::set_runtime_args(*dev, *k_pool, R::RiscId::TRISC2, pool_ra_compute);
    tt::foil::register_cbs(*dev, *k_pool, pool_cbs);
    tt::foil::execute(*dev, *k_pool);

    std::vector<uint16_t> pool_tiles(stream_bytes / 2, 0);
    tt::foil::read_buffer(*dev, *buf_pool_out, pool_tiles.data(), stream_bytes);

    std::vector<uint16_t> y_dev(kHout * kWout * kC);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNtPool; ++t) {
        tt::foil::test::tile_to_row_major(pool_tiles.data() + t * kTileWords, block.data());
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                y_dev[(t * kTileH + r) * kTileW + c] = block[r * kTileW + c];
    }

    // ---- Compare ---------------------------------------------------
    // bf16 rounding stages: conv accumulate (1 round), bias+ReLU (1),
    // maxpool (selection, 0). Two rounding stages → modest tolerance.
    const float kAbsTol = 0.10f;
    const float kRelTol = 0.02f;
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
            "test_stem: %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, y_dev.size(), first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(y_ref[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_stem: FAIL");
        return 1;
    }

    std::printf("test_stem: PASS  (C=%u %ux%u→%ux%u→%ux%u, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kC, kHin, kWin, kHmid, kWmid, kHout, kWout,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_stem: FAIL — %s\n", e.what());
    return 1;
}
