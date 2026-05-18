// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ResNet classifier tail — the operations that turn the feature map
// from the last residual block into per-class logits:
//
//   gap(x) [c]    = mean over (h, w) of x[c, h, w]            (host)
//   logits[k]     = sum_c W[k, c] · gap[c] + bias[k]          (device)
//
// Input shape matches the mini_resnet output: (C=32, H=8, W=8).
// FC fan-out: num_classes = 32. (Real ResNet uses 1000; we pick 32 so
// the matmul stays one tile wide in M without weird padding — the test
// is about the chain mechanics, not the class count.)
//
// FC is just a matmul with a degenerate N dimension: there's only one
// "input vector" but the device matmul kernel needs a 32-aligned N. We
// pad X to (Cin=32, 32) where column 0 is the real input and columns
// 1..31 are zero. Y comes back at (Cout=32, 32); we read column 0.

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

constexpr uint32_t kC     = 32;        // feature channels
constexpr uint32_t kH     = 8;
constexpr uint32_t kW     = 8;
constexpr uint32_t kNcl   = 32;        // num_classes

// FC matmul shape: M=Ncl, K=C, N=1 → one tile per dim.
constexpr uint32_t kMt = kNcl / kTileH;   // 1
constexpr uint32_t kKt = kC   / kTileW;   // 1
constexpr uint32_t kNt = 1;               // pad input to a full tile column

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
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

}  // namespace

int main() try {
    const std::string kernel_root = required_env("TT_FOIL_KERNEL_DIR");
    const std::string fc_dir      = kernel_root + "/fc";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Inputs ----------------------------------------------------
    std::mt19937 rng(0xc1a55);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);

    std::vector<uint16_t> x_chw(kC * kH * kW);
    for (auto& v : x_chw) v = f32_to_bf16(u(rng));

    std::vector<uint16_t> W2d(kNcl * kC);   // (num_classes, Cin)
    for (auto& v : W2d) v = f32_to_bf16(0.1f * u(rng));

    std::vector<float> bias(kNcl);
    for (auto& v : bias) v = 0.1f * u(rng);

    // ---- Reference -------------------------------------------------
    std::vector<float> gap(kC, 0.0f);
    for (uint32_t c = 0; c < kC; ++c) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < kH * kW; ++i)
            acc += bf16_to_f32(x_chw[c * (kH * kW) + i]);
        gap[c] = acc / static_cast<float>(kH * kW);
    }
    std::vector<uint16_t> gap_bf16(kC);
    for (uint32_t c = 0; c < kC; ++c) gap_bf16[c] = f32_to_bf16(gap[c]);

    std::vector<uint16_t> y_ref(kNcl);
    for (uint32_t k = 0; k < kNcl; ++k) {
        float acc = 0.0f;
        for (uint32_t c = 0; c < kC; ++c)
            acc += bf16_to_f32(W2d[k * kC + c]) * bf16_to_f32(gap_bf16[c]);
        y_ref[k] = f32_to_bf16(acc + bias[k]);
    }

    // ---- Build the matmul operands --------------------------------
    // W tile: (kNcl, kC) row-major fits exactly in one tile.
    std::vector<uint16_t> W_mat(kNcl * kC);
    for (uint32_t r = 0; r < kNcl; ++r)
        for (uint32_t c = 0; c < kC; ++c)
            W_mat[r * kC + c] = W2d[r * kC + c];

    // X "matrix": (kC, 32) row-major, column 0 = gap, others zero.
    std::vector<uint16_t> X_mat(kC * kTileW, 0);
    for (uint32_t r = 0; r < kC; ++r) X_mat[r * kTileW + 0] = gap_bf16[r];

    std::vector<uint16_t> w_tiles, x_tiles;
    tile_matrix(W_mat, kMt, kKt, kC,     w_tiles);
    tile_matrix(X_mat, kKt, kNt, kTileW, x_tiles);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    const uint32_t w_bytes = kMt * kKt * kTileBytes;
    const uint32_t x_bytes = kKt * kNt * kTileBytes;
    const uint32_t y_bytes = kMt * kNt * kTileBytes;

    auto buf_W   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes, core);
    auto buf_X   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes, core);
    auto buf_Y   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_W, w_tiles.data(), w_bytes);
    tt::foil::write_buffer(*dev, *buf_X, x_tiles.data(), x_bytes);
    std::vector<uint8_t> zero(y_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_Y, zero.data(), y_bytes);

    uint64_t W_noc = tt::foil::make_noc_dram_addr(*dev, buf_W->device_addr);
    uint64_t X_noc = tt::foil::make_noc_dram_addr(*dev, buf_X->device_addr);
    uint64_t Y_noc = tt::foil::make_noc_dram_addr(*dev, buf_Y->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  fc_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, fc_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, fc_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, fc_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, fc_dir + "/compute.trisc2.elf"},
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
        lo(W_noc), hi(W_noc), lo(X_noc), hi(X_noc),
        kMt, kKt, kNt,
    };
    std::array<uint32_t, 3> ra_ncrisc = { lo(Y_noc), hi(Y_noc), kMt * kNt };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    // ---- Read back, untile, pull column 0 ---------------------------
    std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_Y, y_tiles.data(), y_bytes);
    std::vector<uint16_t> y_mat;
    untile_matrix(y_tiles, kMt, kNt, kTileW, y_mat);   // (kNcl, 32)

    std::vector<uint16_t> y_dev(kNcl);
    for (uint32_t k = 0; k < kNcl; ++k)
        y_dev[k] = f32_to_bf16(bf16_to_f32(y_mat[k * kTileW + 0]) + bias[k]);

    // ---- Compare ---------------------------------------------------
    // Three bf16 rounding stages (gap mean, matmul accumulate, +bias).
    const float kAbsTol = 0.05f;
    const float kRelTol = 0.02f;
    uint32_t bad = 0, first_bad = kNcl;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (uint32_t k = 0; k < kNcl; ++k) {
        float got = bf16_to_f32(y_dev[k]);
        float exp = bf16_to_f32(y_ref[k]);
        float d = std::fabs(got - exp);
        float ref = std::fabs(exp);
        float tol = std::max(kAbsTol, kRelTol * ref);
        if (d > worst_abs) worst_abs = d;
        if (ref > 0.f && (d / ref) > worst_rel) worst_rel = d / ref;
        if (d > tol) {
            if (first_bad == kNcl) first_bad = k;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_classifier_tail: %u/%u mismatches; first at class %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            bad, kNcl, first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(y_ref[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_classifier_tail: FAIL");
        return 1;
    }

    // Find argmax for a friendly print — pretend we just classified.
    uint32_t argmax = 0;
    float    max_logit = bf16_to_f32(y_dev[0]);
    for (uint32_t k = 1; k < kNcl; ++k) {
        float v = bf16_to_f32(y_dev[k]);
        if (v > max_logit) { max_logit = v; argmax = k; }
    }

    std::printf("test_classifier_tail: PASS  "
                "(C=%u %ux%u → GAP → FC %u-way; argmax=%u logit=%.4f, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kC, kH, kW, kNcl, argmax, max_logit,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_classifier_tail: FAIL — %s\n", e.what());
    return 1;
}
