// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-0.6B layer-0 MLP half on real weights.
//
//   y_norm  = RMSNorm(x, ln2_gamma, eps)
//   gate    = y_norm · W_gate
//   up      = y_norm · W_up
//   mlp_out = SiLU(gate) ⊙ up  → · W_down
//   y_out   = x + mlp_out         (residual)
//
// Data files are gitignored (~31 MB of bf16 weights); regenerate them with:
//
//   pip install huggingface_hub safetensors torch
//   python3 tools/export_qwen3_layer.py --model Qwen/Qwen3-0.6B --layer 0 \
//       --out-dir data/qwen3_06b
//   python3 tools/qwen3_mlp_golden.py --layer-dir data/qwen3_06b/layer0 --seq 32
//
// Produces:
//   data/qwen3_06b/layer0/{ln2_gamma,W_gate,W_up,W_down}.bin
//   data/qwen3_06b/layer0/{mlp_input,mlp_golden}.bin
//
// The CMake helper passes TT_FOIL_QWEN3_DATA at it.
//
// All weight binaries are bf16 row-major; this test tiles them on the
// host before uploading.  H = 1024 (Ht=32), FFN = 3072 (FFt=96), S = 32
// (St=1).  Comparison against the numpy reference uses a relatively
// loose absolute tolerance because the chain hits the rsqrt and the
// SiLU SFPU paths, both of which contribute several bits of BF16 drift.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
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
using tt::foil::test::bf16_to_f32;

// Qwen3-0.6B layer-0 geometry.
constexpr uint32_t kS    = 32;
constexpr uint32_t kH    = 1024;
constexpr uint32_t kFFN  = 3072;
constexpr uint32_t kSt   = kS   / kTileH;   // 1
constexpr uint32_t kHt   = kH   / kTileW;   // 32
constexpr uint32_t kFFt  = kFFN / kTileW;   // 96
constexpr float    kEps  = 1e-6f;

std::vector<uint16_t> load_bin(const std::string& path, std::size_t expect_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + path);
    std::vector<uint16_t> v(expect_elems);
    f.read(reinterpret_cast<char*>(v.data()), expect_elems * 2);
    if (f.gcount() != static_cast<std::streamsize>(expect_elems * 2))
        throw std::runtime_error("short read: " + path);
    return v;
}

// Row-major [Rows, Cols] → tile-format stream.
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
std::vector<uint16_t> untile2d(const std::vector<uint16_t>& tiles, uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> rm(Rows * Cols, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < Rt; ++rt)
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            tt::foil::test::tile_to_row_major(tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    rm[(rt * kTileH + r) * Cols + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }
    return rm;
}

// RMSNorm in our kernel reads gamma as Wt tiles where each tile has gamma
// values replicated across all 32 rows. Build that tile-format payload.
std::vector<uint16_t> gamma_to_tiles(const std::vector<uint16_t>& gamma, uint32_t H) {
    std::vector<uint16_t> rm(kTileH * H);
    for (uint32_t r = 0; r < kTileH; ++r)
        for (uint32_t c = 0; c < H; ++c) rm[r * H + c] = gamma[c];
    return tile2d(rm, kTileH, H);
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    const std::string base = data_env ? data_env : "data/qwen3_06b/layer0";

    auto x_rm       = load_bin(base + "/mlp_input.bin",  kS * kH);
    auto golden_rm  = load_bin(base + "/mlp_golden.bin", kS * kH);
    auto ln2g_flat  = load_bin(base + "/ln2_gamma.bin",  kH);
    auto Wgate_rm   = load_bin(base + "/W_gate.bin",     kH * kFFN);
    auto Wup_rm     = load_bin(base + "/W_up.bin",       kH * kFFN);
    auto Wdown_rm   = load_bin(base + "/W_down.bin",     kFFN * kH);

    auto x_tiles      = tile2d(x_rm,      kS,   kH);
    auto Wgate_tiles  = tile2d(Wgate_rm,  kH,   kFFN);
    auto Wup_tiles    = tile2d(Wup_rm,    kH,   kFFN);
    auto Wdown_tiles  = tile2d(Wdown_rm,  kFFN, kH);
    auto ln2g_tiles   = gamma_to_tiles(ln2g_flat, kH);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

    auto T_x      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ln2g   = ol::allocate_tensor_dram(*dev, kHt);
    auto T_Wgate  = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wup    = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wdown  = ol::allocate_tensor_dram(*dev, kFFt * kHt);
    auto T_ynorm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up     = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_out    = ol::allocate_tensor_dram(*dev, kSt * kHt);

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    upload(T_x,     x_tiles);
    upload(T_ln2g,  ln2g_tiles);
    upload(T_Wgate, Wgate_tiles);
    upload(T_Wup,   Wup_tiles);
    upload(T_Wdown, Wdown_tiles);

    auto run = [&](auto factory) {
        auto op = factory();
        ol::execute(*dev, op);
    };

    run([&]() { return ol::make_rmsnorm(*dev, T_x, T_ln2g, T_ynorm, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_ynorm, T_Wgate, T_gate, kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_ynorm, T_Wup,   T_up,   kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_silu(*dev, T_gate, T_silu); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_fused, T_Wdown, T_down, kSt, kFFt, kHt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_eltwise_add(*dev, T_x, T_down, T_out); });
    tt::foil::release_kernels(*dev, core);

    std::vector<uint16_t> out_tiles(kSt * kHt * kTileWords);
    tt::foil::read_buffer(*dev, *T_out.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, kH);

    // Tolerances: BF16 matmul + rsqrt + SFPU through 7 ops on full Qwen3
    // weight magnitudes drifts a few percent.  Use both absolute and
    // relative thresholds, matching what tt-metal's golden-comparison
    // tests use for similar BF16 chains.
    const float kAbsTol = 0.15f;
    const float kRelTol = 0.05f;
    uint32_t bad = 0; float worst_abs = 0.0f; float worst_rel = 0.0f;
    uint32_t first_bad = kS * kH;
    for (uint32_t i = 0; i < kS * kH; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = bf16_to_f32(golden_rm[i]);
        float d_abs = std::fabs(got - exp);
        float d_rel = d_abs / std::max(std::fabs(exp), 1e-3f);
        if (d_abs > worst_abs) worst_abs = d_abs;
        if (d_rel > worst_rel) worst_rel = d_rel;
        if (d_abs > kAbsTol && d_rel > kRelTol) {
            if (first_bad == kS * kH) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_qwen3_mlp: %u/%u mismatches; first at i=%u: got=%.4f expected=%.4f, "
            "worst abs=%.5f rel=%.4f (tol abs=%.5f rel=%.4f)\n",
            bad, kS * kH, first_bad,
            bf16_to_f32(out_rm[first_bad]), bf16_to_f32(golden_rm[first_bad]),
            worst_abs, worst_rel, kAbsTol, kRelTol);
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_qwen3_mlp: FAIL\n");
        return 1;
    }

    std::printf("test_qwen3_mlp: PASS  (S=%u H=%u FFN=%u, worst abs=%.5f rel=%.4f)\n",
                kS, kH, kFFN, worst_abs, worst_rel);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_mlp: FAIL — %s\n", e.what());
    return 1;
}
