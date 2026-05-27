// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B lm_head matmul on real (tied) embed weights.
//
//   logits = hidden @ lm_head_W      [S=32, V=151936]
//   lm_head_W = embed_tokens.T       [H=2048, V=151936]
//
// Pre-transposed and pre-tiled weight buffer (622 MB on disk) is produced
// by models/qwen3_vl_2b/golden/qwen3_lm_head_golden.py — that script also generates the
// hidden-state input and the golden logits, and dumps top-1 argmax per row.
//
// Verifies:
//   • the matmul kernel handles Nt=4748 (much larger than any other op
//     in tt-foil so far) without runtime/L1 issues
//   • numerical match with the numpy reference under bf16 tolerances
//   • top-1 argmax per sequence row exactly matches the host golden
//
// Regenerate inputs:
//   python3 models/qwen3_vl_2b/export_qwen3_layer.py --layer none --model-tensors \
//       --out-dir data/qwen3_vl_2b
//   python3 models/qwen3_vl_2b/golden/qwen3_lm_head_golden.py --data-dir data/qwen3_vl_2b \
//       --hidden 2048 --vocab 151936 --seq 32

#include <algorithm>
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
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kS  = 32;
constexpr uint32_t kH  = 2048;
constexpr uint32_t kV  = 151936;
constexpr uint32_t kSt = kS / kTileH;   // 1
constexpr uint32_t kKt = kH / kTileW;   // 64
constexpr uint32_t kNt = kV / kTileW;   // 4748

std::vector<uint16_t> load_bin(const std::string& path, std::size_t nelem) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + path);
    std::vector<uint16_t> v(nelem);
    f.read(reinterpret_cast<char*>(v.data()), nelem * 2);
    if (f.gcount() != static_cast<std::streamsize>(nelem * 2))
        throw std::runtime_error("short read: " + path);
    return v;
}

std::vector<uint16_t> tile2d(const std::vector<uint16_t>& rm,
                             uint32_t Rows, uint32_t Cols) {
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

std::vector<uint16_t> untile2d(const std::vector<uint16_t>& tiles,
                               uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> rm(static_cast<size_t>(Rows) * Cols, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    size_t idx = 0;
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

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    const std::string base = data_env ? data_env : "data/qwen3_vl_2b";
    const std::string mdir = base + "/model";

    // -----------------------------------------------------------------
    // 1. Load pre-tiled lm_head_W, hidden input, golden logits.
    // -----------------------------------------------------------------
    std::printf("loading lm_head_tiled.bin (%.0f MB)...\n",
                static_cast<double>(kKt) * kNt * kTileBytes / 1e6);
    auto lmhead_tiles = load_bin(mdir + "/lm_head_tiled.bin",
                                 static_cast<size_t>(kKt) * kNt * kTileWords);
    auto x_rm        = load_bin(mdir + "/lm_head_input.bin",  kS * kH);
    auto golden_rm   = load_bin(mdir + "/lm_head_golden.bin",
                                static_cast<size_t>(kS) * kV);

    auto x_tiles = tile2d(x_rm, kS, kH);

    // -----------------------------------------------------------------
    // 2. Upload to DRAM and run matmul.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto T_x = ol::allocate_tensor_dram(*dev, kSt * kKt);
    auto T_W = ol::allocate_tensor_dram(*dev, kKt * kNt);
    auto T_y = ol::allocate_tensor_dram(*dev, kSt * kNt);

    tt::foil::write_buffer(*dev, *T_x.buf, x_tiles.data(),      x_tiles.size()      * 2);
    tt::foil::write_buffer(*dev, *T_W.buf, lmhead_tiles.data(), lmhead_tiles.size() * 2);

    std::printf("running matmul (Mt=%u, Kt=%u, Nt=%u)...\n", kSt, kKt, kNt);
    {
        auto op = ol::make_matmul(*dev, T_x, T_W, T_y, kSt, kKt, kNt);
        ol::execute(*dev, op);
    }

    // -----------------------------------------------------------------
    // 3. Read back & compare.
    // -----------------------------------------------------------------
    std::vector<uint16_t> y_tiles(static_cast<size_t>(kSt) * kNt * kTileWords);
    tt::foil::read_buffer(*dev, *T_y.buf, y_tiles.data(), y_tiles.size() * 2);
    auto y_rm = untile2d(y_tiles, kS, kV);

    // Per-element numeric check.
    const float kAbsTol = 0.50f;
    const float kRelTol = 0.05f;
    uint32_t bad = 0; float worst_abs = 0.0f, worst_rel = 0.0f;
    size_t first_bad = static_cast<size_t>(kS) * kV;
    for (size_t i = 0; i < static_cast<size_t>(kS) * kV; ++i) {
        float got = bf16_to_f32(y_rm[i]);
        float exp = bf16_to_f32(golden_rm[i]);
        float d_abs = std::fabs(got - exp);
        float d_rel = d_abs / std::max(std::fabs(exp), 1e-3f);
        if (d_abs > worst_abs) worst_abs = d_abs;
        if (d_rel > worst_rel) worst_rel = d_rel;
        if (d_abs > kAbsTol && d_rel > kRelTol) {
            if (first_bad == static_cast<size_t>(kS) * kV) first_bad = i;
            ++bad;
        }
    }

    // Per-row top-1 argmax check (what actually matters for token selection).
    uint32_t argmax_mismatch = 0;
    std::vector<uint32_t> got_top1(kS, 0);
    std::vector<uint32_t> ref_top1(kS, 0);
    for (uint32_t s = 0; s < kS; ++s) {
        float gbest = -1e30f, rbest = -1e30f;
        uint32_t gi = 0, ri = 0;
        for (uint32_t v = 0; v < kV; ++v) {
            float gv = bf16_to_f32(y_rm[static_cast<size_t>(s) * kV + v]);
            float rv = bf16_to_f32(golden_rm[static_cast<size_t>(s) * kV + v]);
            if (gv > gbest) { gbest = gv; gi = v; }
            if (rv > rbest) { rbest = rv; ri = v; }
        }
        got_top1[s] = gi;
        ref_top1[s] = ri;
        if (gi != ri) ++argmax_mismatch;
    }

    tt::foil::close_device(std::move(dev));

    std::printf("first 8 top-1 (got / ref): ");
    for (uint32_t s = 0; s < 8; ++s)
        std::printf("[%u / %u] ", got_top1[s], ref_top1[s]);
    std::printf("\n");

    if (bad != 0 || argmax_mismatch != 0) {
        std::fprintf(stderr,
            "test_qwen3_lm_head: %u/%zu logit cells off; "
            "%u/%u rows with wrong top-1 argmax; "
            "worst abs=%.4f rel=%.4f (tol abs=%.2f rel=%.4f)\n",
            bad, static_cast<size_t>(kS) * kV,
            argmax_mismatch, kS,
            worst_abs, worst_rel, kAbsTol, kRelTol);
        if (bad != 0) {
            const uint32_t row = first_bad / kV;
            const uint32_t col = first_bad % kV;
            std::fprintf(stderr,
                "  first bad: row=%u col=%u: got=%.4f exp=%.4f\n",
                row, col,
                bf16_to_f32(y_rm[first_bad]), bf16_to_f32(golden_rm[first_bad]));
        }
        return 1;
    }
    std::printf("test_qwen3_lm_head: PASS  (S=%u H=%u V=%u Nt=%u, "
                "worst abs=%.4f rel=%.4f, top-1 matches all %u rows)\n",
                kS, kH, kV, kNt, worst_abs, worst_rel, kS);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_lm_head: FAIL — %s\n", e.what());
    return 1;
}
