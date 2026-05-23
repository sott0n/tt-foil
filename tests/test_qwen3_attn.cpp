// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B layer-0 attention block on real weights.
//
//   x_norm = RMSNorm(x, ln1_gamma, eps)
//   Q = x_norm · W_q        → [S, num_q  * head_dim]
//   K = x_norm · W_k        → [S, num_kv * head_dim]
//   V = x_norm · W_v        → [S, num_kv * head_dim]
//   Q = RMSNorm(Q, q_norm)  per (token, head) over head_dim   ← Qwen3 q_norm
//   K = RMSNorm(K, k_norm)  per (token, head) over head_dim   ← Qwen3 k_norm
//   Q = RoPE(Q, cos, sin)   multi-head, split-half
//   K = RoPE(K, cos, sin)
//   for q_head in range(num_q):
//       kv_head = q_head // gqa_groups
//       Q_h, K_h, V_h = per-head slices
//       attn_h = MHA(Q_h, K_h^T, V_h, causal_mask)
//   attn = concat(attn_h)
//   proj = attn · W_o
//   y    = x + proj
//
// Compromises (Phase 1 correctness first, perf later):
//   - per-head extraction & K^T transpose are done on the host
//     (PCIe round-trips between RoPE outputs and per-head MHA inputs)
//   - the q_norm/k_norm reuse the standard RMSNorm kernel with
//     NCHt = St * num_heads and Wt = Dt (head_dim chunk = one row group)
//
// Data files are gitignored; regenerate with:
//   python3 models/qwen3_vl_2b/export_qwen3_layer.py --model Qwen/Qwen3-VL-2B-Instruct \
//       --layer 0 --out-dir data/qwen3_vl_2b
//   python3 models/qwen3_vl_2b/golden/qwen3_attn_golden.py --layer-dir data/qwen3_vl_2b/layer0 \
//       --num-q 16 --num-kv 8 --head-dim 128 --rope-theta 5000000.0 --seq 32

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

// Qwen3-VL-2B layer-0 geometry.
constexpr uint32_t kS        = 32;
constexpr uint32_t kH        = 2048;
constexpr uint32_t kNumQ     = 16;
constexpr uint32_t kNumKv    = 8;
constexpr uint32_t kHeadDim  = 128;
constexpr uint32_t kGqaGroups = kNumQ / kNumKv;

constexpr uint32_t kSt      = kS       / kTileH;   // 1
constexpr uint32_t kHt      = kH       / kTileW;   // 64
constexpr uint32_t kDt      = kHeadDim / kTileW;   // 4
constexpr uint32_t kDtHalf  = kDt / 2;             // 2
constexpr uint32_t kNqDt    = kNumQ  * kDt;        // 64
constexpr uint32_t kNkDt    = kNumKv * kDt;        // 32
constexpr float    kEps     = 1e-6f;
constexpr uint32_t kCosTiles = kSt * kDtHalf;       // [S, head_dim/2] tiles

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

// gamma [D] broadcast to [32, D] then tiled — matches the RMSNorm kernel's
// expectation that gamma is replicated across all 32 token rows of each tile.
std::vector<uint16_t> gamma_to_tiles(const std::vector<uint16_t>& gamma, uint32_t D) {
    std::vector<uint16_t> rm(kTileH * D);
    for (uint32_t r = 0; r < kTileH; ++r)
        for (uint32_t c = 0; c < D; ++c) rm[r * D + c] = gamma[c];
    return tile2d(rm, kTileH, D);
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    const std::string base = data_env ? data_env : "data/qwen3_vl_2b/layer0";

    // -----------------------------------------------------------------
    // 1. Load weights and tile them.
    // -----------------------------------------------------------------
    auto x_rm        = load_bin(base + "/attn_input.bin",  kS * kH);
    auto golden_rm   = load_bin(base + "/attn_golden.bin", kS * kH);
    auto ln1g_flat   = load_bin(base + "/ln1_gamma.bin",   kH);
    auto qng_flat    = load_bin(base + "/q_norm.bin",      kHeadDim);
    auto kng_flat    = load_bin(base + "/k_norm.bin",      kHeadDim);
    auto cos_rm      = load_bin(base + "/cos_table.bin",   kS * kHeadDim / 2);
    auto sin_rm      = load_bin(base + "/sin_table.bin",   kS * kHeadDim / 2);
    auto Wq_rm       = load_bin(base + "/W_q.bin",         kH * kNumQ  * kHeadDim);
    auto Wk_rm       = load_bin(base + "/W_k.bin",         kH * kNumKv * kHeadDim);
    auto Wv_rm       = load_bin(base + "/W_v.bin",         kH * kNumKv * kHeadDim);
    auto Wo_rm       = load_bin(base + "/W_o.bin",         kNumQ * kHeadDim * kH);

    auto x_tiles     = tile2d(x_rm,    kS, kH);
    auto cos_tiles   = tile2d(cos_rm,  kS, kHeadDim / 2);
    auto sin_tiles   = tile2d(sin_rm,  kS, kHeadDim / 2);
    auto Wq_tiles    = tile2d(Wq_rm,   kH, kNumQ  * kHeadDim);
    auto Wk_tiles    = tile2d(Wk_rm,   kH, kNumKv * kHeadDim);
    auto Wv_tiles    = tile2d(Wv_rm,   kH, kNumKv * kHeadDim);
    auto Wo_tiles    = tile2d(Wo_rm,   kNumQ * kHeadDim, kH);
    auto ln1g_tiles  = gamma_to_tiles(ln1g_flat, kH);
    auto qng_tiles   = gamma_to_tiles(qng_flat,  kHeadDim);
    auto kng_tiles   = gamma_to_tiles(kng_flat,  kHeadDim);

    // Causal mask [S, S] (BF16 0/1, multiplicative — see ops/mha/).
    std::vector<uint16_t> mask_rm(kS * kS, 0);
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;
    auto mask_tiles = tile2d(mask_rm, kS, kS);

    // -----------------------------------------------------------------
    // 2. Open device, allocate persistent tensors.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

    auto T_x      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ln1g   = ol::allocate_tensor_dram(*dev, kHt);
    auto T_qng    = ol::allocate_tensor_dram(*dev, kDt);
    auto T_kng    = ol::allocate_tensor_dram(*dev, kDt);
    auto T_cos    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_sin    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_Wq     = ol::allocate_tensor_dram(*dev, kHt * kNqDt);
    auto T_Wk     = ol::allocate_tensor_dram(*dev, kHt * kNkDt);
    auto T_Wv     = ol::allocate_tensor_dram(*dev, kHt * kNkDt);
    auto T_Wo     = ol::allocate_tensor_dram(*dev, kNqDt * kHt);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);

    // Intermediate tensors.
    auto T_xnorm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Q      = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qn     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn   = ol::allocate_tensor_dram(*dev, kSt * kNqDt);   // host-assembled
    auto T_proj   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_out    = ol::allocate_tensor_dram(*dev, kSt * kHt);

    // Per-head tensors (allocated once, reused 16 times).
    auto T_Qh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_KhT    = ol::allocate_tensor_dram(*dev, kDt * kSt);
    auto T_Vh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_AttnH  = ol::allocate_tensor_dram(*dev, kSt * kDt);

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    upload(T_x,    x_tiles);
    upload(T_ln1g, ln1g_tiles);
    upload(T_qng,  qng_tiles);
    upload(T_kng,  kng_tiles);
    upload(T_cos,  cos_tiles);
    upload(T_sin,  sin_tiles);
    upload(T_Wq,   Wq_tiles);
    upload(T_Wk,   Wk_tiles);
    upload(T_Wv,   Wv_tiles);
    upload(T_Wo,   Wo_tiles);
    upload(T_mask, mask_tiles);

    auto run = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };

    // -----------------------------------------------------------------
    // 3. Pre-attention: ln1, QKV projections.
    // -----------------------------------------------------------------
    run([&] { return ol::make_rmsnorm(*dev, T_x, T_ln1g, T_xnorm, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    run([&] { return ol::make_matmul(*dev, T_xnorm, T_Wq, T_Q, kSt, kHt, kNqDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_xnorm, T_Wk, T_K, kSt, kHt, kNkDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_xnorm, T_Wv, T_V, kSt, kHt, kNkDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 4. q_norm / k_norm: standard RMSNorm with NCHt = St*num_heads, Wt = Dt.
    // -----------------------------------------------------------------
    run([&] {
        return ol::make_rmsnorm(*dev, T_Q, T_qng, T_Qn,
                                kSt * kNumQ, kDt, kEps);
    });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] {
        return ol::make_rmsnorm(*dev, T_K, T_kng, T_Kn,
                                kSt * kNumKv, kDt, kEps);
    });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 5. RoPE on Q and K (multi-head, in-place into T_Qr / T_Kr).
    // -----------------------------------------------------------------
    run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 6. Per-head GQA loop on host: extract Q_h / K_h / V_h, transpose K,
    //    scale Q by 1/sqrt(head_dim), run MHA, assemble attn.
    // -----------------------------------------------------------------
    std::vector<uint16_t> Qr_all(kSt * kNqDt * kTileWords);
    std::vector<uint16_t> Kr_all(kSt * kNkDt * kTileWords);
    std::vector<uint16_t> V_all (kSt * kNkDt * kTileWords);
    tt::foil::read_buffer(*dev, *T_Qr.buf, Qr_all.data(), Qr_all.size() * 2);
    tt::foil::read_buffer(*dev, *T_Kr.buf, Kr_all.data(), Kr_all.size() * 2);
    tt::foil::read_buffer(*dev, *T_V.buf,  V_all.data(),  V_all.size()  * 2);

    // Reconstruct row-major Q_rope / K_rope / V for easy slicing.
    auto Qr_rm = untile2d(Qr_all, kS, kNumQ  * kHeadDim);
    auto Kr_rm = untile2d(Kr_all, kS, kNumKv * kHeadDim);
    auto V_rm  = untile2d(V_all,  kS, kNumKv * kHeadDim);

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<uint16_t> attn_concat_rm(kS * kNumQ * kHeadDim, 0);

    for (uint32_t h = 0; h < kNumQ; ++h) {
        uint32_t kv = h / kGqaGroups;

        // Q_h: scale by 1/sqrt(d). K^T (transpose). V_h.
        std::vector<uint16_t> Qh_rm(kS * kHeadDim);
        std::vector<uint16_t> KhT_rm(kHeadDim * kS);
        std::vector<uint16_t> Vh_rm(kS * kHeadDim);
        for (uint32_t s = 0; s < kS; ++s) {
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                float q_val = bf16_to_f32(Qr_rm[s * kNumQ  * kHeadDim + h  * kHeadDim + d]);
                Qh_rm[s * kHeadDim + d] = f32_to_bf16(q_val * inv_sqrt_d);
                KhT_rm[d * kS + s]      = Kr_rm[s * kNumKv * kHeadDim + kv * kHeadDim + d];
                Vh_rm[s * kHeadDim + d] = V_rm [s * kNumKv * kHeadDim + kv * kHeadDim + d];
            }
        }
        auto Qh_tiles  = tile2d(Qh_rm,  kS, kHeadDim);
        auto KhT_tiles = tile2d(KhT_rm, kHeadDim, kS);
        auto Vh_tiles  = tile2d(Vh_rm,  kS, kHeadDim);
        upload(T_Qh,  Qh_tiles);
        upload(T_KhT, KhT_tiles);
        upload(T_Vh,  Vh_tiles);

        run([&] {
            return ol::make_mha(*dev, T_Qh, T_KhT, T_Vh, T_mask, T_AttnH, kSt, kDt);
        });
        tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

        std::vector<uint16_t> attn_h_tiles(kSt * kDt * kTileWords);
        tt::foil::read_buffer(*dev, *T_AttnH.buf, attn_h_tiles.data(), attn_h_tiles.size() * 2);
        auto attn_h_rm = untile2d(attn_h_tiles, kS, kHeadDim);
        for (uint32_t s = 0; s < kS; ++s)
            for (uint32_t d = 0; d < kHeadDim; ++d)
                attn_concat_rm[s * kNumQ * kHeadDim + h * kHeadDim + d] =
                    attn_h_rm[s * kHeadDim + d];
    }

    // Upload assembled attn and run W_o + residual.
    auto attn_tiles = tile2d(attn_concat_rm, kS, kNumQ * kHeadDim);
    upload(T_attn, attn_tiles);

    run([&] { return ol::make_matmul(*dev, T_attn, T_Wo, T_proj, kSt, kNqDt, kHt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_eltwise_add(*dev, T_x, T_proj, T_out); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 7. Compare.
    // -----------------------------------------------------------------
    std::vector<uint16_t> out_tiles(kSt * kHt * kTileWords);
    tt::foil::read_buffer(*dev, *T_out.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, kH);

    // Long compound chain (RMSNorm + 3×matmul + 2×RMSNorm + 2×RoPE + 16×MHA
    // + matmul + add) on BF16 with real weights — tolerances follow the
    // qwen3_mlp test envelope.
    const float kAbsTol = 0.20f;
    const float kRelTol = 0.08f;
    uint32_t bad = 0; float worst_abs = 0.0f, worst_rel = 0.0f;
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

    tt::foil::close_device(std::move(dev));

    if (bad != 0) {
        std::fprintf(stderr,
            "test_qwen3_attn: %u/%u mismatches; first at i=%u (row=%u col=%u): "
            "got=%.4f exp=%.4f; worst abs=%.5f rel=%.4f (tol abs=%.5f rel=%.4f)\n",
            bad, kS * kH, first_bad, first_bad / kH, first_bad % kH,
            bf16_to_f32(out_rm[first_bad]), bf16_to_f32(golden_rm[first_bad]),
            worst_abs, worst_rel, kAbsTol, kRelTol);
        return 1;
    }
    std::printf("test_qwen3_attn: PASS  (S=%u H=%u num_q=%u num_kv=%u head_dim=%u, "
                "worst abs=%.5f rel=%.4f)\n",
                kS, kH, kNumQ, kNumKv, kHeadDim, worst_abs, worst_rel);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_attn: FAIL — %s\n", e.what());
    return 1;
}
