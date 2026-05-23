// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B layer-0 full Transformer layer (attention + MLP)
// on real HuggingFace weights.
//
//   --- Attention block ---
//   x_norm1 = RMSNorm(x, ln1_gamma)
//   Q = x_norm1 · W_q;   K = x_norm1 · W_k;   V = x_norm1 · W_v
//   Q = RMSNorm(Q reshaped per-head, q_norm)
//   K = RMSNorm(K reshaped per-head, k_norm)
//   Q = RoPE(Q, cos, sin);   K = RoPE(K, cos, sin)
//   per-head host loop → attn_h via MHA; concat
//   proj = attn · W_o
//   x_mid = x + proj                              (residual #1)
//
//   --- MLP block ---
//   y_norm = RMSNorm(x_mid, ln2_gamma)
//   gate = y_norm · W_gate;   up = y_norm · W_up
//   fused = SiLU(gate) ⊙ up
//   down = fused · W_down
//   y    = x_mid + down                           (residual #2)
//
// Same compromises as test_qwen3_attn (per-head extraction + K^T on
// host, q_norm/k_norm reuse RMSNorm with NCHt=St*num_heads, Wt=Dt).
//
// Regenerate inputs:
//   python3 models/qwen3_vl_2b/export_qwen3_layer.py --model Qwen/Qwen3-VL-2B-Instruct \
//       --layer 0 --out-dir data/qwen3_vl_2b
//   python3 models/qwen3_vl_2b/golden/qwen3_layer_golden.py --layer-dir data/qwen3_vl_2b/layer0 \
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
constexpr uint32_t kS         = 32;
constexpr uint32_t kH         = 2048;
constexpr uint32_t kFFN       = 6144;
constexpr uint32_t kNumQ      = 16;
constexpr uint32_t kNumKv     = 8;
constexpr uint32_t kHeadDim   = 128;
constexpr uint32_t kGqaGroups = kNumQ / kNumKv;

constexpr uint32_t kSt      = kS       / kTileH;   // 1
constexpr uint32_t kHt      = kH       / kTileW;   // 64
constexpr uint32_t kFFt     = kFFN     / kTileW;   // 192
constexpr uint32_t kDt      = kHeadDim / kTileW;   // 4
constexpr uint32_t kDtHalf  = kDt / 2;             // 2
constexpr uint32_t kNqDt    = kNumQ  * kDt;        // 64
constexpr uint32_t kNkDt    = kNumKv * kDt;        // 32
constexpr float    kEps     = 1e-6f;
constexpr uint32_t kCosTiles = kSt * kDtHalf;

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
    // 1. Load.
    // -----------------------------------------------------------------
    auto x_rm        = load_bin(base + "/layer_input.bin",  kS * kH);
    auto golden_rm   = load_bin(base + "/layer_golden.bin", kS * kH);
    auto ln1g_flat   = load_bin(base + "/ln1_gamma.bin",    kH);
    auto ln2g_flat   = load_bin(base + "/ln2_gamma.bin",    kH);
    auto qng_flat    = load_bin(base + "/q_norm.bin",       kHeadDim);
    auto kng_flat    = load_bin(base + "/k_norm.bin",       kHeadDim);
    auto cos_rm      = load_bin(base + "/cos_table.bin",    kS * kHeadDim / 2);
    auto sin_rm      = load_bin(base + "/sin_table.bin",    kS * kHeadDim / 2);
    auto Wq_rm       = load_bin(base + "/W_q.bin",          kH * kNumQ  * kHeadDim);
    auto Wk_rm       = load_bin(base + "/W_k.bin",          kH * kNumKv * kHeadDim);
    auto Wv_rm       = load_bin(base + "/W_v.bin",          kH * kNumKv * kHeadDim);
    auto Wo_rm       = load_bin(base + "/W_o.bin",          kNumQ * kHeadDim * kH);
    auto Wgate_rm    = load_bin(base + "/W_gate.bin",       kH * kFFN);
    auto Wup_rm      = load_bin(base + "/W_up.bin",         kH * kFFN);
    auto Wdown_rm    = load_bin(base + "/W_down.bin",       kFFN * kH);

    auto x_tiles     = tile2d(x_rm,    kS, kH);
    auto cos_tiles   = tile2d(cos_rm,  kS, kHeadDim / 2);
    auto sin_tiles   = tile2d(sin_rm,  kS, kHeadDim / 2);
    auto Wq_tiles    = tile2d(Wq_rm,   kH, kNumQ  * kHeadDim);
    auto Wk_tiles    = tile2d(Wk_rm,   kH, kNumKv * kHeadDim);
    auto Wv_tiles    = tile2d(Wv_rm,   kH, kNumKv * kHeadDim);
    auto Wo_tiles    = tile2d(Wo_rm,   kNumQ * kHeadDim, kH);
    auto Wgate_tiles = tile2d(Wgate_rm, kH,   kFFN);
    auto Wup_tiles   = tile2d(Wup_rm,   kH,   kFFN);
    auto Wdown_tiles = tile2d(Wdown_rm, kFFN, kH);
    auto ln1g_tiles  = gamma_to_tiles(ln1g_flat, kH);
    auto ln2g_tiles  = gamma_to_tiles(ln2g_flat, kH);
    auto qng_tiles   = gamma_to_tiles(qng_flat,  kHeadDim);
    auto kng_tiles   = gamma_to_tiles(kng_flat,  kHeadDim);

    std::vector<uint16_t> mask_rm(kS * kS, 0);
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;
    auto mask_tiles = tile2d(mask_rm, kS, kS);

    // -----------------------------------------------------------------
    // 2. Device setup.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

    // Persistent.
    auto T_x      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ln1g   = ol::allocate_tensor_dram(*dev, kHt);
    auto T_ln2g   = ol::allocate_tensor_dram(*dev, kHt);
    auto T_qng    = ol::allocate_tensor_dram(*dev, kDt);
    auto T_kng    = ol::allocate_tensor_dram(*dev, kDt);
    auto T_cos    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_sin    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_Wq     = ol::allocate_tensor_dram(*dev, kHt * kNqDt);
    auto T_Wk     = ol::allocate_tensor_dram(*dev, kHt * kNkDt);
    auto T_Wv     = ol::allocate_tensor_dram(*dev, kHt * kNkDt);
    auto T_Wo     = ol::allocate_tensor_dram(*dev, kNqDt * kHt);
    auto T_Wgate  = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wup    = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wdown  = ol::allocate_tensor_dram(*dev, kFFt * kHt);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);

    // Intermediates (attention).
    auto T_xnorm1 = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Q      = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qn     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn   = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_proj   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid   = ol::allocate_tensor_dram(*dev, kSt * kHt);

    // Intermediates (MLP).
    auto T_ynorm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up     = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_out    = ol::allocate_tensor_dram(*dev, kSt * kHt);

    // Per-head MHA buffers.
    auto T_Qh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_KhT    = ol::allocate_tensor_dram(*dev, kDt * kSt);
    auto T_Vh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_AttnH  = ol::allocate_tensor_dram(*dev, kSt * kDt);

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    upload(T_x,     x_tiles);
    upload(T_ln1g,  ln1g_tiles);
    upload(T_ln2g,  ln2g_tiles);
    upload(T_qng,   qng_tiles);
    upload(T_kng,   kng_tiles);
    upload(T_cos,   cos_tiles);
    upload(T_sin,   sin_tiles);
    upload(T_Wq,    Wq_tiles);
    upload(T_Wk,    Wk_tiles);
    upload(T_Wv,    Wv_tiles);
    upload(T_Wo,    Wo_tiles);
    upload(T_Wgate, Wgate_tiles);
    upload(T_Wup,   Wup_tiles);
    upload(T_Wdown, Wdown_tiles);
    upload(T_mask,  mask_tiles);

    auto run = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };

    // -----------------------------------------------------------------
    // 3. Attention block.
    // -----------------------------------------------------------------
    run([&] { return ol::make_rmsnorm(*dev, T_x, T_ln1g, T_xnorm1, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_xnorm1, T_Wq, T_Q, kSt, kHt, kNqDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_xnorm1, T_Wk, T_K, kSt, kHt, kNkDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_xnorm1, T_Wv, T_V, kSt, kHt, kNkDt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_rmsnorm(*dev, T_Q, T_qng, T_Qn, kSt * kNumQ,  kDt, kEps); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_rmsnorm(*dev, T_K, T_kng, T_Kn, kSt * kNumKv, kDt, kEps); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // Per-head GQA loop on host (matches test_qwen3_attn).
    std::vector<uint16_t> Qr_all(kSt * kNqDt * kTileWords);
    std::vector<uint16_t> Kr_all(kSt * kNkDt * kTileWords);
    std::vector<uint16_t> V_all (kSt * kNkDt * kTileWords);
    tt::foil::read_buffer(*dev, *T_Qr.buf, Qr_all.data(), Qr_all.size() * 2);
    tt::foil::read_buffer(*dev, *T_Kr.buf, Kr_all.data(), Kr_all.size() * 2);
    tt::foil::read_buffer(*dev, *T_V.buf,  V_all.data(),  V_all.size()  * 2);
    auto Qr_rm = untile2d(Qr_all, kS, kNumQ  * kHeadDim);
    auto Kr_rm = untile2d(Kr_all, kS, kNumKv * kHeadDim);
    auto V_rm  = untile2d(V_all,  kS, kNumKv * kHeadDim);

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<uint16_t> attn_concat_rm(kS * kNumQ * kHeadDim, 0);
    for (uint32_t h = 0; h < kNumQ; ++h) {
        uint32_t kv = h / kGqaGroups;
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
        upload(T_Qh,  tile2d(Qh_rm,  kS, kHeadDim));
        upload(T_KhT, tile2d(KhT_rm, kHeadDim, kS));
        upload(T_Vh,  tile2d(Vh_rm,  kS, kHeadDim));
        run([&] {
            return ol::make_mha(*dev, T_Qh, T_KhT, T_Vh, T_mask, T_AttnH, kSt, kDt);
        });
        tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
        std::vector<uint16_t> ah_tiles(kSt * kDt * kTileWords);
        tt::foil::read_buffer(*dev, *T_AttnH.buf, ah_tiles.data(), ah_tiles.size() * 2);
        auto ah_rm = untile2d(ah_tiles, kS, kHeadDim);
        for (uint32_t s = 0; s < kS; ++s)
            for (uint32_t d = 0; d < kHeadDim; ++d)
                attn_concat_rm[s * kNumQ * kHeadDim + h * kHeadDim + d] = ah_rm[s * kHeadDim + d];
    }
    upload(T_attn, tile2d(attn_concat_rm, kS, kNumQ * kHeadDim));
    run([&] { return ol::make_matmul(*dev, T_attn, T_Wo, T_proj, kSt, kNqDt, kHt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_eltwise_add(*dev, T_x, T_proj, T_xmid); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 4. MLP block.
    // -----------------------------------------------------------------
    run([&] { return ol::make_rmsnorm(*dev, T_xmid, T_ln2g, T_ynorm, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_ynorm, T_Wgate, T_gate, kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_ynorm, T_Wup,   T_up,   kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_silu(*dev, T_gate, T_silu); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_matmul(*dev, T_fused, T_Wdown, T_down, kSt, kFFt, kHt); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);
    run([&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_out); });
    tt::foil::release_kernels(*dev, core);
    tt::foil::reset_l1(*dev, core);

    // -----------------------------------------------------------------
    // 5. Compare.
    // -----------------------------------------------------------------
    std::vector<uint16_t> out_tiles(kSt * kHt * kTileWords);
    tt::foil::read_buffer(*dev, *T_out.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, kH);

    // Tolerance: extending past attention (worst 0.031 in test_qwen3_attn)
    // through one more RMSNorm and a 3-matmul + SFPU MLP — drift grows but
    // stays comfortably within these bounds for BF16 real weights.
    const float kAbsTol = 0.30f;
    const float kRelTol = 0.10f;
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
            "test_qwen3_layer: %u/%u mismatches; first at i=%u (row=%u col=%u): "
            "got=%.4f exp=%.4f; worst abs=%.5f rel=%.4f (tol abs=%.5f rel=%.4f)\n",
            bad, kS * kH, first_bad, first_bad / kH, first_bad % kH,
            bf16_to_f32(out_rm[first_bad]), bf16_to_f32(golden_rm[first_bad]),
            worst_abs, worst_rel, kAbsTol, kRelTol);
        return 1;
    }
    std::printf("test_qwen3_layer: PASS  (S=%u H=%u FFN=%u num_q=%u num_kv=%u head_dim=%u, "
                "worst abs=%.5f rel=%.4f)\n",
                kS, kH, kFFN, kNumQ, kNumKv, kHeadDim, worst_abs, worst_rel);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_layer: FAIL — %s\n", e.what());
    return 1;
}
