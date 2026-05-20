// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: a Qwen3-style Transformer block end-to-end on one Tensix core.
//
// Composition (single head, no RoPE, no KV cache):
//
//   x_norm1 = RMSNorm(x, ln1_gamma, eps)
//   Q = x_norm1 · W_q_scaled        (W_q already scaled by 1/sqrt(D) on host)
//   K = x_norm1 · W_k
//   V = x_norm1 · W_v
//   attn_out = MHA(Q, K^T, V, causal_mask)
//   proj = attn_out · W_o
//   x_mid = x + proj                 (residual #1)
//
//   y_norm = RMSNorm(x_mid, ln2_gamma, eps)
//   gate = y_norm · W_gate
//   up   = y_norm · W_up
//   mlp_fused = SiLU(gate) ⊙ up
//   down = mlp_fused · W_down
//   y_out = x_mid + down             (residual #2)
//
// Compromise: K is transposed on the host between K-projection and MHA
// because we don't yet have a tile-transpose op or a matmul-with-transpose-b
// kernel.  Follow-up: fuse the transpose into the kernel.
//
// Each op_lib invocation is scoped so its shared_ptr<Kernel> drops before
// we move on, and tt::foil::release_kernels reclaims the per-core
// KERNEL_CONFIG region between ops (it fills very fast otherwise).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Block dimensions (all multiples of 32). Kept small to fit comfortably in
// L1 across a 13-op chain.
constexpr uint32_t kS    = 64;
constexpr uint32_t kH    = 64;
constexpr uint32_t kFFN  = 128;
constexpr uint32_t kSt   = kS / kTileH;     // = 2
constexpr uint32_t kHt   = kH / kTileW;     // = 2
constexpr uint32_t kFFt  = kFFN / kTileW;   // = 4
constexpr float    kEps  = 1e-5f;

std::string required_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) throw std::runtime_error(std::string("Missing env var: ") + name);
    return v;
}

// Deterministic [-a, +a] value.
float rng(uint32_t seed, float amp) {
    uint32_t h = seed * 1664525u + 1013904223u;
    return amp * (-1.0f + 2.0f * (h % 1024u) / 1023.0f);
}

// Convert a row-major [Rows, Cols] BF16 block → tile-format stream.
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

// Float reference helpers.
std::vector<float> mat_mul(const std::vector<float>& A, const std::vector<float>& B,
                           uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> C(M * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
    return C;
}
std::vector<float> rmsnorm_ref(const std::vector<float>& x, const std::vector<float>& gamma,
                               uint32_t Rows, uint32_t Cols, float eps) {
    std::vector<float> y(Rows * Cols);
    for (uint32_t r = 0; r < Rows; ++r) {
        float ss = 0.0f;
        for (uint32_t c = 0; c < Cols; ++c) ss += x[r * Cols + c] * x[r * Cols + c];
        float inv = 1.0f / std::sqrt(ss / Cols + eps);
        for (uint32_t c = 0; c < Cols; ++c) y[r * Cols + c] = x[r * Cols + c] * inv * gamma[c];
    }
    return y;
}
std::vector<float> mha_ref(const std::vector<float>& Q, const std::vector<float>& K,
                           const std::vector<float>& V, uint32_t S, uint32_t D) {
    // Q already scaled by 1/sqrt(D)
    std::vector<float> scores(S * S, 0.0f);
    for (uint32_t i = 0; i < S; ++i)
        for (uint32_t j = 0; j < S; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < D; ++k) s += Q[i * D + k] * K[j * D + k];
            scores[i * S + j] = (j <= i) ? s : -1e30f;
        }
    std::vector<float> out(S * D, 0.0f);
    for (uint32_t i = 0; i < S; ++i) {
        double sum = 0.0;
        std::vector<float> e(S);
        for (uint32_t j = 0; j <= i; ++j) { e[j] = std::exp(scores[i * S + j]); sum += e[j]; }
        for (uint32_t k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (uint32_t j = 0; j <= i; ++j)
                acc += (e[j] / static_cast<float>(sum)) * V[j * D + k];
            out[i * D + k] = acc;
        }
    }
    return out;
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // -----------------------------------------------------------------
    // 1. Dummy weights / inputs (F32) — deterministic, small amplitudes
    //    so BF16 quantization stays comfortable through the chain.
    // -----------------------------------------------------------------
    std::vector<float> x(kS * kH);
    for (uint32_t i = 0; i < kS * kH; ++i) x[i] = rng(i, 0.5f);

    auto mk_w = [](uint32_t rows, uint32_t cols, uint32_t seed) {
        std::vector<float> w(rows * cols);
        for (uint32_t i = 0; i < rows * cols; ++i) w[i] = rng(seed + i, 0.1f);
        return w;
    };
    auto ln1_gamma = mk_w(1, kH,    1);   // [H]
    auto ln2_gamma = mk_w(1, kH,    2);
    auto W_q       = mk_w(kH, kH,   100);
    auto W_k       = mk_w(kH, kH,   200);
    auto W_v       = mk_w(kH, kH,   300);
    auto W_o       = mk_w(kH, kH,   400);
    auto W_gate    = mk_w(kH, kFFN, 500);
    auto W_up      = mk_w(kH, kFFN, 600);
    auto W_down    = mk_w(kFFN, kH, 700);

    // Pre-scale W_q so Q comes out already divided by sqrt(D).
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kH));
    std::vector<float> W_q_scaled = W_q;
    for (auto& v : W_q_scaled) v *= inv_sqrt_d;

    // ln_gamma broadcasted across all S rows so the tile-format input to
    // RMSNorm matches what the kernel expects (gamma replicated to 32 rows).
    auto bcast_gamma = [](const std::vector<float>& g, uint32_t Rows, uint32_t Cols) {
        std::vector<float> out(Rows * Cols);
        for (uint32_t r = 0; r < Rows; ++r)
            for (uint32_t c = 0; c < Cols; ++c) out[r * Cols + c] = g[c];
        return out;
    };
    auto ln1_gamma_2d = bcast_gamma(ln1_gamma, kS, kH);
    auto ln2_gamma_2d = bcast_gamma(ln2_gamma, kS, kH);

    // -----------------------------------------------------------------
    // 2. Host reference forward pass (F32).
    // -----------------------------------------------------------------
    auto x_norm1_f = rmsnorm_ref(x, ln1_gamma, kS, kH, kEps);
    auto Q_f       = mat_mul(x_norm1_f, W_q_scaled, kS, kH, kH);
    auto K_f       = mat_mul(x_norm1_f, W_k,        kS, kH, kH);
    auto V_f       = mat_mul(x_norm1_f, W_v,        kS, kH, kH);
    auto attn_f    = mha_ref(Q_f, K_f, V_f, kS, kH);
    auto proj_f    = mat_mul(attn_f, W_o, kS, kH, kH);
    std::vector<float> x_mid_f(kS * kH);
    for (uint32_t i = 0; i < kS * kH; ++i) x_mid_f[i] = x[i] + proj_f[i];

    auto y_norm_f  = rmsnorm_ref(x_mid_f, ln2_gamma, kS, kH, kEps);
    auto gate_f    = mat_mul(y_norm_f, W_gate, kS, kH, kFFN);
    auto up_f      = mat_mul(y_norm_f, W_up,   kS, kH, kFFN);
    std::vector<float> fused_f(kS * kFFN);
    for (uint32_t i = 0; i < kS * kFFN; ++i) {
        float g = gate_f[i];
        float s = g / (1.0f + std::exp(-g));  // SiLU
        fused_f[i] = s * up_f[i];
    }
    auto down_f = mat_mul(fused_f, W_down, kS, kFFN, kH);
    std::vector<float> ref_f(kS * kH);
    for (uint32_t i = 0; i < kS * kH; ++i) ref_f[i] = x_mid_f[i] + down_f[i];

    // -----------------------------------------------------------------
    // 3. BF16 tile-format payloads.
    // -----------------------------------------------------------------
    auto to_bf16 = [](const std::vector<float>& f) {
        std::vector<uint16_t> b(f.size());
        for (size_t i = 0; i < f.size(); ++i) b[i] = f32_to_bf16(f[i]);
        return b;
    };
    auto x_tiles   = tile2d(to_bf16(x),            kS,  kH);
    auto ln1g_t    = tile2d(to_bf16(ln1_gamma_2d), kS,  kH);
    auto ln2g_t    = tile2d(to_bf16(ln2_gamma_2d), kS,  kH);
    auto Wq_t      = tile2d(to_bf16(W_q_scaled),   kH,  kH);
    auto Wk_t      = tile2d(to_bf16(W_k),          kH,  kH);
    auto Wv_t      = tile2d(to_bf16(W_v),          kH,  kH);
    auto Wo_t      = tile2d(to_bf16(W_o),          kH,  kH);
    auto Wgate_t   = tile2d(to_bf16(W_gate),       kH,  kFFN);
    auto Wup_t     = tile2d(to_bf16(W_up),         kH,  kFFN);
    auto Wdown_t   = tile2d(to_bf16(W_down),       kFFN, kH);

    // Causal mask (BF16 0/1, [St × St] tile grid).
    std::vector<float> mask_f(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_f[i * kS + j] = 1.0f;
    auto mask_t = tile2d(to_bf16(mask_f), kS, kS);

    // -----------------------------------------------------------------
    // 4. Device setup.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

    // Persistent DRAM tensors (all live across the whole forward).
    auto T_x      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ln1g   = ol::allocate_tensor_dram(*dev, kHt);  // gamma is 1 tile-row
    auto T_ln2g   = ol::allocate_tensor_dram(*dev, kHt);
    auto T_Wq     = ol::allocate_tensor_dram(*dev, kHt * kHt);
    auto T_Wk     = ol::allocate_tensor_dram(*dev, kHt * kHt);
    auto T_Wv     = ol::allocate_tensor_dram(*dev, kHt * kHt);
    auto T_Wo     = ol::allocate_tensor_dram(*dev, kHt * kHt);
    auto T_Wgate  = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wup    = ol::allocate_tensor_dram(*dev, kHt * kFFt);
    auto T_Wdown  = ol::allocate_tensor_dram(*dev, kFFt * kHt);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);

    // Intermediate / output tensors (overwritten each op).
    auto T_xnorm1 = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Q      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_K      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_V      = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_KT     = ol::allocate_tensor_dram(*dev, kHt * kSt);  // K transposed
    auto T_attn   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_proj   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up     = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_out    = ol::allocate_tensor_dram(*dev, kSt * kHt);

    // Upload weights / inputs.
    auto upload = [&](auto& tensor, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *tensor.buf, tiles.data(), tiles.size() * 2);
    };
    upload(T_x,     x_tiles);
    // ln_gamma in the rmsnorm kernel is replicated to all 32 rows of each
    // tile and indexed by tile-col only, so the test uploads the first
    // tile-row (kHt tiles) of the broadcasted gamma.
    upload(T_ln1g, std::vector<uint16_t>(ln1g_t.begin(), ln1g_t.begin() + kHt * kTileWords));
    upload(T_ln2g, std::vector<uint16_t>(ln2g_t.begin(), ln2g_t.begin() + kHt * kTileWords));
    upload(T_Wq,    Wq_t);
    upload(T_Wk,    Wk_t);
    upload(T_Wv,    Wv_t);
    upload(T_Wo,    Wo_t);
    upload(T_Wgate, Wgate_t);
    upload(T_Wup,   Wup_t);
    upload(T_Wdown, Wdown_t);
    upload(T_mask,  mask_t);

    // -----------------------------------------------------------------
    // 5. Forward pass — each op scoped so its kernel drops, then
    //    release_kernels() reclaims the KERNEL_CONFIG region.
    // -----------------------------------------------------------------
    auto run = [&](auto factory) {
        auto op = factory();
        ol::execute(*dev, op);
        // op destructs at end of scope → kernel ptr drops
    };

    // RMSNorm #1
    run([&]() { return ol::make_rmsnorm(*dev, T_x, T_ln1g, T_xnorm1, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);

    // QKV projections
    run([&]() { return ol::make_matmul(*dev, T_xnorm1, T_Wq, T_Q, kSt, kHt, kHt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_xnorm1, T_Wk, T_K, kSt, kHt, kHt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_xnorm1, T_Wv, T_V, kSt, kHt, kHt); });
    tt::foil::release_kernels(*dev, core);

    // K → KT (host round-trip; ugly, fix later)
    {
        std::vector<uint16_t> k_tiles(kSt * kHt * kTileWords);
        tt::foil::read_buffer(*dev, *T_K.buf, k_tiles.data(), k_tiles.size() * 2);
        auto k_rm  = untile2d(k_tiles, kS, kH);
        std::vector<uint16_t> kt_rm(kH * kS);
        for (uint32_t s = 0; s < kS; ++s)
            for (uint32_t d = 0; d < kH; ++d) kt_rm[d * kS + s] = k_rm[s * kH + d];
        auto kt_tiles = tile2d(kt_rm, kH, kS);
        tt::foil::write_buffer(*dev, *T_KT.buf, kt_tiles.data(), kt_tiles.size() * 2);
    }

    // MHA
    run([&]() { return ol::make_mha(*dev, T_Q, T_KT, T_V, T_mask, T_attn, kSt, kHt); });
    tt::foil::release_kernels(*dev, core);

    // Output projection + residual #1
    run([&]() { return ol::make_matmul(*dev, T_attn, T_Wo, T_proj, kSt, kHt, kHt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_eltwise_add(*dev, T_x, T_proj, T_xmid); });
    tt::foil::release_kernels(*dev, core);

    // RMSNorm #2
    run([&]() { return ol::make_rmsnorm(*dev, T_xmid, T_ln2g, T_ynorm, kSt, kHt, kEps); });
    tt::foil::release_kernels(*dev, core);

    // MLP: gate + up projections
    run([&]() { return ol::make_matmul(*dev, T_ynorm, T_Wgate, T_gate, kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_matmul(*dev, T_ynorm, T_Wup, T_up, kSt, kHt, kFFt); });
    tt::foil::release_kernels(*dev, core);

    // SiLU + elementwise mul
    run([&]() { return ol::make_silu(*dev, T_gate, T_silu); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });
    tt::foil::release_kernels(*dev, core);

    // Down projection + residual #2
    run([&]() { return ol::make_matmul(*dev, T_fused, T_Wdown, T_down, kSt, kFFt, kHt); });
    tt::foil::release_kernels(*dev, core);
    run([&]() { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_out); });
    tt::foil::release_kernels(*dev, core);

    // -----------------------------------------------------------------
    // 6. Read back and compare.
    // -----------------------------------------------------------------
    std::vector<uint16_t> out_tiles(kSt * kHt * kTileWords);
    tt::foil::read_buffer(*dev, *T_out.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, kH);

    // Tolerance: BF16 errors compound across 14 ops + reductions; allow a
    // generous absolute bound but require it to dominate over residual scale.
    const float kAbsTol = 0.15f;
    uint32_t bad = 0; float worst = 0.0f; uint32_t first_bad = kS * kH;
    for (uint32_t i = 0; i < kS * kH; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = ref_f[i];
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) { if (first_bad == kS * kH) first_bad = i; ++bad; }
    }
    if (bad != 0) {
        std::fprintf(stderr,
            "test_transformer_block: %u/%u mismatches; first at i=%u "
            "(row=%u col=%u): got=%.4f expected=%.4f, worst abs diff=%.5f (tol=%.5f)\n",
            bad, kS * kH, first_bad, first_bad / kH, first_bad % kH,
            bf16_to_f32(out_rm[first_bad]), ref_f[first_bad], worst, kAbsTol);
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_transformer_block: FAIL\n");
        return 1;
    }

    std::printf("test_transformer_block: PASS  (S=%u H=%u FFN=%u, worst=%.5f, tol=%.5f)\n",
                kS, kH, kFFN, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_transformer_block: FAIL — %s\n", e.what());
    return 1;
}
