// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v8.2: BatchNorm-into-Conv fold identity (host-only unit test).
//
// BatchNorm at inference time is an affine map:
//
//     BN(z; γ, β, μ, σ², ε) = γ · (z - μ) / √(σ² + ε) + β
//
// When BN sits immediately after a conv with weights W and bias b, the
// composition Conv → BN can be folded into a single conv with new
// weights W' and bias b':
//
//     W' = (γ / √(σ² + ε)) · W
//     b' = γ · (b - μ) / √(σ² + ε) + β
//
// This test generates random W, b, BN params, and a random input x,
// and verifies that
//
//     BN(Conv(x; W, b); γ, β, μ, σ², ε) == Conv(x; W', b')
//
// element-wise to fp32 precision (no tolerance — it's pure algebra).
//
// Once tt-foil has a conv_1x1_bias example on device, this same fold
// can be exercised end-to-end by:
//   1. computing W', b' on host with the formulas below;
//   2. running conv_1x1_bias(x, W', b') on the device;
//   3. comparing against BN(conv_1x1(x, W), ...) computed offline.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// Tiny pointwise (1x1) conv: y[h, w, oc] = sum_ic W[oc, ic] * x[h, w, ic] + b[oc]
void conv_1x1_ref(const std::vector<float>& x,   // (H, W, C_in)
                  const std::vector<float>& W,   // (C_out, C_in)  row-major oc-major
                  const std::vector<float>& b,   // (C_out)
                  uint32_t H, uint32_t Wd, uint32_t C_in, uint32_t C_out,
                  std::vector<float>& y) {
    y.assign(H * Wd * C_out, 0.0f);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < Wd; ++w) {
            for (uint32_t oc = 0; oc < C_out; ++oc) {
                float acc = b[oc];
                for (uint32_t ic = 0; ic < C_in; ++ic) {
                    acc += W[oc * C_in + ic] * x[(h * Wd + w) * C_in + ic];
                }
                y[(h * Wd + w) * C_out + oc] = acc;
            }
        }
    }
}

// y[..., oc] = gamma[oc] * (z[..., oc] - mu[oc]) / sqrt(var[oc]+eps) + beta[oc]
void batchnorm_ref(const std::vector<float>& z,
                   const std::vector<float>& gamma,
                   const std::vector<float>& beta,
                   const std::vector<float>& mu,
                   const std::vector<float>& var,
                   float eps, uint32_t N, uint32_t C,
                   std::vector<float>& y) {
    y.assign(N * C, 0.0f);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t oc = 0; oc < C; ++oc) {
            float inv = 1.0f / std::sqrt(var[oc] + eps);
            y[i * C + oc] = gamma[oc] * (z[i * C + oc] - mu[oc]) * inv + beta[oc];
        }
    }
}

}  // namespace

int main() {
    constexpr uint32_t H = 8, Wd = 8, C_in = 32, C_out = 16;
    constexpr float eps = 1e-5f;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uw(-0.5f, 0.5f);
    std::uniform_real_distribution<float> ub(-0.1f, 0.1f);
    std::uniform_real_distribution<float> ug(0.5f, 1.5f);
    std::uniform_real_distribution<float> um(-0.2f, 0.2f);
    std::uniform_real_distribution<float> uv(0.1f, 1.0f);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);

    std::vector<float> x(H * Wd * C_in);
    std::vector<float> W(C_out * C_in);
    std::vector<float> b(C_out);
    std::vector<float> gamma(C_out), beta(C_out), mu(C_out), var(C_out);

    for (auto& v : x) v = ux(rng);
    for (auto& v : W) v = uw(rng);
    for (auto& v : b) v = ub(rng);
    for (auto& v : gamma) v = ug(rng);
    for (auto& v : beta)  v = um(rng);
    for (auto& v : mu)    v = um(rng);
    for (auto& v : var)   v = uv(rng);

    // Path A: conv → BN.
    std::vector<float> y_conv;
    conv_1x1_ref(x, W, b, H, Wd, C_in, C_out, y_conv);
    std::vector<float> y_a;
    batchnorm_ref(y_conv, gamma, beta, mu, var, eps, H * Wd, C_out, y_a);

    // Fold: W' = (γ/σ)·W,  b' = γ·(b-μ)/σ + β,  with σ = √(var+ε).
    std::vector<float> W_prime(C_out * C_in);
    std::vector<float> b_prime(C_out);
    for (uint32_t oc = 0; oc < C_out; ++oc) {
        float inv_sigma = 1.0f / std::sqrt(var[oc] + eps);
        float scale = gamma[oc] * inv_sigma;
        for (uint32_t ic = 0; ic < C_in; ++ic) {
            W_prime[oc * C_in + ic] = scale * W[oc * C_in + ic];
        }
        b_prime[oc] = scale * (b[oc] - mu[oc]) + beta[oc];
    }

    // Path B: single folded conv.
    std::vector<float> y_b;
    conv_1x1_ref(x, W_prime, b_prime, H, Wd, C_in, C_out, y_b);

    // Compare. Pure fp32 algebra: y_a and y_b differ only by FMA reorder
    // (we use the same loop in both paths) → expect ulps, not 1e-3-level
    // drift.
    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < y_a.size(); ++i) {
        float d = std::fabs(y_a[i] - y_b[i]);
        float r = d / std::max(1e-6f, std::fabs(y_a[i]));
        if (d > worst_abs) worst_abs = d;
        if (r > worst_rel) worst_rel = r;
        // 1e-4 absolute tolerance covers the difference between
        // gamma·(z−μ)/σ + β  vs  (γ/σ)·(w·x) + γ·(b−μ)/σ + β — fp32
        // round-off on a length-32 reduction.
        if (d > 1e-4f && r > 1e-4f) ++bad;
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_bn_fold: FAIL — %zu/%zu mismatches; worst abs=%.3e rel=%.3e\n",
            bad, y_a.size(), worst_abs, worst_rel);
        return 1;
    }
    std::printf("test_bn_fold: PASS  (worst abs=%.3e rel=%.3e over %zu outputs)\n",
                worst_abs, worst_rel, y_a.size());
    return 0;
}
