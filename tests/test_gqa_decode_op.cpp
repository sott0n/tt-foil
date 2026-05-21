// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: GqaDecode via op_lib.
//
// Single-tile sanity (St_q = St_kv = 1, no padding to mask): the decode
// kernel with full-1 mask should match the same single-head attention
// reference test_mha_op uses.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef GQAD_NUM_Q
#define GQAD_NUM_Q 2
#endif
#ifndef GQAD_NUM_KV
#define GQAD_NUM_KV 1
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kStQ = 1, kStKv = 1, kDt = 2;
constexpr uint32_t kS = kStQ * kTileH, kSk = kStKv * kTileH, kD = kDt * kTileW;
constexpr uint32_t kNumQ = GQAD_NUM_Q, kNumKv = GQAD_NUM_KV;
constexpr uint32_t kGqa = kNumQ / kNumKv;

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
}  // namespace

int main() try {
    static_assert(kNumQ % kNumKv == 0, "num_q must be a multiple of num_kv");

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<float> Q1(kS * kD), K1(kSk * kD), V1(kSk * kD);
    auto det = [](uint32_t i, float a, float b) {
        uint32_t h = i * 1103515245u + 12345u;
        return -1.0f + 2.0f * (h % 1024u) / 1023.0f + a * 0.1f + b * 0.05f;
    };
    for (uint32_t r = 0; r < kS; ++r)
        for (uint32_t c = 0; c < kD; ++c)
            Q1[r * kD + c] = 0.3f * det(r * 31 + c, r % 4, c % 4);
    for (uint32_t r = 0; r < kSk; ++r)
        for (uint32_t c = 0; c < kD; ++c) {
            K1[r * kD + c] = 0.3f * det(r * 17 + c + 100u, c % 4, r % 4);
            V1[r * kD + c] = 0.3f * det(r * 11 + c + 200u, r % 8, c % 8);
        }
    const float inv_sqrt_d = 1.0f / std::sqrt((float)kD);

    // Full-1 mask (no padding to discard for this sanity case).
    std::vector<float> mask_f(kS * kSk, 1.0f);

    // Single-head reference (Q row 0 attends to all kSk positions).
    std::vector<float> ref1(kS * kD, 0.0f);
    {
        std::vector<float> scores(kS * kSk), p(kS * kSk);
        for (uint32_t i = 0; i < kS; ++i)
            for (uint32_t j = 0; j < kSk; ++j) {
                float s = 0.0f;
                for (uint32_t k = 0; k < kD; ++k)
                    s += Q1[i * kD + k] * inv_sqrt_d * K1[j * kD + k];
                scores[i * kSk + j] = s;
            }
        for (uint32_t i = 0; i < kS; ++i) {
            double sum = 0.0;
            for (uint32_t j = 0; j < kSk; ++j) sum += std::exp(scores[i * kSk + j]);
            for (uint32_t j = 0; j < kSk; ++j)
                p[i * kSk + j] = static_cast<float>(std::exp(scores[i * kSk + j]) / sum);
        }
        for (uint32_t i = 0; i < kS; ++i)
            for (uint32_t k = 0; k < kD; ++k) {
                float s = 0.0f;
                for (uint32_t j = 0; j < kSk; ++j) s += p[i * kSk + j] * V1[j * kD + k];
                ref1[i * kD + k] = s;
            }
    }

    const uint32_t Nq = kNumQ  * kD;
    const uint32_t Nk = kNumKv * kD;
    std::vector<uint16_t> q_rm(kS * Nq), v_rm(kSk * Nk), kt_rm(Nk * kSk), mask_rm(kS * kSk);
    for (uint32_t h = 0; h < kNumQ; ++h)
        for (uint32_t r = 0; r < kS; ++r)
            for (uint32_t c = 0; c < kD; ++c)
                q_rm[r * Nq + h * kD + c] = f32_to_bf16(Q1[r * kD + c] * inv_sqrt_d);
    for (uint32_t kv = 0; kv < kNumKv; ++kv) {
        for (uint32_t r = 0; r < kSk; ++r)
            for (uint32_t c = 0; c < kD; ++c)
                v_rm[r * Nk + kv * kD + c] = f32_to_bf16(V1[r * kD + c]);
        for (uint32_t d = 0; d < kD; ++d)
            for (uint32_t s = 0; s < kSk; ++s)
                kt_rm[(kv * kD + d) * kSk + s] = f32_to_bf16(K1[s * kD + d]);
    }
    for (uint32_t i = 0; i < kS * kSk; ++i) mask_rm[i] = f32_to_bf16(mask_f[i]);

    auto q_tiles  = tile2d(q_rm,    kS,  Nq);
    auto kt_tiles = tile2d(kt_rm,   Nk,  kSk);
    auto v_tiles  = tile2d(v_rm,    kSk, Nk);
    auto m_tiles  = tile2d(mask_rm, kS,  kSk);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;
    auto q  = ol::allocate_tensor_dram(*dev, kStQ * kNumQ  * kDt);
    auto kt = ol::allocate_tensor_dram(*dev, kNumKv * kDt  * kStKv);
    auto v  = ol::allocate_tensor_dram(*dev, kStKv * kNumKv * kDt);
    auto m  = ol::allocate_tensor_dram(*dev, kStQ * kStKv);
    ol::TensorDesc out;
    tt::foil::write_buffer(*dev, *q.buf,  q_tiles.data(),  q_tiles.size()  * 2);
    tt::foil::write_buffer(*dev, *kt.buf, kt_tiles.data(), kt_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *v.buf,  v_tiles.data(),  v_tiles.size()  * 2);
    tt::foil::write_buffer(*dev, *m.buf,  m_tiles.data(),  m_tiles.size()  * 2);

    auto op = ol::make_gqa_decode(*dev, q, kt, v, m, out,
                                  kStQ, kStKv, kDt, kNumQ, kNumKv);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(kStQ * kNumQ * kDt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), out_tiles.size() * 2);

    std::vector<uint16_t> out_rm(kS * Nq);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    const uint32_t Ct = Nq / kTileW;
    for (uint32_t rt = 0; rt < kStQ; ++rt)
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            tt::foil::test::tile_to_row_major(out_tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    out_rm[(rt * kTileH + r) * Nq + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }

    tt::foil::close_device(std::move(dev));

    const float kAbsTol = 0.04f;
    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t h = 0; h < kNumQ; ++h)
        for (uint32_t r = 0; r < kS; ++r)
            for (uint32_t c = 0; c < kD; ++c) {
                float got = bf16_to_f32(out_rm[r * Nq + h * kD + c]);
                float exp = ref1[r * kD + c];
                float d = std::fabs(got - exp);
                if (d > worst) worst = d;
                if (d > kAbsTol) ++bad;
            }
    if (bad) {
        std::fprintf(stderr, "test_gqa_decode_op: %u mismatches, worst=%.5f tol=%.5f\n",
                     bad, worst, kAbsTol);
        return 1;
    }
    std::printf("test_gqa_decode_op: PASS  (St_q=%u St_kv=%u Dt=%u num_q=%u num_kv=%u worst=%.5f)\n",
                kStQ, kStKv, kDt, kNumQ, kNumKv, worst);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_gqa_decode_op: FAIL — %s\n", e.what());
    return 1;
}
