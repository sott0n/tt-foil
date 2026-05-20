// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: MHA via op_lib.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

#ifndef MHA_OP_ST
#define MHA_OP_ST 2
#endif
#ifndef MHA_OP_DT
#define MHA_OP_DT 2
#endif

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kSt = MHA_OP_ST;
constexpr uint32_t kDt = MHA_OP_DT;
constexpr uint32_t kS = kSt * kTileH;
constexpr uint32_t kD = kDt * kTileW;
constexpr uint32_t kElems = kS * kD;

auto tile2d = [](const std::vector<uint16_t>& rm, uint32_t Rows, uint32_t Cols) {
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
};
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<float> Q(kElems), K(kElems), V(kElems);
    auto deterministic = [](uint32_t i, float a, float b) {
        uint32_t h = i * 1103515245u + 12345u;
        return -1.0f + 2.0f * (h % 1024u) / 1023.0f + a * 0.1f + b * 0.05f;
    };
    for (uint32_t r = 0; r < kS; ++r)
        for (uint32_t c = 0; c < kD; ++c) {
            Q[r * kD + c] = 0.3f * deterministic(r * 31 + c,        r % 4, c % 4);
            K[r * kD + c] = 0.3f * deterministic(r * 17 + c + 100u, c % 4, r % 4);
            V[r * kD + c] = 0.3f * deterministic(r * 11 + c + 200u, r % 8, c % 8);
        }
    const float inv_sqrt_d = 1.0f / std::sqrt((float)kD);

    // Causal mask
    std::vector<float> mask_f(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_f[i * kS + j] = 1.0f;

    // Host reference
    std::vector<float> scores(kS * kS), attn(kS * kS), ref_f(kElems);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j < kS; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < kD; ++k) s += Q[i * kD + k] * inv_sqrt_d * K[j * kD + k];
            scores[i * kS + j] = s;
        }
    for (uint32_t i = 0; i < kS; ++i) {
        double sum = 0.0;
        for (uint32_t j = 0; j < kS; ++j) sum += std::exp(scores[i * kS + j]) * mask_f[i * kS + j];
        for (uint32_t j = 0; j < kS; ++j)
            attn[i * kS + j] = static_cast<float>(std::exp(scores[i * kS + j]) * mask_f[i * kS + j] / sum);
    }
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t k = 0; k < kD; ++k) {
            float s = 0.0f;
            for (uint32_t j = 0; j < kS; ++j) s += attn[i * kS + j] * V[j * kD + k];
            ref_f[i * kD + k] = s;
        }

    // BF16 row-major then tile-format
    std::vector<uint16_t> q_rm(kElems), kt_rm(kElems), v_rm(kElems), mask_rm(kS * kS), ref_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        q_rm[i] = f32_to_bf16(Q[i] * inv_sqrt_d);
        v_rm[i] = f32_to_bf16(V[i]);
        ref_rm[i] = f32_to_bf16(ref_f[i]);
    }
    for (uint32_t s = 0; s < kS; ++s)
        for (uint32_t d = 0; d < kD; ++d) kt_rm[d * kS + s] = f32_to_bf16(K[s * kD + d]);
    for (uint32_t i = 0; i < kS * kS; ++i) mask_rm[i] = f32_to_bf16(mask_f[i]);

    auto q_tiles  = tile2d(q_rm,    kS, kD);
    auto kt_tiles = tile2d(kt_rm,   kD, kS);
    auto v_tiles  = tile2d(v_rm,    kS, kD);
    auto m_tiles  = tile2d(mask_rm, kS, kS);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto q  = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto kt = ol::allocate_tensor_dram(*dev, kDt * kSt);
    auto v  = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto m  = ol::allocate_tensor_dram(*dev, kSt * kSt);
    ol::TensorDesc out;

    tt::foil::write_buffer(*dev, *q.buf,  q_tiles.data(),  kSt * kDt * kTileBytes);
    tt::foil::write_buffer(*dev, *kt.buf, kt_tiles.data(), kDt * kSt * kTileBytes);
    tt::foil::write_buffer(*dev, *v.buf,  v_tiles.data(),  kSt * kDt * kTileBytes);
    tt::foil::write_buffer(*dev, *m.buf,  m_tiles.data(),  kSt * kSt * kTileBytes);

    auto op = ol::make_mha(*dev, q, kt, v, m, out, kSt, kDt);
    ol::execute(*dev, op);

    std::vector<uint16_t> out_tiles(kSt * kDt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *out.buf, out_tiles.data(), kSt * kDt * kTileBytes);

    std::vector<uint16_t> out_rm(kElems);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < kSt; ++rt)
        for (uint32_t ct = 0; ct < kDt; ++ct) {
            tt::foil::test::tile_to_row_major(out_tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    out_rm[(rt * kTileH + r) * kD + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }

    uint32_t bad = 0; float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float d = std::fabs(bf16_to_f32(out_rm[i]) - bf16_to_f32(ref_rm[i]));
        if (d > worst) worst = d;
        if (d > 0.03f) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "test_mha_op: %u bad, worst=%.5f\n", bad, worst);
        std::fprintf(stderr, "test_mha_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_mha_op: PASS  (St=%u Dt=%u worst=%.5f)\n", kSt, kDt, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_mha_op: FAIL — %s\n", e.what());
    return 1;
}
