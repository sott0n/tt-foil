// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: MHA (single-head, single-tile, no mask, no cache).
//   out = softmax(Q · K^T) · V
//   (Q is pre-scaled by 1/sqrt(D) on host; K is transposed on host.)
//
// Kernel CB layout:
//   0 cb_q          Q tile (Q/sqrt(D) on host)
//   1 cb_kt         K^T tile
//   2 cb_v          V tile
//   3 cb_reduce     scaler tile = BF16(1.0)
//   4 cb_scores     intermediate
//   5 cb_exp        intermediate
//   6 cb_sum        intermediate
//   7 cb_recip      intermediate
//   8 cb_softmaxed  intermediate
//  16 cb_out        output tile [S, D]

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// First cut: single-tile MHA → S = D = 32.
constexpr uint32_t kS = kTileH;   // 32 query tokens
constexpr uint32_t kD = kTileW;   // 32 head dim
constexpr uint32_t kElems = kS * kD;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

void single_tile_from_rm(const std::vector<uint16_t>& rm32x32, std::vector<uint16_t>& out) {
    out.clear();
    tt::foil::test::row_major_to_tile(rm32x32.data(), out);
}

void make_const_tile(float val, std::vector<uint16_t>& out) {
    out.clear();
    std::vector<uint16_t> block(kTileH * kTileW, f32_to_bf16(val));
    tt::foil::test::row_major_to_tile(block.data(), out);
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Host-side reference ----
    // Generate Q, K, V with small bounded values (BF16 + exp likes |x| small).
    std::vector<float> Q(kElems), K(kElems), V(kElems);
    auto deterministic = [](uint32_t i, float a, float b) {
        // Cheap LCG-ish pattern → [-1, +1]
        uint32_t h = i * 1103515245u + 12345u;
        return -1.0f + 2.0f * static_cast<float>(h % 1024u) / 1023.0f
               + a * 0.1f + b * 0.05f;
    };
    for (uint32_t r = 0; r < kS; ++r) {
        for (uint32_t c = 0; c < kD; ++c) {
            Q[r * kD + c] = 0.3f * deterministic(r * 31 + c,        r % 4, c % 4);
            K[r * kD + c] = 0.3f * deterministic(r * 17 + c + 100u, c % 4, r % 4);
            V[r * kD + c] = 0.3f * deterministic(r * 11 + c + 200u, r % 8, c % 8);
        }
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kD));

    // scores = (Q · 1/sqrt(D)) · K^T  →  [S, S]
    std::vector<float> scores(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i) {
        for (uint32_t j = 0; j < kS; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < kD; ++k) {
                s += (Q[i * kD + k] * inv_sqrt_d) * K[j * kD + k];
            }
            scores[i * kS + j] = s;
        }
    }
    // softmax per row
    std::vector<float> attn(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i) {
        double sum = 0.0;
        for (uint32_t j = 0; j < kS; ++j) sum += std::exp(scores[i * kS + j]);
        for (uint32_t j = 0; j < kS; ++j) attn[i * kS + j] = std::exp(scores[i * kS + j]) / sum;
    }
    // out = attn · V  →  [S, D]
    std::vector<float> ref_f(kS * kD, 0.0f);
    for (uint32_t i = 0; i < kS; ++i) {
        for (uint32_t k = 0; k < kD; ++k) {
            float s = 0.0f;
            for (uint32_t j = 0; j < kS; ++j) s += attn[i * kS + j] * V[j * kD + k];
            ref_f[i * kD + k] = s;
        }
    }

    // ---- Build BF16 tiles to upload ----
    // Q_scaled = Q / sqrt(D) (host-side scale)
    std::vector<uint16_t> q_rm(kElems), kt_rm(kElems), v_rm(kElems), ref_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        q_rm[i] = f32_to_bf16(Q[i] * inv_sqrt_d);
        v_rm[i] = f32_to_bf16(V[i]);
        ref_rm[i] = f32_to_bf16(ref_f[i]);
    }
    // KT: transpose K  [S, D] → [D, S]
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j < kD; ++j)
            kt_rm[j * kS + i] = f32_to_bf16(K[i * kD + j]);
    // Note: with kS == kD == 32, both [S,D] and [D,S] are 32x32 — same tile layout.

    std::vector<uint16_t> q_tile, kt_tile, v_tile;
    single_tile_from_rm(q_rm,  q_tile);
    single_tile_from_rm(kt_rm, kt_tile);
    single_tile_from_rm(v_rm,  v_tile);

    std::vector<uint16_t> scaler_tile;
    make_const_tile(1.0f, scaler_tile);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_q      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_kt     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_v      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);

    auto l1_q         = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_kt        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_v         = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_reduce    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_scores    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_exp       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_sum       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_recip     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_softmaxed = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_out       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_q,      q_tile.data(),      kTileBytes);
    tt::foil::write_buffer(*dev, *buf_kt,     kt_tile.data(),     kTileBytes);
    tt::foil::write_buffer(*dev, *buf_v,      v_tile.data(),      kTileBytes);
    tt::foil::write_buffer(*dev, *buf_scaler, scaler_tile.data(), kTileBytes);
    {
        std::vector<uint8_t> zero(kTileBytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), kTileBytes);
    }

    uint64_t q_noc      = tt::foil::make_noc_dram_addr(*dev, buf_q->device_addr);
    uint64_t kt_noc     = tt::foil::make_noc_dram_addr(*dev, buf_kt->device_addr);
    uint64_t v_noc      = tt::foil::make_noc_dram_addr(*dev, buf_v->device_addr);
    uint64_t scaler_noc = tt::foil::make_noc_dram_addr(*dev, buf_scaler->device_addr);
    uint64_t dst_noc    = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/mha.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/mha.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/mha.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 10> cbs = {{
        {0,  l1_q->device_addr,         kTileBytes, 1, kTileBytes},
        {1,  l1_kt->device_addr,        kTileBytes, 1, kTileBytes},
        {2,  l1_v->device_addr,         kTileBytes, 1, kTileBytes},
        {3,  l1_reduce->device_addr,    kTileBytes, 1, kTileBytes},
        {4,  l1_scores->device_addr,    kTileBytes, 1, kTileBytes},
        {5,  l1_exp->device_addr,       kTileBytes, 1, kTileBytes},
        {6,  l1_sum->device_addr,       kTileBytes, 1, kTileBytes},
        {7,  l1_recip->device_addr,     kTileBytes, 1, kTileBytes},
        {8,  l1_softmaxed->device_addr, kTileBytes, 1, kTileBytes},
        {16, l1_out->device_addr,       kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 8> ra_brisc = {
        static_cast<uint32_t>(q_noc & 0xffffffffu),      static_cast<uint32_t>(q_noc >> 32),
        static_cast<uint32_t>(kt_noc & 0xffffffffu),     static_cast<uint32_t>(kt_noc >> 32),
        static_cast<uint32_t>(v_noc & 0xffffffffu),      static_cast<uint32_t>(v_noc >> 32),
        static_cast<uint32_t>(scaler_noc & 0xffffffffu), static_cast<uint32_t>(scaler_noc >> 32),
    };
    std::array<uint32_t, 0> ra_trisc{};
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu), static_cast<uint32_t>(dst_noc >> 32),
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    // ---- Verify ----
    std::vector<uint16_t> out_tile(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, out_tile.data(), kTileBytes);
    std::vector<uint16_t> out_rm(kElems, 0);
    tt::foil::test::tile_to_row_major(out_tile.data(), out_rm.data());

    const float kAbsTol = 0.03f;
    uint32_t bad = 0, first_bad = kElems;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = bf16_to_f32(ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kElems) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_mha: %u/%u mismatches; first at i=%u (row=%u col=%u): "
            "got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, kElems, first_bad, first_bad / kD, first_bad % kD,
            bf16_to_f32(out_rm[first_bad]),
            bf16_to_f32(ref_rm[first_bad]), worst);
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_mha: FAIL\n");
        return 1;
    }

    std::printf("test_mha: PASS  (S=%u, D=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kS, kD, worst, kAbsTol);

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_mha: FAIL — %s\n", e.what());
    return 1;
}
