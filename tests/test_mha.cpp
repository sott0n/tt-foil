// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: MHA (single-head, no mask, no cache).
//   out = softmax(Q · K^T) · V
//   (Q is pre-scaled by 1/sqrt(D) on host; K is transposed on host.)
//
// Configurable via CMake (-DMHA_ST=<St> -DMHA_DT=<Dt>):
//   S = MHA_ST * 32     (sequence length)
//   D = MHA_DT * 32     (head dim)

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

#ifndef MHA_ST
#define MHA_ST 2
#endif
#ifndef MHA_DT
#define MHA_DT 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kSt = MHA_ST;
constexpr uint32_t kDt = MHA_DT;
constexpr uint32_t kS  = kSt * kTileH;
constexpr uint32_t kD  = kDt * kTileW;
constexpr uint32_t kElems = kS * kD;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Stream a 2D row-major [Rows, Cols] block into tile format.
//   Rows = St * 32, Cols = Dt * 32.
//   Output tiles laid out as St rows × Dt cols, index = qt * Dt + dt.
void tile_stream_2d(const std::vector<uint16_t>& rm, uint32_t Rows, uint32_t Cols,
                    std::vector<uint16_t>& out) {
    out.clear();
    const uint32_t RowTiles = Rows / kTileH;
    const uint32_t ColTiles = Cols / kTileW;
    out.reserve(static_cast<size_t>(RowTiles) * ColTiles * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < RowTiles; ++rt) {
        for (uint32_t ct = 0; ct < ColTiles; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = rm[(rt * kTileH + r) * Cols + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

void untile_stream_2d(const std::vector<uint16_t>& tiles, uint32_t Rows, uint32_t Cols,
                      std::vector<uint16_t>& rm) {
    rm.assign(Rows * Cols, 0);
    const uint32_t RowTiles = Rows / kTileH;
    const uint32_t ColTiles = Cols / kTileW;
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
    for (uint32_t rt = 0; rt < RowTiles; ++rt) {
        for (uint32_t ct = 0; ct < ColTiles; ++ct) {
            tt::foil::test::tile_to_row_major(tiles.data() + idx * kTileWords, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    rm[(rt * kTileH + r) * Cols + ct * kTileW + c] = block[r * kTileW + c];
            ++idx;
        }
    }
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
    std::vector<float> Q(kElems), K(kElems), V(kElems);
    auto deterministic = [](uint32_t i, float a, float b) {
        uint32_t h = i * 1103515245u + 12345u;
        return -1.0f + 2.0f * static_cast<float>(h % 1024u) / 1023.0f
               + a * 0.1f + b * 0.05f;
    };
    for (uint32_t r = 0; r < kS; ++r) {
        for (uint32_t c = 0; c < kD; ++c) {
            Q[r * kD + c] = 0.3f * deterministic(r * 31 + c,         r % 4, c % 4);
            K[r * kD + c] = 0.3f * deterministic(r * 17 + c + 100u,  c % 4, r % 4);
            V[r * kD + c] = 0.3f * deterministic(r * 11 + c + 200u,  r % 8, c % 8);
        }
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kD));

    std::vector<float> scores(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j < kS; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < kD; ++k)
                s += (Q[i * kD + k] * inv_sqrt_d) * K[j * kD + k];
            scores[i * kS + j] = s;
        }

    // Causal mask: M[i, j] = 1 if j <= i, else 0.
    std::vector<float> mask_f(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j)
            mask_f[i * kS + j] = 1.0f;

    std::vector<float> attn(kS * kS, 0.0f);
    for (uint32_t i = 0; i < kS; ++i) {
        double sum = 0.0;
        for (uint32_t j = 0; j < kS; ++j) sum += std::exp(scores[i * kS + j]) * mask_f[i * kS + j];
        for (uint32_t j = 0; j < kS; ++j) {
            float v = std::exp(scores[i * kS + j]) * mask_f[i * kS + j];
            attn[i * kS + j] = static_cast<float>(v / sum);
        }
    }

    std::vector<float> ref_f(kElems, 0.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t k = 0; k < kD; ++k) {
            float s = 0.0f;
            for (uint32_t j = 0; j < kS; ++j) s += attn[i * kS + j] * V[j * kD + k];
            ref_f[i * kD + k] = s;
        }

    // ---- BF16 row-major buffers ----
    std::vector<uint16_t> q_rm(kElems), kt_rm(kElems), v_rm(kElems), ref_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        q_rm[i]   = f32_to_bf16(Q[i] * inv_sqrt_d);
        v_rm[i]   = f32_to_bf16(V[i]);
        ref_rm[i] = f32_to_bf16(ref_f[i]);
    }
    // KT: transpose K [S, D] → [D, S]. Note kt_rm has shape [D, S], rm-indexed as
    // kt_rm[d * S + s] = K[s, d].
    std::vector<uint16_t> kt_rm_DxS(kElems);
    for (uint32_t s = 0; s < kS; ++s)
        for (uint32_t d = 0; d < kD; ++d)
            kt_rm_DxS[d * kS + s] = f32_to_bf16(K[s * kD + d]);

    // ---- Tile-format payloads ----
    std::vector<uint16_t> q_tiles, kt_tiles, v_tiles;
    tile_stream_2d(q_rm,    kS, kD, q_tiles);
    tile_stream_2d(kt_rm_DxS, kD, kS, kt_tiles);
    tile_stream_2d(v_rm,    kS, kD, v_tiles);
    const uint32_t qkv_bytes  = kSt * kDt * kTileBytes;
    const uint32_t kt_bytes   = kDt * kSt * kTileBytes;
    const uint32_t mask_bytes = kSt * kSt * kTileBytes;

    // mask: [S, S] BF16, then tile to [St, St] grid of tiles
    std::vector<uint16_t> mask_rm(kS * kS);
    for (uint32_t i = 0; i < kS * kS; ++i) mask_rm[i] = f32_to_bf16(mask_f[i]);
    std::vector<uint16_t> mask_tiles;
    tile_stream_2d(mask_rm, kS, kS, mask_tiles);

    std::vector<uint16_t> scaler_tile;
    make_const_tile(1.0f, scaler_tile);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_q      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, qkv_bytes,  core);
    auto buf_kt     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kt_bytes,   core);
    auto buf_v      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, qkv_bytes,  core);
    auto buf_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_mask   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, mask_bytes, core);
    auto buf_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, qkv_bytes,  core);

    // L1 CB buffers. fifo_size == num_pages * page_size.
    auto l1_q         = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, qkv_bytes,            core);
    auto l1_kt        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kt_bytes,             core);
    auto l1_v         = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, qkv_bytes,            core);
    auto l1_reduce    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,           core);
    auto l1_scores    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kSt * kTileBytes,     core);
    auto l1_exp       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kSt * kTileBytes,     core);
    auto l1_sum       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,           core);
    auto l1_recip     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,           core);
    auto l1_softmaxed = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kSt * kTileBytes,     core);
    auto l1_exp_m     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kSt * kTileBytes,     core);
    auto l1_mask      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, mask_bytes,           core);
    auto l1_out       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,           core);

    tt::foil::write_buffer(*dev, *buf_q,      q_tiles.data(),     qkv_bytes);
    tt::foil::write_buffer(*dev, *buf_kt,     kt_tiles.data(),    kt_bytes);
    tt::foil::write_buffer(*dev, *buf_v,      v_tiles.data(),     qkv_bytes);
    tt::foil::write_buffer(*dev, *buf_scaler, scaler_tile.data(), kTileBytes);
    tt::foil::write_buffer(*dev, *buf_mask,   mask_tiles.data(),  mask_bytes);
    {
        std::vector<uint8_t> zero(qkv_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), qkv_bytes);
    }

    uint64_t q_noc      = tt::foil::make_noc_dram_addr(*dev, buf_q->device_addr);
    uint64_t kt_noc     = tt::foil::make_noc_dram_addr(*dev, buf_kt->device_addr);
    uint64_t v_noc      = tt::foil::make_noc_dram_addr(*dev, buf_v->device_addr);
    uint64_t scaler_noc = tt::foil::make_noc_dram_addr(*dev, buf_scaler->device_addr);
    uint64_t mask_noc   = tt::foil::make_noc_dram_addr(*dev, buf_mask->device_addr);
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

    std::array<tt::foil::CbConfig, 12> cbs = {{
        {0,  l1_q->device_addr,         qkv_bytes,         kSt * kDt, kTileBytes},
        {1,  l1_kt->device_addr,        kt_bytes,          kDt * kSt, kTileBytes},
        {2,  l1_v->device_addr,         qkv_bytes,         kSt * kDt, kTileBytes},
        {3,  l1_reduce->device_addr,    kTileBytes,        1,         kTileBytes},
        {4,  l1_scores->device_addr,    kSt * kTileBytes,  kSt,       kTileBytes},
        {5,  l1_exp->device_addr,       kSt * kTileBytes,  kSt,       kTileBytes},
        {6,  l1_sum->device_addr,       kTileBytes,        1,         kTileBytes},
        {7,  l1_recip->device_addr,     kTileBytes,        1,         kTileBytes},
        {8,  l1_softmaxed->device_addr, kSt * kTileBytes,  kSt,       kTileBytes},
        {9,  l1_exp_m->device_addr,     kSt * kTileBytes,  kSt,       kTileBytes},
        {10, l1_mask->device_addr,      mask_bytes,        kSt * kSt, kTileBytes},
        {16, l1_out->device_addr,       kTileBytes,        1,         kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 12> ra_brisc = {
        static_cast<uint32_t>(q_noc & 0xffffffffu),      static_cast<uint32_t>(q_noc >> 32),
        static_cast<uint32_t>(kt_noc & 0xffffffffu),     static_cast<uint32_t>(kt_noc >> 32),
        static_cast<uint32_t>(v_noc & 0xffffffffu),      static_cast<uint32_t>(v_noc >> 32),
        static_cast<uint32_t>(scaler_noc & 0xffffffffu), static_cast<uint32_t>(scaler_noc >> 32),
        static_cast<uint32_t>(mask_noc & 0xffffffffu),   static_cast<uint32_t>(mask_noc >> 32),
        kSt, kDt,
    };
    std::array<uint32_t, 2> ra_trisc = {kSt, kDt};
    std::array<uint32_t, 4> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu), static_cast<uint32_t>(dst_noc >> 32),
        kSt, kDt,
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> out_tiles(kSt * kDt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, out_tiles.data(), qkv_bytes);
    std::vector<uint16_t> out_rm;
    untile_stream_2d(out_tiles, kS, kD, out_rm);

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

    std::printf("test_mha: PASS  (St=%u, Dt=%u, S=%u, D=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kSt, kDt, kS, kD, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_mha: FAIL — %s\n", e.what());
    return 1;
}
