// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device LLM runner for Qwen3-VL-2B. Extends qwen3_run with:
//   - Variable prefill sequence length (kS inferred from token_ids.bin size).
//   - Optional visual_embeds.bin: host-side injection of visual features at
//     IMAGE_PAD_TOKEN (151655) positions before upload to device.
//   - KT cache written in slot-major order (required by gqa_decode for kSt > 1).
//   - Separate prefill / decode tensor sets (ops validate num_tiles == exact).
//   - kv_append slot1_r offset: (kSt-1)*32 + t  →  decode slot starts after prefill.
//
// Usage:
//   TT_FOIL_QWEN3_DATA=data/qwen3_vl_2b TT_FOIL_OPS_DIR=ops \
//   TT_FOIL_DEVICE=0 \
//   ./build/models/qwen3_vl_2b/qwen3vl_run <token_ids.bin> <num_decode> [<visual_embeds.bin>]
//
//   token_ids.bin    : uint32[kS] — kS must be a multiple of 32.
//                      IMAGE_PAD positions hold token id 151655.
//   visual_embeds.bin: bf16[N_vis × kH] row-major — one row per image-pad token.
//   num_decode       : greedy decode steps after prefill.
//
// Required env: TT_FOIL_QWEN3_DATA, TT_FOIL_OPS_DIR
// Optional env: TT_FOIL_DEVICE (default 0), TT_FOIL_FAST_DISPATCH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <immintrin.h>
#include <map>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "fast_dispatch.hpp"
#include "device.hpp"
#include "tile_utils.hpp"

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

// Model geometry constants (Qwen3-VL-2B LLM body — identical to qwen3_run).
constexpr uint32_t kNumLayers = 28;
constexpr uint32_t kH         = 2048;
constexpr uint32_t kV         = 151936;
constexpr uint32_t kFFN       = 6144;
constexpr uint32_t kNumQ      = 16;
constexpr uint32_t kNumKv     = 8;
constexpr uint32_t kHeadDim   = 128;
constexpr float    kRopeTheta = 5000000.0f;
constexpr float    kEps       = 1e-6f;

// Tile-count constants.
constexpr uint32_t kHt       = kH   / kTileW;   // 64
constexpr uint32_t kFFt      = kFFN / kTileW;   // 192
constexpr uint32_t kDt       = kHeadDim / kTileW; // 4
constexpr uint32_t kDtHalf   = kDt / 2;           // 2
constexpr uint32_t kNqDt     = kNumQ  * kDt;      // 64
constexpr uint32_t kNkDt     = kNumKv * kDt;      // 32
constexpr uint32_t kNqkvDt   = kNqDt + 2 * kNkDt; // 128
constexpr uint32_t kFFtFused = 2 * kFFt;           // 384
constexpr uint32_t kVt       = kV / kTileW;        // 4748

// kStDec: decode is always 1 query tile (1 token padded into a 32-row tile).
constexpr uint32_t kStDec    = 1;

// Qwen3-VL image-pad token id.
constexpr uint32_t IMAGE_PAD_TOKEN = 151655;

// Extra decode slots allocated beyond the prefill region.
// 4 slots × 32 rows = 128 positions of decode capacity.
constexpr uint32_t kDecodeSlots = 4;

using Clock = std::chrono::steady_clock;

struct Prof {
    std::map<std::string, std::pair<double, uint64_t>> acc;
    void add(const std::string& tag, double ms) {
        auto& e = acc[tag];
        e.first  += ms;
        e.second += 1;
    }
    void report() {
        double total = 0;
        for (auto& [k, v] : acc) total += v.first;
        std::fprintf(stderr, "\n=== profile (ms) ===\n");
        std::vector<std::pair<std::string, std::pair<double, uint64_t>>> v(acc.begin(), acc.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
        for (auto& [k, e] : v)
            std::fprintf(stderr, "  %-34s %10.1f ms  (%6lu calls, %7.2f ms/call, %5.1f%%)\n",
                k.c_str(), e.first, (unsigned long)e.second,
                e.first / std::max<uint64_t>(1, e.second),
                100.0 * e.first / std::max(1e-9, total));
        std::fprintf(stderr, "  %-34s %10.1f ms\n", "TOTAL", total);
    }
};
static Prof g_prof;

struct ScopedTimer {
    std::string tag;
    Clock::time_point t0;
    ScopedTimer(std::string s) : tag(std::move(s)), t0(Clock::now()) {}
    ~ScopedTimer() {
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        g_prof.add(tag, ms);
    }
};
#define TIMED(tag) ScopedTimer _t_##__LINE__(tag)

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
std::vector<uint16_t> load_bin(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + p);
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    if (f.gcount() != static_cast<std::streamsize>(n * 2))
        throw std::runtime_error("short read: " + p);
    return v;
}
std::vector<uint32_t> load_u32_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open: " + p);
    std::streamsize sz = f.tellg();
    if (sz % 4 != 0) throw std::runtime_error("odd-byte uint32 file: " + p);
    f.seekg(0);
    std::vector<uint32_t> v(sz / 4);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}
std::vector<uint16_t> load_bf16_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open: " + p);
    std::streamsize sz = f.tellg();
    if (sz % 2 != 0) throw std::runtime_error("odd-byte bf16 file: " + p);
    f.seekg(0);
    std::vector<uint16_t> v(sz / 2);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

// ---------------------------------------------------------------------------
// Tile layout helpers (AVX2-vectorised for the hot paths).
// ---------------------------------------------------------------------------
__attribute__((target("avx2")))
std::vector<uint16_t> tile2d(const std::vector<uint16_t>& rm, uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> out(static_cast<size_t>(Rt) * Ct * kTileWords);
    uint16_t* dst = out.data();
    const uint16_t* src = rm.data();
    for (uint32_t rt = 0; rt < Rt; ++rt) {
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            for (uint32_t fr = 0; fr < 2; ++fr) {
                for (uint32_t fc = 0; fc < 2; ++fc) {
                    for (uint32_t r = 0; r < 16; ++r) {
                        const uint16_t* s = src
                            + (static_cast<size_t>(rt * kTileH + fr * 16 + r)) * Cols
                            + ct * kTileW + fc * 16;
                        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v);
                        dst += 16;
                    }
                }
            }
        }
    }
    return out;
}
std::vector<uint16_t> untile2d(const std::vector<uint16_t>& tiles, uint32_t Rows, uint32_t Cols) {
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
std::vector<uint16_t> gamma_to_tiles(const std::vector<uint16_t>& g, uint32_t D) {
    std::vector<uint16_t> rm(kTileH * D);
    for (uint32_t r = 0; r < kTileH; ++r)
        for (uint32_t c = 0; c < D; ++c) rm[r * D + c] = g[c];
    return tile2d(rm, kTileH, D);
}

// ---------------------------------------------------------------------------
// Layer weights
// ---------------------------------------------------------------------------
struct LayerW {
    tt::foil::op_lib::TensorDesc ln1g, ln2g, qng, kng;
    // Separate Q/K/V and gate/up weights: fused matmul output ordering
    // interleaves rows for Mt>1 (tile (rt, ct) at index rt*Nt+ct), so a
    // simple offset view into the fused output captures Q_row0+K_row0+
    // V_row0 instead of Q_rows0..kSt-1. Using separate matmuls per head
    // keeps each output contiguous.
    tt::foil::op_lib::TensorDesc Wq, Wk, Wv, Wo;
    tt::foil::op_lib::TensorDesc Wgate, Wup, Wdown;
};

struct TiledLayer {
    std::vector<uint16_t> ln1g, ln2g, qng, kng;
    std::vector<uint16_t> Wq, Wk, Wv;
    std::vector<uint16_t> Wo;
    std::vector<uint16_t> Wgate, Wup;
    std::vector<uint16_t> Wdown;
};

TiledLayer prepare_layer(const std::string& d) {
    auto LD = [&](const char* n, std::size_t k) { return load_bin(d + "/" + n, k); };
    auto ln1g  = LD("ln1_gamma.bin", kH);
    auto ln2g  = LD("ln2_gamma.bin", kH);
    auto qng   = LD("q_norm.bin",    kHeadDim);
    auto kng   = LD("k_norm.bin",    kHeadDim);
    auto Wq    = LD("W_q.bin",       kH * kNumQ  * kHeadDim);
    auto Wk    = LD("W_k.bin",       kH * kNumKv * kHeadDim);
    auto Wv    = LD("W_v.bin",       kH * kNumKv * kHeadDim);
    auto Wo    = LD("W_o.bin",       kNumQ * kHeadDim * kH);
    auto Wgate = LD("W_gate.bin",    kH * kFFN);
    auto Wup   = LD("W_up.bin",      kH * kFFN);
    auto Wdown = LD("W_down.bin",    kFFN * kH);

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (auto& v : qng) v = f32_to_bf16(bf16_to_f32(v) * inv_sqrt_d);

    TiledLayer T;
    T.ln1g = gamma_to_tiles(ln1g, kH);
    T.ln2g = gamma_to_tiles(ln2g, kH);
    T.qng  = gamma_to_tiles(qng,  kHeadDim);
    T.kng  = gamma_to_tiles(kng,  kHeadDim);

    T.Wq    = tile2d(Wq,    kH,    kNumQ  * kHeadDim);
    T.Wk    = tile2d(Wk,    kH,    kNumKv * kHeadDim);
    T.Wv    = tile2d(Wv,    kH,    kNumKv * kHeadDim);
    T.Wo    = tile2d(Wo,    kNumQ * kHeadDim, kH);
    T.Wgate = tile2d(Wgate, kH,    kFFN);
    T.Wup   = tile2d(Wup,   kH,    kFFN);
    T.Wdown = tile2d(Wdown, kFFN,  kH);
    return T;
}

LayerW upload_layer(tt::foil::Device& dev, const TiledLayer& T) {
    namespace ol = tt::foil::op_lib;
    LayerW L;
    L.ln1g  = ol::allocate_tensor_dram(dev, kHt);
    L.ln2g  = ol::allocate_tensor_dram(dev, kHt);
    L.qng   = ol::allocate_tensor_dram(dev, kDt);
    L.kng   = ol::allocate_tensor_dram(dev, kDt);
    L.Wq    = ol::allocate_tensor_dram(dev, kHt * kNqDt);
    L.Wk    = ol::allocate_tensor_dram(dev, kHt * kNkDt);
    L.Wv    = ol::allocate_tensor_dram(dev, kHt * kNkDt);
    L.Wo    = ol::allocate_tensor_dram(dev, kNqDt * kHt);
    L.Wgate = ol::allocate_tensor_dram(dev, kHt * kFFt);
    L.Wup   = ol::allocate_tensor_dram(dev, kHt * kFFt);
    L.Wdown = ol::allocate_tensor_dram(dev, kFFt * kHt);
    auto up = [&](auto& t, const std::vector<uint16_t>& d) {
        tt::foil::write_buffer(dev, *t.buf, d.data(), d.size() * 2);
    };
    up(L.ln1g,  T.ln1g);
    up(L.ln2g,  T.ln2g);
    up(L.qng,   T.qng);
    up(L.kng,   T.kng);
    up(L.Wq,    T.Wq);
    up(L.Wk,    T.Wk);
    up(L.Wv,    T.Wv);
    up(L.Wo,    T.Wo);
    up(L.Wgate, T.Wgate);
    up(L.Wup,   T.Wup);
    up(L.Wdown, T.Wdown);
    return L;
}

// ---------------------------------------------------------------------------
// A zero-copy offset view into an existing TensorDesc buffer.
// ---------------------------------------------------------------------------
namespace ol = tt::foil::op_lib;
ol::TensorDesc make_view(const ol::TensorDesc& base,
                         uint32_t tile_offset,
                         uint32_t num_tiles_view) {
    ol::TensorDesc v;
    v.buf = std::make_shared<tt::foil::Buffer>();
    v.buf->location    = tt::foil::BufferLocation::DRAM;
    v.buf->device_addr = base.buf->device_addr
                         + static_cast<uint64_t>(tile_offset) * kTileBytes;
    v.buf->size_bytes  = static_cast<std::size_t>(num_tiles_view) * kTileBytes;
    v.num_tiles        = num_tiles_view;
    return v;
}

}  // namespace

int main(int argc, char** argv) try {
    const auto wall_t0 = Clock::now();
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: qwen3vl_run <token_ids.bin> <num_decode> [<visual_embeds.bin>]\n"
            "  env: TT_FOIL_QWEN3_DATA TT_FOIL_OPS_DIR [TT_FOIL_DEVICE]\n");
        return 2;
    }
    const std::string prompt_path  = argv[1];
    const uint32_t   kNumDecode    = std::stoul(argv[2]);
    const std::string vis_path     = (argc >= 4) ? argv[3] : "";

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;
    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    if (!data_env) throw std::runtime_error("set TT_FOIL_QWEN3_DATA");
    const std::string root = data_env;
    const std::string mdir = root + "/model";

    // -----------------------------------------------------------------------
    // Token IDs — kS inferred from file size (must be a multiple of 32).
    // -----------------------------------------------------------------------
    auto token_ids = load_u32_file(prompt_path);
    const uint32_t kS = static_cast<uint32_t>(token_ids.size());
    if (kS == 0 || kS % kTileH != 0)
        throw std::runtime_error("token_ids.bin size must be a multiple of 32 uint32s, got " +
                                 std::to_string(kS));
    const uint32_t kSt         = kS / kTileH;
    const uint32_t kStKvDec    = kSt + kDecodeSlots;  // prefill + extra decode slots

    if (kNumDecode > kDecodeSlots * kTileH)
        throw std::runtime_error("num_decode > decode capacity (" +
                                 std::to_string(kDecodeSlots * kTileH) + ")");

    std::fprintf(stderr,
        "qwen3vl_run: kS=%u (kSt=%u), kStKvDec=%u, kNumDecode=%u\n",
        kS, kSt, kStKvDec, kNumDecode);

    // -----------------------------------------------------------------------
    // Visual embeddings (optional).
    // -----------------------------------------------------------------------
    std::vector<uint16_t> visual_embeds;
    uint32_t num_vis = 0;
    if (!vis_path.empty()) {
        visual_embeds = load_bf16_file(vis_path);
        if (visual_embeds.size() % kH != 0)
            throw std::runtime_error("visual_embeds.bin size not a multiple of kH=" +
                                     std::to_string(kH));
        num_vis = static_cast<uint32_t>(visual_embeds.size() / kH);
        std::fprintf(stderr, "  visual tokens: %u\n", num_vis);
    }

    // -----------------------------------------------------------------------
    // Embed-tokens table (host side — used for both prefill injection and
    // per-step decode embedding lookup).
    // -----------------------------------------------------------------------
    std::fprintf(stderr, "loading embed_tokens (%.0f MB) + lm_head + final_norm...\n",
        static_cast<double>(kV) * kH * 2 / 1e6);
    auto fut_embed_table = std::async(std::launch::async, [&]{
        return load_bin(mdir + "/embed_tokens.bin", static_cast<std::size_t>(kV) * kH);
    });
    auto fut_lmhead_tiles = std::async(std::launch::async, [&]{
        return load_bin(mdir + "/lm_head_tiled.bin",
                        static_cast<std::size_t>(kHt) * kVt * kTileWords);
    });
    auto fut_final_g_tiles = std::async(std::launch::async, [&]{
        auto g = load_bin(mdir + "/final_norm.bin", kH);
        return gamma_to_tiles(g, kH);
    });

    // -----------------------------------------------------------------------
    // RoPE tables [kS × kHalf] for prefill; per-position for decode.
    // -----------------------------------------------------------------------
    const uint32_t kHalf = kHeadDim / 2;
    auto rope_at = [&](uint32_t pos) {
        std::vector<uint16_t> c(kHalf), s(kHalf);
        for (uint32_t i = 0; i < kHalf; ++i) {
            double freq  = 1.0 / std::pow(static_cast<double>(kRopeTheta),
                                          static_cast<double>(i) / kHalf);
            double angle = static_cast<double>(pos) * freq;
            c[i] = f32_to_bf16(static_cast<float>(std::cos(angle)));
            s[i] = f32_to_bf16(static_cast<float>(std::sin(angle)));
        }
        return std::make_pair(std::move(c), std::move(s));
    };

    std::vector<uint16_t> cos_rm(static_cast<size_t>(kS) * kHalf);
    std::vector<uint16_t> sin_rm(static_cast<size_t>(kS) * kHalf);
    for (uint32_t r = 0; r < kS; ++r) {
        auto [c, s] = rope_at(r);
        std::copy(c.begin(), c.end(), cos_rm.begin() + r * kHalf);
        std::copy(s.begin(), s.end(), sin_rm.begin() + r * kHalf);
    }

    // Prefill causal mask: flash gqa_fused needs only the within-block 32x32
    // lower-triangular tile (off-diagonal key blocks are fully in-range).
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    std::vector<uint16_t> tri_rm(static_cast<size_t>(kTileH) * kTileW, 0);
    for (uint32_t r = 0; r < kTileH; ++r)
        for (uint32_t c = 0; c <= r; ++c)
            tri_rm[r * kTileW + c] = one_bf16;

    auto cos_tiles    = tile2d(cos_rm, kS, kHalf);
    auto sin_tiles    = tile2d(sin_rm, kS, kHalf);
    auto mask_tiles   = tile2d(tri_rm, kTileH, kTileW);  // single tile
    auto final_g_tiles = fut_final_g_tiles.get();

    // -----------------------------------------------------------------------
    // Build prefill initial hidden state on host (embed_tokens + VL inject).
    // -----------------------------------------------------------------------
    auto embed_table = fut_embed_table.get();
    std::vector<uint16_t> hidden_rm(static_cast<size_t>(kS) * kH, 0);
    {
        uint32_t vis_idx = 0;
        for (uint32_t s = 0; s < kS; ++s) {
            const uint32_t tid = token_ids[s];
            if (tid == IMAGE_PAD_TOKEN && vis_idx < num_vis) {
                std::copy(
                    visual_embeds.begin() + vis_idx * kH,
                    visual_embeds.begin() + (vis_idx + 1) * kH,
                    hidden_rm.begin() + static_cast<size_t>(s) * kH);
                ++vis_idx;
            } else {
                const uint32_t safe_tid = std::min(tid, kV - 1);
                std::copy(
                    embed_table.begin() + static_cast<size_t>(safe_tid) * kH,
                    embed_table.begin() + static_cast<size_t>(safe_tid + 1) * kH,
                    hidden_rm.begin() + static_cast<size_t>(s) * kH);
            }
        }
        if (vis_idx != num_vis)
            std::fprintf(stderr,
                "WARN: visual_embeds has %u rows but only %u IMAGE_PAD tokens found\n",
                num_vis, vis_idx);
    }
    auto prefill_hidden_tiles = tile2d(hidden_rm, kS, kH);
    hidden_rm.clear();
    hidden_rm.shrink_to_fit();

    // -----------------------------------------------------------------------
    // Open device.
    // -----------------------------------------------------------------------
    // Persistent matmul grids (matmul OpCache).
    //
    // L1 CB layout depends only on Kt. Mt/Nt are RTAs. All Kt=64 matmuls
    // (qkv/o/ffn_gate+up) share one pinned kernel on kMatmulKt64Grid;
    // ffn_down (Kt=192) gets its own grid; lm_head keeps its 8-way
    // sharding on a third grid. Disjoint cores so per-shape L1 stays
    // isolated.
    //
    // Per-core L1 (within 855 KB user arena):
    //   kMatmulKt64Grid  Kt=64,  cb_b depth 2*Kt → ~386 KB
    //   kMatmulKt192Grid Kt=192, cb_b depth Kt   → ~770 KB
    //   kLmHeadGrid      Kt=64,  cb_b depth 2*Kt → ~386 KB
    //
    // Row 2 already hosts the prefill cached ops (kPreRms... below), so
    // these persistent matmul grids start from row 3.
    const std::vector<tt::foil::CoreCoord> kMatmulKt64Grid = {
        {3, 0}, {3, 1}, {3, 2}, {3, 3},
    };
    const std::vector<tt::foil::CoreCoord> kMatmulKt192Grid = {
        {4, 0}, {4, 1}, {4, 2}, {4, 3},
    };
    const std::vector<tt::foil::CoreCoord> kLmHeadGrid = {
        {5, 0}, {5, 1}, {5, 2}, {5, 3},
        {5, 4}, {5, 5}, {5, 6}, {5, 7},
    };
    // (0,0) is the default transient core for embed/argmax/etc. (`core` below)
    // + where rmsnorm_rope is pinned. Must be booted.
    std::vector<tt::foil::CoreCoord> boot_cores = { {0, 0} };
    for (const auto& c : kMatmulKt64Grid)  boot_cores.push_back(c);
    for (const auto& c : kMatmulKt192Grid) boot_cores.push_back(c);
    for (const auto& c : kLmHeadGrid)      boot_cores.push_back(c);
    const bool use_fd = std::getenv("TT_FOIL_FAST_DISPATCH") != nullptr;
    const tt::foil::CoreCoord fd_dispatcher_core{1, 0};
    if (use_fd) boot_cores.push_back(fd_dispatcher_core);

    const tt::foil::CoreCoord kCachedCore       {1, 1};
    const tt::foil::CoreCoord kCachedAddRmsCore {1, 2};
    const tt::foil::CoreCoord kCachedSiluCore   {1, 3};
    const tt::foil::CoreCoord kCachedGqaCore    {1, 4};
    const tt::foil::CoreCoord kCachedKvCore     {1, 5};
    boot_cores.push_back(kCachedCore);
    boot_cores.push_back(kCachedAddRmsCore);
    boot_cores.push_back(kCachedSiluCore);
    boot_cores.push_back(kCachedGqaCore);
    boot_cores.push_back(kCachedKvCore);
    // Prefill cached cores — separate from decode cached cores because
    // each op type has TWO shapes (kSt=10 prefill, kStDec=1 decode) and
    // op_cache pin_persistent would otherwise stack both on one core
    // and overflow L1 (~1.5 MB usable per Tensix on Blackhole).
    const tt::foil::CoreCoord kPreRmsCore       {2, 0};
    const tt::foil::CoreCoord kPreRopeQCore     {2, 1};
    const tt::foil::CoreCoord kPreRopeKCore     {2, 2};
    const tt::foil::CoreCoord kPreAddRmsCore    {2, 3};
    const tt::foil::CoreCoord kPreSiluCore      {2, 4};
    const tt::foil::CoreCoord kPreTransposeCore {2, 5};
    const tt::foil::CoreCoord kPreGqaCore       {2, 6};
    const tt::foil::CoreCoord kPreAddCore       {2, 7};
    const tt::foil::CoreCoord kPreKvSnapCore    {1, 6};
    boot_cores.push_back(kPreRmsCore);
    boot_cores.push_back(kPreRopeQCore);
    boot_cores.push_back(kPreRopeKCore);
    boot_cores.push_back(kPreAddRmsCore);
    boot_cores.push_back(kPreSiluCore);
    boot_cores.push_back(kPreTransposeCore);
    boot_cores.push_back(kPreGqaCore);
    boot_cores.push_back(kPreAddCore);
    boot_cores.push_back(kPreKvSnapCore);

    auto dev = tt::foil::open_device(pcie_index, "", boot_cores);
    tt::foil::CoreCoord core{0, 0};

    std::unique_ptr<tt::foil::FastDispatch> fd_owner;
    if (use_fd) {
        fd_owner = std::make_unique<tt::foil::FastDispatch>(*dev, fd_dispatcher_core);
        fd_owner->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        dev->fast_dispatch = fd_owner.get();
        std::fprintf(stderr, "qwen3vl_run: fast-dispatch enabled\n");
    }

    // -----------------------------------------------------------------------
    // Static device tensors (shared between prefill and decode).
    // -----------------------------------------------------------------------
    auto T_W_lm    = ol::allocate_tensor_dram(*dev, kHt * kVt);
    auto T_final_g = ol::allocate_tensor_dram(*dev, kHt);
    auto T_logits  = ol::allocate_tensor_dram(*dev, kVt);  // 1-row lm_head output

    {
        auto lmhead_tiles = fut_lmhead_tiles.get();
        tt::foil::write_buffer(*dev, *T_W_lm.buf, lmhead_tiles.data(), lmhead_tiles.size() * 2);
    }
    tt::foil::write_buffer(*dev, *T_final_g.buf, final_g_tiles.data(), final_g_tiles.size() * 2);

    ol::TensorDesc T_argmax;
    T_argmax.buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, 2048);
    T_argmax.num_tiles = 0;

    // Decode-specific 1-row RoPE + mask.
    auto T_dcos  = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dsin  = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dmask = ol::allocate_tensor_dram(*dev, kStDec * kStKvDec);

    // -----------------------------------------------------------------------
    // Prefill tensors (all sized for kSt rows).
    // -----------------------------------------------------------------------
    auto T_cos     = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_sin_pre = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_mask    = ol::allocate_tensor_dram(*dev, 1);  // single tri tile (flash)
    tt::foil::write_buffer(*dev, *T_cos.buf,     cos_tiles.data(),  cos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_sin_pre.buf, sin_tiles.data(),  sin_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_mask.buf,    mask_tiles.data(), mask_tiles.size() * 2);

    auto T_pre_in    = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_pre_out   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xnorm1    = ol::allocate_tensor_dram(*dev, kSt * kHt);

    auto T_Q_pre     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K_pre     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V_pre     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr_pre    = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr_pre    = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Kt_pre    = ol::allocate_tensor_dram(*dev, kNkDt * kSt);   // transposed KT for gqa_fused
    auto T_attn_pre  = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_proj_pre  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid_pre  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm_pre = ol::allocate_tensor_dram(*dev, kSt * kHt);

    auto T_gate_pre   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up_pre     = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu_pre   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down_pre   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_normed_pre = ol::allocate_tensor_dram(*dev, kSt * kHt);

    // View of the LAST tile-row of T_normed_pre — fed into the 1-row lm_head.
    auto T_normed_last = make_view(T_normed_pre, (kSt - 1) * kHt, kHt);

    tt::foil::write_buffer(*dev, *T_pre_in.buf,
                           prefill_hidden_tiles.data(), prefill_hidden_tiles.size() * 2);
    prefill_hidden_tiles.clear();

    // -----------------------------------------------------------------------
    // Decode tensors (kStDec = 1 row throughout).
    // -----------------------------------------------------------------------
    auto T_dec_in    = ol::allocate_tensor_dram(*dev, kStDec * kHt);
    auto T_dec_out   = ol::allocate_tensor_dram(*dev, kStDec * kHt);
    auto T_dec_xnorm = ol::allocate_tensor_dram(*dev, kStDec * kHt);

    auto T_Q_dec     = ol::allocate_tensor_dram(*dev, kStDec * kNqDt);
    auto T_K_dec     = ol::allocate_tensor_dram(*dev, kStDec * kNkDt);
    auto T_V_dec     = ol::allocate_tensor_dram(*dev, kStDec * kNkDt);
    auto T_Qr_dec    = ol::allocate_tensor_dram(*dev, kStDec * kNqDt);
    auto T_Kr_dec    = ol::allocate_tensor_dram(*dev, kStDec * kNkDt);
    auto T_attn_dec  = ol::allocate_tensor_dram(*dev, kStDec * kNqDt);
    auto T_proj_dec  = ol::allocate_tensor_dram(*dev, kStDec * kHt);
    auto T_xmid_dec  = ol::allocate_tensor_dram(*dev, kStDec * kHt);
    auto T_ynorm_dec = ol::allocate_tensor_dram(*dev, kStDec * kHt);

    auto T_gate_dec   = ol::allocate_tensor_dram(*dev, kStDec * kFFt);
    auto T_up_dec     = ol::allocate_tensor_dram(*dev, kStDec * kFFt);
    auto T_silu_dec   = ol::allocate_tensor_dram(*dev, kStDec * kFFt);
    auto T_down_dec   = ol::allocate_tensor_dram(*dev, kStDec * kHt);
    auto T_normed_dec = ol::allocate_tensor_dram(*dev, kStDec * kHt);

    // -----------------------------------------------------------------------
    // Per-layer KV caches.
    // -----------------------------------------------------------------------
    std::vector<ol::TensorDesc> T_Kt_cache(kNumLayers);
    std::vector<ol::TensorDesc> T_V_cache (kNumLayers);
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        T_Kt_cache[li] = ol::allocate_tensor_dram(*dev, kStKvDec * kNkDt);  // slot-major [StKv, Nk]
        T_V_cache[li]  = ol::allocate_tensor_dram(*dev, kStKvDec * kNkDt);  // slot-major [StKv, Nk]
    }
    // gqa_decode masks decode slots after exp() so garbage rows in K^T / V
    // would become NaN; pre:kv_snapshot writes prefill slots AND zeros the
    // decode slots in one dispatch per layer.

    // -----------------------------------------------------------------------
    // Dispatch helpers.
    // -----------------------------------------------------------------------
    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run1 = [&](const char* tag, auto factory) {
        auto t0 = Clock::now();
        auto op = factory();
        ol::execute(*dev, op);
        tt::foil::release_kernels(*dev, core);
        tt::foil::reset_l1(*dev, core);
        g_prof.add(tag, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };
    auto run1_on = [&](const char* tag, tt::foil::CoreCoord c, auto factory) {
        auto t0 = Clock::now();
        auto op = factory();
        ol::execute(*dev, op);
        tt::foil::release_kernels(*dev, c);
        tt::foil::reset_l1(*dev, c);
        g_prof.add(tag, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };
    // Multi-core matmul via OpCache: first call on a (Kt, n_cores,
    // first_core) key builds + pins; subsequent calls are RTA refresh +
    // dispatch. No release_kernels / reset_l1 — they'd unpin the kernel.
    auto run_matmul_on = [&](const char* tag,
                              const std::vector<tt::foil::CoreCoord>& grid,
                              const ol::TensorDesc& a,
                              const ol::TensorDesc& b,
                              ol::TensorDesc& out,
                              uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        auto t0 = Clock::now();
        // Weight-stationary block height for prefill (Mt>1): cache as many
        // A-rows as cb_a (mb_max*Kt) + cb_b + cb_out fit in the L1 user arena,
        // capped at Mt. The Blackhole Tensix DEFAULT_UNRESERVED arena is
        // MEM_L1_SIZE(1536 KB) - DU_base(108.5 KB) = 1427.5 KB = 713 tiles;
        // we budget 700 (≈26 KB safety margin). These matmul grids are pinned
        // and dedicated (one matmul's CBs resident per core), so the whole
        // arena is available. cb_b is double-buffered (2*Kt) when 6*Kt+2 ≤ 855
        // (matmul.cpp threshold), else single (Kt). Result: Kt=64 → mb=8,
        // Kt=192 (ffn_down) → mb=2 (was 4 / 1 under the old 427-tile budget;
        // ffn_d weight re-reads halve → -44%). Decode (Mt=1) stays mb=1 ==
        // the original per-mt-row matmul, byte-identical.
        uint32_t mb_max = 1;
        if (Mt > 1) {
            const uint32_t budget = 700;  // L1 user tiles (713 real, 13 spare)
            const uint32_t cb_b = ((6 * Kt + 2) <= 855) ? (2 * Kt) : Kt;
            uint32_t cand = (cb_b + 2 < budget) ? (budget - cb_b - 2) / Kt : 1;
            if (cand < 1) cand = 1;
            mb_max = std::min(cand, Mt);
        }
        auto op = ol::make_matmul_grid_cached(*dev, a, b, out, Mt, Kt, Nt, grid, "", mb_max);
        ol::execute(*dev, op);
        g_prof.add(tag, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };
    // Route by Kt: Kt=64 (qkv/o/ffn_gate+up) → kMatmulKt64Grid;
    // Kt=192 (ffn_down) → kMatmulKt192Grid.
    auto run_matmul = [&](const char* tag,
                           const ol::TensorDesc& a, const ol::TensorDesc& b,
                           ol::TensorDesc& out,
                           uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        const auto& grid = (Kt == kHt) ? kMatmulKt64Grid : kMatmulKt192Grid;
        run_matmul_on(tag, grid, a, b, out, Mt, Kt, Nt);
    };
    auto run_matmul_lmhead = [&](const char* tag,
                                  const ol::TensorDesc& a, const ol::TensorDesc& b,
                                  ol::TensorDesc& out,
                                  uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        run_matmul_on(tag, kLmHeadGrid, a, b, out, Mt, Kt, Nt);
    };

    auto rope_tile = [&](const std::vector<uint16_t>& row) {
        std::vector<uint16_t> rm(static_cast<size_t>(kTileH) * kHalf, 0);
        for (uint32_t c = 0; c < kHalf; ++c) rm[c] = row[c];
        return tile2d(rm, kTileH, kHalf);
    };
    auto build_mask_tile = [&](uint32_t valid_cols) {
        std::vector<uint16_t> rm(static_cast<size_t>(kTileH) * (kStKvDec * kTileW), 0);
        for (uint32_t c = 0; c < valid_cols; ++c) rm[c] = one_bf16;
        return tile2d(rm, kTileH, kStKvDec * kTileW);
    };

    // -----------------------------------------------------------------------
    // Layer weights — parallel pipeline.
    // -----------------------------------------------------------------------
    std::fprintf(stderr, "loading %u layers (parallel pipeline)...\n", kNumLayers);
    std::vector<LayerW> layers(kNumLayers);
    {
        TIMED("weights:load+upload(28L)");
        constexpr uint32_t kWorkers  = 16;
        constexpr uint32_t kInFlight = 18;
        std::counting_semaphore<kInFlight> slots{kInFlight};
        std::vector<std::promise<TiledLayer>> proms(kNumLayers);
        std::vector<std::future<TiledLayer>>  futs(kNumLayers);
        for (uint32_t i = 0; i < kNumLayers; ++i) futs[i] = proms[i].get_future();
        std::atomic<uint32_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(kWorkers);
        for (uint32_t w = 0; w < kWorkers; ++w) {
            workers.emplace_back([&]{
                for (;;) {
                    uint32_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= kNumLayers) return;
                    slots.acquire();
                    try {
                        proms[i].set_value(prepare_layer(root + "/layer" + std::to_string(i)));
                    } catch (...) {
                        proms[i].set_exception(std::current_exception());
                    }
                }
            });
        }
        for (uint32_t i = 0; i < kNumLayers; ++i) {
            auto tl = futs[i].get();
            slots.release();
            layers[i] = upload_layer(*dev, tl);
        }
        for (auto& t : workers) t.join();
    }

    // -----------------------------------------------------------------------
    // Prefill
    // -----------------------------------------------------------------------
    std::fprintf(stderr, "prefill: %u tokens, %u layers\n", kS, kNumLayers);
    const auto prefill_t0 = Clock::now();
    {
    TIMED("prefill:total");

    // Prime T_xnorm1 from layer-0 ln1g before the layer loop.
    // Prefill ops use dedicated cores ({2,X}) to avoid pinning a second
    // shape on the decode cached core or on the matmul grid (which would
    // overflow L1 — each op pins ~500 KB-1 MB of CB buffers).
    run1_on("pre:rmsnorm", kPreRmsCore, [&] {
        return ol::make_rmsnorm(*dev, T_pre_in, layers[0].ln1g, T_xnorm1, kSt, kHt, kEps, kPreRmsCore);
    });

    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::fprintf(stderr, "  prefill layer %u...\n", li);

        // Separate Q/K/V matmuls — fused matmul output interleaves rows for
        // Mt>1, which breaks the offset-view slicing of QKV (see LayerW comment).
        run_matmul("pre:matmul_q", T_xnorm1, w.Wq, T_Q_pre, kSt, kHt, kNqDt);
        run_matmul("pre:matmul_k", T_xnorm1, w.Wk, T_K_pre, kSt, kHt, kNkDt);
        run_matmul("pre:matmul_v", T_xnorm1, w.Wv, T_V_pre, kSt, kHt, kNkDt);

        // RMSNorm + RoPE on Q and K.
        run1_on("pre:rmsnorm_rope_q", kPreRopeQCore, [&] {
            return ol::make_rmsnorm_rope(*dev, T_Q_pre, w.qng, T_cos, T_sin_pre, T_Qr_pre,
                                         kSt, kNumQ, kDtHalf, kEps, kPreRopeQCore);
        });
        run1_on("pre:rmsnorm_rope_k", kPreRopeKCore, [&] {
            return ol::make_rmsnorm_rope(*dev, T_K_pre, w.kng, T_cos, T_sin_pre, T_Kr_pre,
                                         kSt, kNumKv, kDtHalf, kEps, kPreRopeKCore);
        });

        // Transpose K for gqa_fused input (block-major [kNkDt, kSt]).
        run1_on("pre:transpose", kPreTransposeCore, [&] {
            return ol::make_transpose_2d(*dev, T_Kr_pre, T_Kt_pre, kSt, kNkDt, kPreTransposeCore);
        });

        // KV cache snapshot — re-position the prefill K^T (block-major from
        // pre:transpose) and V (already slot-major) into the per-layer
        // slot-major caches, then zero the decode slots. Done entirely on
        // device (28 layers × 1 dispatch) replacing 28 layers × 8 small
        // host PCIe transactions per layer.
        run1_on("pre:kv_snapshot", kPreKvSnapCore, [&] {
            return ol::make_kv_snapshot(*dev, T_Kt_pre, T_V_pre,
                                        T_Kt_cache[li], T_V_cache[li],
                                        kSt, kNkDt, kStKvDec, kPreKvSnapCore);
        });

        // Fused GQA attention.
        run1_on("pre:gqa_fused", kPreGqaCore, [&] {
            return ol::make_gqa_fused(*dev, T_Qr_pre, T_Kt_pre, T_V_pre, T_mask, T_attn_pre,
                                      kSt, kDt, kNumQ, kNumKv, kPreGqaCore);
        });

        // O projection.
        run_matmul("pre:matmul_o", T_attn_pre, w.Wo, T_proj_pre, kSt, kNqDt, kHt);

        // Residual add + post-attention RMSNorm.
        run1_on("pre:add_rmsnorm", kPreAddRmsCore, [&] {
            return ol::make_add_rmsnorm(*dev, T_pre_in, T_proj_pre, w.ln2g,
                                         T_xmid_pre, T_ynorm_pre, kSt, kHt, kEps, kPreAddRmsCore);
        });

        // Separate gate / up matmuls (same row-interleave reason as QKV).
        run_matmul("pre:matmul_gate", T_ynorm_pre, w.Wgate, T_gate_pre, kSt, kHt, kFFt);
        run_matmul("pre:matmul_up",   T_ynorm_pre, w.Wup,   T_up_pre,   kSt, kHt, kFFt);

        // SiLU * up.
        run1_on("pre:silu_mul", kPreSiluCore, [&] {
            return ol::make_silu_mul(*dev, T_gate_pre, T_up_pre, T_silu_pre, kPreSiluCore);
        });

        // Down projection.
        run_matmul("pre:matmul_ffn_d", T_silu_pre, w.Wdown, T_down_pre, kSt, kFFt, kHt);

        // Residual add + (optionally) next layer's ln1g.
        if (li + 1 < kNumLayers) {
            const LayerW& wn = layers[li + 1];
            run1_on("pre:add_rmsnorm", kPreAddRmsCore, [&] {
                return ol::make_add_rmsnorm(*dev, T_xmid_pre, T_down_pre, wn.ln1g,
                                             T_pre_out, T_xnorm1, kSt, kHt, kEps, kPreAddRmsCore);
            });
        } else {
            // Last layer: plain add; final RMSNorm uses T_final_g.
            run1_on("pre:add", kPreAddCore, [&] {
                return ol::make_eltwise_add(*dev, T_xmid_pre, T_down_pre, T_pre_out, kPreAddCore);
            });
        }
        std::swap(T_pre_in, T_pre_out);
    }
    }  // TIMED prefill:total

    // Final norm + lm_head on last row only.
    run1_on("pre:final_rmsnorm", kPreRmsCore, [&] {
        return ol::make_rmsnorm(*dev, T_pre_in, T_final_g, T_normed_pre, kSt, kHt, kEps, kPreRmsCore);
    });
    run_matmul_lmhead("pre:lm_head", T_normed_last, T_W_lm, T_logits, kStDec, kHt, kVt);

    // Prefill argmax: the prediction point is the LAST sequence row of
    // T_logits, matching qwen3_run's `argmax_row(kS - 1)` semantics.
    // lm_head ran with Mt=kStDec=1 (one tile-row), so the last sequence
    // row is row 31 within the single output tile. Device argmax handles
    // the row offset directly — replaces a 9.4 MB tile readback +
    // host-side argmax over kV = 151,936 BF16 elements.
    uint32_t cur_token = 0;
    run1("pre:argmax", [&] {
        return ol::make_argmax_row0(*dev, T_logits, kVt, T_argmax, core,
                                    /*kernel_dir=*/"", /*row_in_tile=*/kTileH - 1);
    });
    tt::foil::read_buffer(*dev, *T_argmax.buf, &cur_token, 4);
    std::fprintf(stderr, "  first generated token = %u\n", cur_token);
    std::printf("%u\n", cur_token);
    std::fflush(stdout);
    const auto first_token_t = Clock::now();

    // -----------------------------------------------------------------------
    // Decode loop
    // -----------------------------------------------------------------------
    uint32_t decode_tokens_emitted = 0;
    const auto decode_loop_t0 = Clock::now();
    {
    TIMED("decode:total");
    for (uint32_t t = 0; t < kNumDecode; ++t) {
        const uint32_t pos      = kS + t;
        const uint32_t valid_kv = pos + 1;
        if (pos >= kStKvDec * kTileH) {
            std::fprintf(stderr, "  decode beyond cache capacity, stopping\n");
            break;
        }
        std::fprintf(stderr, "  decode step %u: pos=%u token=%u\n", t, pos, cur_token);

        // Embed the new token on host and upload to T_dec_in.
        {
            const uint32_t safe_tid = std::min(cur_token, kV - 1);
            std::vector<uint16_t> row(kH);
            std::copy(embed_table.begin() + static_cast<size_t>(safe_tid) * kH,
                      embed_table.begin() + static_cast<size_t>(safe_tid + 1) * kH,
                      row.begin());
            std::vector<uint16_t> rm(static_cast<size_t>(kTileH) * kH, 0);
            std::copy(row.begin(), row.end(), rm.begin());
            upload(T_dec_in, tile2d(rm, kTileH, kH));
        }

        auto [cos_row, sin_row] = rope_at(pos);
        upload(T_dcos,  rope_tile(cos_row));
        upload(T_dsin,  rope_tile(sin_row));
        upload(T_dmask, build_mask_tile(valid_kv));

        // Prime T_dec_xnorm with layer-0 ln1g.
        run1_on("dec:rmsnorm", kCachedCore, [&] {
            return ol::make_rmsnorm(*dev, T_dec_in, layers[0].ln1g, T_dec_xnorm,
                                    kStDec, kHt, kEps, kCachedCore);
        });

        for (uint32_t li = 0; li < kNumLayers; ++li) {
            const LayerW& w = layers[li];

            run_matmul("dec:matmul_q", T_dec_xnorm, w.Wq, T_Q_dec, kStDec, kHt, kNqDt);
            run_matmul("dec:matmul_k", T_dec_xnorm, w.Wk, T_K_dec, kStDec, kHt, kNkDt);
            run_matmul("dec:matmul_v", T_dec_xnorm, w.Wv, T_V_dec, kStDec, kHt, kNkDt);

            run1_on("dec:rmsnorm_rope_q", kCachedCore, [&] {
                return ol::make_rmsnorm_rope(*dev, T_Q_dec, w.qng, T_dcos, T_dsin, T_Qr_dec,
                                             kStDec, kNumQ, kDtHalf, kEps, kCachedCore);
            });
            run1_on("dec:rmsnorm_rope_k", kCachedCore, [&] {
                return ol::make_rmsnorm_rope(*dev, T_K_dec, w.kng, T_dcos, T_dsin, T_Kr_dec,
                                             kStDec, kNumKv, kDtHalf, kEps, kCachedCore);
            });

            // Append new K/V to the cache.
            // slot1_r must encode the decode step offset such that
            //   slot = 1 + slot1_r / 32 >= kSt  (decoding starts AFTER all prefill slots).
            // Using slot1_r = (kSt - 1) * 32 + t:
            //   step 0 → slot = 1 + (kSt-1) = kSt  ✓
            {
                const uint32_t kv_slot1_r = (kSt - 1) * kTileH + t;
                run1_on("dec:kv_append", kCachedKvCore, [&] {
                    return ol::make_kv_append(*dev, T_Kr_dec, T_V_dec,
                                              T_Kt_cache[li], T_V_cache[li],
                                              kv_slot1_r, kNkDt, kStKvDec, kCachedKvCore);
                });
            }

            run1_on("dec:gqa_decode", kCachedGqaCore, [&] {
                return ol::make_gqa_decode(*dev, T_Qr_dec, T_Kt_cache[li], T_V_cache[li],
                                           T_dmask, T_attn_dec,
                                           kStDec, kStKvDec, kDt, kNumQ, kNumKv, kCachedGqaCore);
            });

            run_matmul("dec:matmul_o", T_attn_dec, w.Wo, T_proj_dec, kStDec, kNqDt, kHt);

            run1_on("dec:add_rmsnorm", kCachedAddRmsCore, [&] {
                return ol::make_add_rmsnorm(*dev, T_dec_in, T_proj_dec, w.ln2g,
                                             T_xmid_dec, T_ynorm_dec, kStDec, kHt, kEps, kCachedAddRmsCore);
            });

            run_matmul("dec:matmul_gate", T_ynorm_dec, w.Wgate, T_gate_dec, kStDec, kHt, kFFt);
            run_matmul("dec:matmul_up",   T_ynorm_dec, w.Wup,   T_up_dec,   kStDec, kHt, kFFt);

            run1_on("dec:silu_mul", kCachedSiluCore, [&] {
                return ol::make_silu_mul(*dev, T_gate_dec, T_up_dec, T_silu_dec, kCachedSiluCore);
            });

            run_matmul("dec:matmul_ffn_d", T_silu_dec, w.Wdown, T_down_dec, kStDec, kFFt, kHt);

            if (li + 1 < kNumLayers) {
                const LayerW& wn = layers[li + 1];
                run1_on("dec:add_rmsnorm", kCachedAddRmsCore, [&] {
                    return ol::make_add_rmsnorm(*dev, T_xmid_dec, T_down_dec, wn.ln1g,
                                                 T_dec_out, T_dec_xnorm, kStDec, kHt, kEps, kCachedAddRmsCore);
                });
            } else {
                run1("dec:add", [&] {
                    return ol::make_eltwise_add(*dev, T_xmid_dec, T_down_dec, T_dec_out);
                });
            }
            std::swap(T_dec_in, T_dec_out);
        }

        run1_on("dec:final_rmsnorm", kCachedCore, [&] {
            return ol::make_rmsnorm(*dev, T_dec_in, T_final_g, T_normed_dec,
                                    kStDec, kHt, kEps, kCachedCore);
        });
        run_matmul_lmhead("dec:lm_head", T_normed_dec, T_W_lm, T_logits, kStDec, kHt, kVt);
        run1("dec:argmax", [&] {
            return ol::make_argmax_row0(*dev, T_logits, kVt, T_argmax, core);
        });
        uint32_t nxt = 0;
        tt::foil::read_buffer(*dev, *T_argmax.buf, &nxt, 4);
        std::printf("%u\n", nxt);
        std::fflush(stdout);
        cur_token = nxt;
        ++decode_tokens_emitted;
    }
    }  // TIMED decode:total
    const auto decode_loop_end = Clock::now();

    g_prof.report();
    if (fd_owner) {
        fd_owner->push_terminate();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        dev->fast_dispatch = nullptr;
    }
    tt::foil::close_device(std::move(dev));

    const double wall_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wall_t0).count();
    std::fprintf(stderr, "\n=== real_wall ===\n  wall_ms %.1f  (= %.2f s)\n",
                 wall_ms, wall_ms / 1000.0);

    const double ttft_cold_ms = std::chrono::duration<double, std::milli>(
        first_token_t - wall_t0).count();
    const double ttft_warm_ms = std::chrono::duration<double, std::milli>(
        first_token_t - prefill_t0).count();
    const double decode_loop_ms = std::chrono::duration<double, std::milli>(
        decode_loop_end - decode_loop_t0).count();
    const uint32_t generated_tokens = 1 + decode_tokens_emitted;
    std::fprintf(stderr, "\n=== llm_perf ===\n");
    std::fprintf(stderr, "  generated_tokens   %u  (= 1 prefill + %u decode)\n",
                 generated_tokens, decode_tokens_emitted);
    std::fprintf(stderr, "  ttft_cold_ms       %.1f\n", ttft_cold_ms);
    std::fprintf(stderr, "  ttft_warm_ms       %.1f\n", ttft_warm_ms);
    double itl_ms = 0.0, decode_tps = 0.0;
    if (decode_tokens_emitted > 0) {
        itl_ms     = decode_loop_ms / decode_tokens_emitted;  // inter-token latency
        decode_tps = 1000.0 / itl_ms;
        std::fprintf(stderr, "  decode_loop_ms     %.1f  (%u tokens)\n",
                     decode_loop_ms, decode_tokens_emitted);
        std::fprintf(stderr, "  tpot_ms            %.2f\n", itl_ms);
        std::fprintf(stderr, "  tokens_per_sec     %.2f\n", decode_tps);
    }

    // -----------------------------------------------------------------
    // perf_metrics: the six headline numbers mirrored in README.md.
    // Prefill throughput uses the warm window (prefill start → first
    // token) over the kS prompt tokens — this is the variable-length
    // path, so it scales with prompt size. Cores utilization is the
    // static footprint vs the Blackhole 14x10 = 140 Tensix worker grid.
    // -----------------------------------------------------------------
    constexpr uint32_t kWorkerGrid = 14 * 10;  // Blackhole functional workers
    const double prefill_tps =
        ttft_warm_ms > 0.0 ? (double)kS / (ttft_warm_ms / 1000.0) : 0.0;
    const double core_util = 100.0 * (double)boot_cores.size() / (double)kWorkerGrid;
    std::fprintf(stderr, "\n=== perf_metrics ===\n");
    std::fprintf(stderr, "  prefill_tokens_per_sec   %8.1f  (%u tokens / %.1f ms)\n",
                 prefill_tps, kS, ttft_warm_ms);
    std::fprintf(stderr, "  decode_tokens_per_sec    %8.2f\n", decode_tps);
    std::fprintf(stderr, "  ttft_ms                  %8.1f warm / %.1f cold\n",
                 ttft_warm_ms, ttft_cold_ms);
    std::fprintf(stderr, "  end_to_end_latency_ms    %8.1f  (= %.2f s)\n",
                 wall_ms, wall_ms / 1000.0);
    std::fprintf(stderr, "  inter_token_latency_ms   %8.2f\n", itl_ms);
    std::fprintf(stderr, "  cores_utilization        %7.1f%%  (%zu booted / %u worker grid; matmul 4-8 active)\n",
                 core_util, boot_cores.size(), kWorkerGrid);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qwen3vl_run: FAIL — %s\n", e.what());
    return 1;
}
