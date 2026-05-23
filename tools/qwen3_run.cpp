// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-only Qwen3-VL-2B inference CLI. Loads weights from
//   $TT_FOIL_QWEN3_DATA/{model,layer0..layerN-1}/, runs prefill on the
// given prompt token-IDs (a binary file of uint32[seq]) and N greedy
// decode steps, and prints the generated token-IDs to stdout (one per
// line). No numpy / golden / tokenizer dependency — pair with
// tools/qwen3_chat.py for prompt → tokens → text round-tripping.
//
// Usage:
//   TT_FOIL_QWEN3_DATA=data/qwen3_vl_2b TT_FOIL_OPS_DIR=ops \
//   TT_FOIL_DEVICE=0 ./build/tools/qwen3_run <prompt_ids.bin> <num_decode>
//
// Required env:
//   TT_FOIL_QWEN3_DATA  — root of exported weights
//   TT_FOIL_OPS_DIR     — root of prebuilt kernels (ops/)
//   TT_FOIL_DEVICE      — PCIe chip index (optional, default 0)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

constexpr uint32_t kNumLayers = 28;
constexpr uint32_t kS         = 32;
constexpr uint32_t kH         = 2048;
constexpr uint32_t kV         = 151936;
constexpr uint32_t kFFN       = 6144;
constexpr uint32_t kNumQ      = 16;
constexpr uint32_t kNumKv     = 8;
constexpr uint32_t kHeadDim   = 128;
constexpr float    kRopeTheta = 5000000.0f;

constexpr uint32_t kSt      = kS       / kTileH;   // 1
constexpr uint32_t kStDec   = 1;
constexpr uint32_t kStKvDec = 2;
constexpr uint32_t kHt      = kH       / kTileW;
constexpr uint32_t kFFt     = kFFN     / kTileW;
constexpr uint32_t kDt      = kHeadDim / kTileW;
constexpr uint32_t kDtHalf  = kDt / 2;
constexpr uint32_t kNqDt    = kNumQ  * kDt;
constexpr uint32_t kNkDt    = kNumKv * kDt;
constexpr uint32_t kVt      = kV / kTileW;
constexpr float    kEps     = 1e-6f;

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
            std::fprintf(stderr, "  %-32s %10.1f ms   (%6lu calls, %7.2f ms/call, %5.1f%%)\n",
                k.c_str(), e.first, (unsigned long)e.second,
                e.first / std::max<uint64_t>(1, e.second),
                100.0 * e.first / std::max(1e-9, total));
        std::fprintf(stderr, "  %-32s %10.1f ms\n", "TOTAL", total);
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

std::vector<uint16_t> load_bin(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + p);
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    if (f.gcount() != static_cast<std::streamsize>(n * 2))
        throw std::runtime_error("short read: " + p);
    return v;
}
std::vector<uint32_t> load_u32(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + p);
    std::vector<uint32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    if (f.gcount() != static_cast<std::streamsize>(n * 4))
        throw std::runtime_error("short read: " + p);
    return v;
}

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

// iter7: Wq, Wk, Wv are fused into a single Wqkv weight along the Nt axis.
// Layout (kt-row major, like every other tile-format matrix in this file):
//   for each kt in [0, kHt):
//     Wq tiles  [kt, 0..kNqDt)
//     Wk tiles  [kt, kNqDt..kNqDt+kNkDt)
//     Wv tiles  [kt, kNqDt+kNkDt..kNqDt+2*kNkDt)
// So one matmul(A=hidden, B=Wqkv, Mt=1, Kt=kHt, Nt=kNqkvDt) produces a
// single QKV-concat tensor; downstream T_Q/T_K/T_V are zero-copy
// offset views into it.
constexpr uint32_t kNqkvDt   = kNqDt + 2 * kNkDt;  // 64 + 32 + 32 = 128
constexpr uint32_t kFFtFused = 2 * kFFt;           // gate + up

struct LayerW {
    tt::foil::op_lib::TensorDesc ln1g, ln2g, qng, kng;
    tt::foil::op_lib::TensorDesc Wqkv, Wo;
    // iter8: Wgate + Wup are fused along Nt into Wgateup. Wdown is
    // separate because it operates on the post-silu * up product.
    tt::foil::op_lib::TensorDesc Wgateup, Wdown;
};

// CPU-only stage: load 11 .bin files from disk and tile-ize them. No
// device interaction so safe to run on worker threads in parallel.
struct TiledLayer {
    std::vector<uint16_t> ln1g, ln2g, qng, kng;
    std::vector<uint16_t> Wqkv, Wo;          // Wqkv = concat(Wq, Wk, Wv) along Nt
    std::vector<uint16_t> Wgateup, Wdown;    // Wgateup = concat(Wgate, Wup) along Nt
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
    T.ln1g  = gamma_to_tiles(ln1g, kH);
    T.ln2g  = gamma_to_tiles(ln2g, kH);
    T.qng   = gamma_to_tiles(qng,  kHeadDim);
    T.kng   = gamma_to_tiles(kng,  kHeadDim);
    auto Wq_t = tile2d(Wq, kH, kNumQ  * kHeadDim);
    auto Wk_t = tile2d(Wk, kH, kNumKv * kHeadDim);
    auto Wv_t = tile2d(Wv, kH, kNumKv * kHeadDim);
    // Concat Wq | Wk | Wv along Nt, kt-row by kt-row. The per-row slice
    // counts come straight from each weight's column tile count.
    T.Wqkv.reserve(static_cast<size_t>(kHt) * kNqkvDt * kTileWords);
    for (uint32_t kt = 0; kt < kHt; ++kt) {
        auto append = [&](const std::vector<uint16_t>& src, uint32_t cols_per_row) {
            const size_t row_words = static_cast<size_t>(cols_per_row) * kTileWords;
            const size_t off = static_cast<size_t>(kt) * row_words;
            T.Wqkv.insert(T.Wqkv.end(), src.begin() + off, src.begin() + off + row_words);
        };
        append(Wq_t, kNqDt);
        append(Wk_t, kNkDt);
        append(Wv_t, kNkDt);
    }
    T.Wo    = tile2d(Wo,    kNumQ * kHeadDim, kH);
    // Concat Wgate | Wup along Nt, kt-row by kt-row. Same pattern as Wqkv.
    auto Wgate_t = tile2d(Wgate, kH, kFFN);
    auto Wup_t   = tile2d(Wup,   kH, kFFN);
    T.Wgateup.reserve(static_cast<size_t>(kHt) * kFFtFused * kTileWords);
    for (uint32_t kt = 0; kt < kHt; ++kt) {
        const size_t row_words_ff = static_cast<size_t>(kFFt) * kTileWords;
        const size_t off = static_cast<size_t>(kt) * row_words_ff;
        T.Wgateup.insert(T.Wgateup.end(), Wgate_t.begin() + off, Wgate_t.begin() + off + row_words_ff);
        T.Wgateup.insert(T.Wgateup.end(), Wup_t.begin()   + off, Wup_t.begin()   + off + row_words_ff);
    }
    T.Wdown = tile2d(Wdown, kFFN, kH);
    return T;
}

// Device-side: allocate DRAM tensors and upload bytes. Single-threaded
// (DRAM bump-allocator + UMD write_to_device not thread-safe).
LayerW upload_layer(tt::foil::Device& dev, const TiledLayer& T) {
    namespace ol = tt::foil::op_lib;
    LayerW L;
    L.ln1g  = ol::allocate_tensor_dram(dev, kHt);
    L.ln2g  = ol::allocate_tensor_dram(dev, kHt);
    L.qng   = ol::allocate_tensor_dram(dev, kDt);
    L.kng   = ol::allocate_tensor_dram(dev, kDt);
    L.Wqkv  = ol::allocate_tensor_dram(dev, kHt * kNqkvDt);
    L.Wo    = ol::allocate_tensor_dram(dev, kNqDt * kHt);
    L.Wgateup = ol::allocate_tensor_dram(dev, kHt * kFFtFused);
    L.Wdown   = ol::allocate_tensor_dram(dev, kFFt * kHt);
    auto up = [&](auto& t, const std::vector<uint16_t>& d) {
        tt::foil::write_buffer(dev, *t.buf, d.data(), d.size() * 2);
    };
    up(L.ln1g,  T.ln1g);
    up(L.ln2g,  T.ln2g);
    up(L.qng,   T.qng);
    up(L.kng,   T.kng);
    up(L.Wqkv,  T.Wqkv);
    up(L.Wo,      T.Wo);
    up(L.Wgateup, T.Wgateup);
    up(L.Wdown,   T.Wdown);
    return L;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: qwen3_run <prompt_ids.bin (uint32[32])> <num_decode>\n"
            "  env: TT_FOIL_QWEN3_DATA TT_FOIL_OPS_DIR [TT_FOIL_DEVICE]\n");
        return 2;
    }
    const std::string prompt_path = argv[1];
    const uint32_t kNumDecode = std::stoul(argv[2]);
    if (kS + kNumDecode > kStKvDec * kTileH)
        throw std::runtime_error("num_decode > 32 would overflow St_kv=2 cache");

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;
    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    if (!data_env) throw std::runtime_error("set TT_FOIL_QWEN3_DATA");
    const std::string root = data_env;
    const std::string mdir = root + "/model";

    namespace ol = tt::foil::op_lib;

    // -----------------------------------------------------------------
    // Inputs
    // -----------------------------------------------------------------
    auto token_ids = load_u32(prompt_path, kS);
    // Kick off the three big one-shot loads (embed_tokens 622 MB,
    // lm_head_tiled 622 MB, final_norm gamma) in background threads so
    // they overlap with host-side RoPE table generation and the 28-layer
    // parallel weight pipeline below.
    std::fprintf(stderr, "loading embed_tokens (622 MB) + lm_head (622 MB) in background...\n");
    auto fut_embed_table = std::async(std::launch::async, [&]{
        return load_bin(mdir + "/embed_tokens.bin",
                        static_cast<std::size_t>(kV) * kH);
    });
    auto fut_lmhead_tiles = std::async(std::launch::async, [&]{
        return load_bin(mdir + "/lm_head_tiled.bin",
                        static_cast<std::size_t>(kHt) * kVt * kTileWords);
    });
    auto fut_final_g_tiles = std::async(std::launch::async, [&]{
        auto g = load_bin(mdir + "/final_norm.bin", kH);
        return gamma_to_tiles(g, kH);
    });

    // RoPE host-side: build cos/sin row for any position.
    const uint32_t kHalf = kHeadDim / 2;
    auto rope_at = [&](uint32_t pos) {
        std::vector<uint16_t> c(kHalf), s(kHalf);
        for (uint32_t i = 0; i < kHalf; ++i) {
            double freq  = 1.0 / std::pow(static_cast<double>(kRopeTheta),
                                          static_cast<double>(i) / static_cast<double>(kHalf));
            double angle = static_cast<double>(pos) * freq;
            c[i] = f32_to_bf16(static_cast<float>(std::cos(angle)));
            s[i] = f32_to_bf16(static_cast<float>(std::sin(angle)));
        }
        return std::make_pair(std::move(c), std::move(s));
    };

    // Prefill cos/sin tables [kS, kHalf] row-major.
    std::vector<uint16_t> cos_rm(kS * kHalf), sin_rm(kS * kHalf);
    for (uint32_t r = 0; r < kS; ++r) {
        auto [c, s] = rope_at(r);
        std::copy(c.begin(), c.end(), cos_rm.begin() + r * kHalf);
        std::copy(s.begin(), s.end(), sin_rm.begin() + r * kHalf);
    }

    // Prefill causal mask [S, S].
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    std::vector<uint16_t> mask_rm(kS * kS, 0);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;

    auto cos_tiles = tile2d(cos_rm, kS, kHalf);
    auto sin_tiles = tile2d(sin_rm, kS, kHalf);
    auto mask_tiles = tile2d(mask_rm, kS, kS);
    auto final_g_tiles = fut_final_g_tiles.get();

    // -----------------------------------------------------------------
    // Open device, upload static tensors.
    // -----------------------------------------------------------------
    // Boot a 1×8 grid. Most matmul shapes in qwen3 are Mt=1 with modest
    // Kt/Nt, where per-core dispatch overhead (sequential ELF NOC writes
    // in dispatch_stage_setup) dominates over per-core compute. Adding
    // cores beyond 4 strictly regresses those (see iter11 lesson in
    // bench/HISTORY.md). lm_head is the exception — Nt=4748 with kVt
    // tiles of compute amortizes the dispatch cost, so it benefits from
    // 8-way sharding (113 → 64 ms/call on decode).
    //
    // kMatmulGrid (4 cores) used for qkv/o/ffn matmuls and other ops.
    // kLmHeadGrid (8 cores) used only for lm_head (Vt=4748 ragged shards).
    //   matmul_qkv  Nt=128  →  32 tile/core  (clean, 4-way)
    //   matmul_o    Nt=64   →  16 tile/core  (clean, 4-way)
    //   ffn_gateup  Nt=384  →  96 tile/core  (clean, 4-way)
    //   ffn_down    Nt=64   →  16 tile/core  (clean, 4-way)
    //   lm_head     Nt=4748 → 4×594 + 4×593  (ragged 8-way)
    const std::vector<tt::foil::CoreCoord> kMatmulGrid = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3},
    };
    const std::vector<tt::foil::CoreCoord> kLmHeadGrid = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3},
        {0, 4}, {0, 5}, {0, 6}, {0, 7},
    };
    std::vector<tt::foil::CoreCoord> boot_cores = kLmHeadGrid;
    auto dev = tt::foil::open_device(pcie_index, "", boot_cores);
    tt::foil::CoreCoord core{0, 0};


    ol::TensorDesc T_embed_table;
    T_embed_table.buf = tt::foil::allocate_buffer(
        *dev, tt::foil::BufferLocation::DRAM,
        static_cast<std::size_t>(kV) * kH * 2);
    T_embed_table.num_tiles = 0;
    {
        auto embed_table = fut_embed_table.get();
        tt::foil::write_buffer(*dev, *T_embed_table.buf, embed_table.data(),
                               static_cast<std::size_t>(kV) * kH * 2);
    }

    auto T_W_lm = ol::allocate_tensor_dram(*dev, kHt * kVt);
    {
        auto lmhead_tiles = fut_lmhead_tiles.get();
        tt::foil::write_buffer(*dev, *T_W_lm.buf, lmhead_tiles.data(),
                               lmhead_tiles.size() * 2);
    }

    auto T_cos     = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_sin     = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_mask    = ol::allocate_tensor_dram(*dev, kSt * kSt);
    auto T_final_g = ol::allocate_tensor_dram(*dev, kHt);
    tt::foil::write_buffer(*dev, *T_cos.buf,     cos_tiles.data(),    cos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_sin.buf,     sin_tiles.data(),    sin_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_mask.buf,    mask_tiles.data(),   mask_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_final_g.buf, final_g_tiles.data(), final_g_tiles.size() * 2);

    auto T_dcos  = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dsin  = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dmask = ol::allocate_tensor_dram(*dev, kStDec * kStKvDec);

    auto T_embed_rm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_layer_in  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_layer_out = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xnorm1    = ol::allocate_tensor_dram(*dev, kSt * kHt);
    // Fused QKV output: one big buffer that the matmul writes into,
    // plus three zero-copy offset views (T_Q, T_K, T_V) into that buffer
    // for the downstream rmsnorm/rope/gqa to consume.
    auto T_QKV = ol::allocate_tensor_dram(*dev, kSt * kNqkvDt);
    auto qkv_base = T_QKV.buf->device_addr;
    auto make_view = [&](uint32_t tile_offset, uint32_t num_tiles_view) {
        ol::TensorDesc v;
        v.buf = std::make_shared<tt::foil::Buffer>();
        v.buf->location    = tt::foil::BufferLocation::DRAM;
        v.buf->device_addr = qkv_base + static_cast<uint64_t>(tile_offset) * kTileBytes;
        v.buf->size_bytes  = static_cast<std::size_t>(num_tiles_view) * kTileBytes;
        v.num_tiles        = num_tiles_view;
        return v;
    };
    auto T_Q = make_view(0,                 kSt * kNqDt);
    auto T_K = make_view(kSt * kNqDt,       kSt * kNkDt);
    auto T_V = make_view(kSt * (kNqDt + kNkDt), kSt * kNkDt);
    auto T_Qn = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn  = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kt    = ol::allocate_tensor_dram(*dev, kNkDt * kSt);
    auto T_proj  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm = ol::allocate_tensor_dram(*dev, kSt * kHt);
    // Fused gate+up matmul output buffer; T_gate and T_up are zero-copy
    // offset views into it (same trick as T_QKV).
    auto T_gateup = ol::allocate_tensor_dram(*dev, kSt * kFFtFused);
    auto gateup_base = T_gateup.buf->device_addr;
    auto make_ff_view = [&](uint32_t tile_offset, uint32_t num_tiles_view) {
        ol::TensorDesc v;
        v.buf = std::make_shared<tt::foil::Buffer>();
        v.buf->location    = tt::foil::BufferLocation::DRAM;
        v.buf->device_addr = gateup_base + static_cast<uint64_t>(tile_offset) * kTileBytes;
        v.buf->size_bytes  = static_cast<std::size_t>(num_tiles_view) * kTileBytes;
        v.num_tiles        = num_tiles_view;
        return v;
    };
    auto T_gate  = make_ff_view(0,                 kSt * kFFt);
    auto T_up    = make_ff_view(kSt * kFFt,        kSt * kFFt);
    auto T_silu  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_normed = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_logits = ol::allocate_tensor_dram(*dev, kSt * kVt);
    auto T_dembed_rm = ol::allocate_tensor_dram(*dev, kHt);

    // Per-layer device-resident KV caches.
    std::vector<ol::TensorDesc> T_Kt_cache(kNumLayers);
    std::vector<ol::TensorDesc> T_V_cache (kNumLayers);
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        T_Kt_cache[li] = ol::allocate_tensor_dram(*dev, kNkDt * kStKvDec);
        T_V_cache[li]  = ol::allocate_tensor_dram(*dev, kStKvDec * kNkDt);
    }
    // 4-byte DRAM result slot for device-side decode argmax. Padded to
    // one tile (2048 B) so the bump allocator stays tile-aligned for any
    // subsequent allocations. (Several op_lib readers/writers index into
    // DRAM at `base + tile_idx * 2048` and silently miscompute when
    // `base` isn't a multiple of 2048 — keep the bump pointer on tile
    // boundaries whenever we allocate non-tile-format DRAM.)
    ol::TensorDesc T_argmax;
    T_argmax.buf = tt::foil::allocate_buffer(
        *dev, tt::foil::BufferLocation::DRAM, /*bytes=*/2048);
    T_argmax.num_tiles = 0;

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run  = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };
    auto step = [&] { tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core); };
    auto run1 = [&](const char* tag, auto factory) {
        auto t0 = Clock::now();
        auto op = factory(); ol::execute(*dev, op);
        tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core);
        g_prof.add(tag, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };
    // Multi-core matmul: builds an N-core MatMulGridOp, dispatches via
    // dispatch_execute_multi, then releases per-core kernel-config +
    // L1 across the whole grid so the next caller starts from clean
    // L1 on every core.
    auto run_matmul_on = [&](const char* tag,
                             const std::vector<tt::foil::CoreCoord>& grid,
                             const ol::TensorDesc& a,
                             const ol::TensorDesc& b,
                             ol::TensorDesc& out,
                             uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        auto t0 = Clock::now();
        auto op = ol::make_matmul_grid(*dev, a, b, out, Mt, Kt, Nt, grid);
        ol::execute(*dev, op);
        for (const auto& c : grid) {
            tt::foil::release_kernels(*dev, c);
            tt::foil::reset_l1(*dev, c);
        }
        g_prof.add(tag, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };
    auto run_matmul_grid = [&](const char* tag,
                               const ol::TensorDesc& a,
                               const ol::TensorDesc& b,
                               ol::TensorDesc& out,
                               uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        run_matmul_on(tag, kMatmulGrid, a, b, out, Mt, Kt, Nt);
    };
    auto run_matmul_lmhead = [&](const char* tag,
                                 const ol::TensorDesc& a,
                                 const ol::TensorDesc& b,
                                 ol::TensorDesc& out,
                                 uint32_t Mt, uint32_t Kt, uint32_t Nt) {
        run_matmul_on(tag, kLmHeadGrid, a, b, out, Mt, Kt, Nt);
    };

    const uint32_t kTotalNk    = kNkDt * kTileW;
    const uint32_t kSlot1Rows  = kStKvDec * kTileH - kS;
    const std::size_t kSlot0Bytes = static_cast<std::size_t>(kNkDt) * kTileBytes;
    std::vector<std::vector<uint16_t>> cache_K_slot1_rm(kNumLayers,
        std::vector<uint16_t>(static_cast<size_t>(kSlot1Rows) * kTotalNk, 0));
    std::vector<std::vector<uint16_t>> cache_V_slot1_rm(kNumLayers,
        std::vector<uint16_t>(static_cast<size_t>(kSlot1Rows) * kTotalNk, 0));

    auto rope_tile = [&](const std::vector<uint16_t>& row) {
        std::vector<uint16_t> rm(kTileH * kHalf, 0);
        for (uint32_t c = 0; c < kHalf; ++c) rm[c] = row[c];
        return tile2d(rm, kTileH, kHalf);
    };
    auto build_mask_tile = [&](uint32_t valid_cols) {
        std::vector<uint16_t> rm(kTileH * (kStKvDec * kTileW), 0);
        for (uint32_t c = 0; c < valid_cols; ++c) rm[c] = one_bf16;
        return tile2d(rm, kTileH, kStKvDec * kTileW);
    };
    auto build_slot1_KT_tiles = [&](const std::vector<uint16_t>& slot1_K_rm) {
        std::vector<uint16_t> KT_rm(static_cast<size_t>(kTotalNk) * kSlot1Rows, 0);
        for (uint32_t r = 0; r < kSlot1Rows; ++r)
            for (uint32_t c = 0; c < kTotalNk; ++c)
                KT_rm[c * kSlot1Rows + r] = slot1_K_rm[r * kTotalNk + c];
        return tile2d(KT_rm, kTotalNk, kSlot1Rows);
    };
    auto build_slot1_V_tiles = [&](const std::vector<uint16_t>& slot1_V_rm) {
        return tile2d(slot1_V_rm, kSlot1Rows, kTotalNk);
    };

    // -----------------------------------------------------------------
    // Layer weights — load all 28 layers before building ops so that
    // make_* can use real layer-0 tensors for the initial CB sizing.
    // -----------------------------------------------------------------
    std::fprintf(stderr, "loading %u layers (parallel pipeline)...\n", kNumLayers);
    std::vector<LayerW> layers(kNumLayers);
    {
        TIMED("weights:load+upload(28L)");
        // Bounded producer/consumer pipeline:
        //   W worker threads call prepare_layer (disk read + tile2d, CPU
        //     bound, ~80 MB output per layer)
        //   main thread consumes futures in order and calls upload_layer
        //     (DRAM allocate + write_buffer, single-threaded on UMD)
        //   counting_semaphore caps RAM by limiting in-flight prepared
        //     layers to kInFlight (≈ kInFlight × 80 MB).
        constexpr uint32_t kWorkers  = 4;
        constexpr uint32_t kInFlight = 6;  // worker slots + a little queue
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
                    slots.acquire();  // wait for free RAM slot
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
            slots.release();  // free RAM slot now that we own the bytes
            layers[i] = upload_layer(*dev, tl);
        }
        for (auto& t : workers) t.join();
    }

    // -----------------------------------------------------------------
    // Embedding lookup → T_layer_in (one-shot, prefill only).
    // -----------------------------------------------------------------
    std::fprintf(stderr, "embedding lookup...\n");
    std::vector<uint32_t> tid_vec(token_ids.begin(), token_ids.end());
    run1("embed", [&] { return ol::make_embedding(*dev, T_embed_table, tid_vec, kH, T_embed_rm); });
    std::vector<uint16_t> hidden_rm(kS * kH);
    tt::foil::read_buffer(*dev, *T_embed_rm.buf, hidden_rm.data(), kS * kH * 2);
    upload(T_layer_in, tile2d(hidden_rm, kS, kH));

    // -----------------------------------------------------------------
    // Prefill — same one-shot dispatch pattern as decode. Matmul calls
    // (QKV, O, FFN gate/up/down, lm_head) use run_matmul_grid (1×4);
    // every other op stays on (0,0) via run1.
    // -----------------------------------------------------------------
    {
    TIMED("prefill:total");
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::fprintf(stderr, "  prefill layer %u …\n", li);

        run1("pre:rmsnorm",    [&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); });
        // Fused QKV matmul: A·[Wq|Wk|Wv]. Output lands in T_QKV; T_Q,
        // T_K, T_V are pre-set offset views into the same buffer.
        run_matmul_grid("pre:matmul_qkv", T_xnorm1, w.Wqkv, T_QKV, kSt, kHt, kNqkvDt);
        run1("pre:rmsnorm_qk", [&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); });
        run1("pre:rmsnorm_qk", [&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); });
        run1("pre:rope",       [&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); });
        run1("pre:rope",       [&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); });
        run1("pre:transpose",  [&] { return ol::make_transpose_2d(*dev, T_Kr, T_Kt, kSt, kNkDt); });

        {
            TIMED("pre:kv_cache_snapshot(host)");
            std::vector<uint16_t> Kr_tiles(static_cast<size_t>(kSt) * kNkDt * kTileWords);
            std::vector<uint16_t> V_tiles (static_cast<size_t>(kSt) * kNkDt * kTileWords);
            tt::foil::read_buffer(*dev, *T_Kr.buf, Kr_tiles.data(), Kr_tiles.size() * 2);
            tt::foil::read_buffer(*dev, *T_V.buf,  V_tiles.data(),  V_tiles.size()  * 2);
            auto Kr_rm = untile2d(Kr_tiles, kS, kTotalNk);
            auto V_rm  = untile2d(V_tiles,  kS, kTotalNk);
            std::vector<uint16_t> KT_slot0_rm(static_cast<size_t>(kTotalNk) * kS, 0);
            for (uint32_t r = 0; r < kS; ++r)
                for (uint32_t c = 0; c < kTotalNk; ++c)
                    KT_slot0_rm[c * kS + r] = Kr_rm[r * kTotalNk + c];
            auto KT_slot0_tiles = tile2d(KT_slot0_rm, kTotalNk, kS);
            auto V_slot0_tiles  = tile2d(V_rm,        kS, kTotalNk);
            tt::foil::write_buffer(*dev, *T_Kt_cache[li].buf, 0,
                                   KT_slot0_tiles.data(), KT_slot0_tiles.size() * 2);
            tt::foil::write_buffer(*dev, *T_V_cache[li].buf, 0,
                                   V_slot0_tiles.data(),  V_slot0_tiles.size()  * 2);
            std::vector<uint16_t> zeros(kSlot0Bytes / 2, 0);
            tt::foil::write_buffer(*dev, *T_Kt_cache[li].buf, kSlot0Bytes,
                                   zeros.data(), zeros.size() * 2);
            tt::foil::write_buffer(*dev, *T_V_cache[li].buf, kSlot0Bytes,
                                   zeros.data(), zeros.size() * 2);
        }

        run1("pre:gqa_fused",  [&] {
            return ol::make_gqa_fused(*dev, T_Qr, T_Kt, T_V, T_mask, T_attn,
                                      kSt, kDt, kNumQ, kNumKv);
        });
        run_matmul_grid("pre:matmul_o", T_attn, w.Wo, T_proj, kSt, kNqDt, kHt);
        run1("pre:add",        [&] { return ol::make_eltwise_add(*dev, T_layer_in, T_proj, T_xmid); });
        run1("pre:rmsnorm",    [&] { return ol::make_rmsnorm(*dev, T_xmid, w.ln2g, T_ynorm, kSt, kHt, kEps); });
        // Fused gate+up matmul (iter8). One dispatch instead of two; T_gate
        // and T_up are pre-set offset views into T_gateup.
        run_matmul_grid("pre:matmul_ffn", T_ynorm, w.Wgateup, T_gateup, kSt, kHt, kFFtFused);
        run1("pre:silu_mul",   [&] { return ol::make_silu_mul(*dev, T_gate, T_up, T_fused); });
        run_matmul_grid("pre:matmul_ffn", T_fused, w.Wdown, T_down, kSt, kFFt, kHt);
        run1("pre:add",        [&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_layer_out); });
        std::swap(T_layer_in, T_layer_out);
    }
    }

    // Prefill final norm + lm_head → argmax(row S-1) is the first decode input.
    std::fprintf(stderr, "  prefill final norm + lm_head ...\n");
    run1("pre:final_rmsnorm", [&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); });
    run_matmul_lmhead("pre:lm_head", T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
    std::vector<uint16_t> logits_tiles(static_cast<size_t>(kSt) * kVt * kTileWords);
    {
        TIMED("pre:logits_readback");
        tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
    }
    auto logits_rm = untile2d(logits_tiles, kS, kV);

    // Decode always continues from position kS (= just past the padded
    // prompt), so the KV cache's prefill slot 0 (positions 0..S-1) is
    // never overwritten and slot 1 grows append-only. Predictions
    // therefore implicitly "see" any endoftext pad tokens in the prompt
    // tail — for short prompts the model effectively learns to continue
    // a freshly-padded sequence rather than the prompt's exact last
    // token, which is fine for a demo.
    auto argmax_row = [&](uint32_t row) {
        float best = -1e30f; uint32_t bi = 0;
        for (uint32_t v = 0; v < kV; ++v) {
            float lv = bf16_to_f32(logits_rm[static_cast<size_t>(row) * kV + v]);
            if (lv > best) { best = lv; bi = v; }
        }
        return bi;
    };
    uint32_t cur_token = argmax_row(kS - 1);
    std::fprintf(stderr, "  first generated token = %u\n", cur_token);

    std::printf("%u\n", cur_token);
    std::fflush(stdout);

    {
    TIMED("decode:total");
    for (uint32_t t = 0; t < kNumDecode; ++t) {
        const uint32_t pos      = kS + t;
        const uint32_t valid_kv = pos + 1;
        if (pos >= kStKvDec * kTileH) {
            std::fprintf(stderr, "  decode beyond cache capacity, stopping\n");
            break;
        }
        std::fprintf(stderr, "  decode step %u: pos=%u input=%u\n", t, pos, cur_token);

        std::vector<uint32_t> dec_tid = {cur_token};
        run1("dec:embed", [&] { return ol::make_embedding(*dev, T_embed_table, dec_tid, kH, T_dembed_rm); });
        std::vector<uint16_t> dec_row(kH);
        tt::foil::read_buffer(*dev, *T_dembed_rm.buf, dec_row.data(), kH * 2);
        std::vector<uint16_t> dec_hidden_rm(kTileH * kH, 0);
        for (uint32_t c = 0; c < kH; ++c) dec_hidden_rm[c] = dec_row[c];
        upload(T_layer_in, tile2d(dec_hidden_rm, kTileH, kH));

        auto [cos_row, sin_row] = rope_at(pos);
        upload(T_dcos,  rope_tile(cos_row));
        upload(T_dsin,  rope_tile(sin_row));
        upload(T_dmask, build_mask_tile(valid_kv));

        for (uint32_t li = 0; li < kNumLayers; ++li) {
            const LayerW& w = layers[li];

            run1("dec:rmsnorm",    [&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); });
            // Fused QKV matmul (iter7) — see comments above the T_QKV
            // allocation. One dispatch instead of three.
            run_matmul_grid("dec:matmul_qkv", T_xnorm1, w.Wqkv, T_QKV, kSt, kHt, kNqkvDt);
            run1("dec:rmsnorm_qk", [&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); });
            run1("dec:rmsnorm_qk", [&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); });
            run1("dec:rope",       [&] { return ol::make_rope(*dev, T_Qn, T_dcos, T_dsin, T_Qr, kSt, kNumQ,  kDtHalf); });
            run1("dec:rope",       [&] { return ol::make_rope(*dev, T_Kn, T_dcos, T_dsin, T_Kr, kSt, kNumKv, kDtHalf); });

            {
                const uint32_t slot1_r = pos - kS;
                run1("dec:kv_append", [&] {
                    return ol::make_kv_append(*dev, T_Kr, T_V,
                                              T_Kt_cache[li], T_V_cache[li],
                                              slot1_r, kNkDt, kStKvDec, core);
                });
            }

            run1("dec:gqa_decode", [&] {
                return ol::make_gqa_decode(*dev, T_Qr, T_Kt_cache[li], T_V_cache[li],
                                           T_dmask, T_attn,
                                           kStDec, kStKvDec, kDt, kNumQ, kNumKv);
            });
            run_matmul_grid("dec:matmul_o", T_attn, w.Wo, T_proj, kSt, kNqDt, kHt);
            run1("dec:add",        [&] { return ol::make_eltwise_add(*dev, T_layer_in, T_proj, T_xmid); });
            run1("dec:rmsnorm",    [&] { return ol::make_rmsnorm(*dev, T_xmid, w.ln2g, T_ynorm, kSt, kHt, kEps); });
            // Fused gate+up matmul (iter8). One dispatch instead of two.
            run_matmul_grid("dec:matmul_ffn", T_ynorm, w.Wgateup, T_gateup, kSt, kHt, kFFtFused);
            run1("dec:silu_mul",   [&] { return ol::make_silu_mul(*dev, T_gate, T_up, T_fused); });
            run_matmul_grid("dec:matmul_ffn", T_fused, w.Wdown, T_down, kSt, kFFt, kHt);
            run1("dec:add",        [&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_layer_out); });
            std::swap(T_layer_in, T_layer_out);
        }

        run1("dec:final_rmsnorm", [&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); });
        run_matmul_lmhead("dec:lm_head", T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
        // Device-side argmax over row 0 of T_logits (single-core BRISC
        // scan in ops/argmax_row0). Replaces the 9.7-MB tile readback +
        // CPU argmax with a 4-byte readback.
        run1("dec:argmax", [&] {
            return ol::make_argmax_row0(*dev, T_logits, kVt, T_argmax, core);
        });
        uint32_t nxt = 0;
        {
            TIMED("dec:argmax_readback");
            tt::foil::read_buffer(*dev, *T_argmax.buf, &nxt, 4);
        }
        std::printf("%u\n", nxt);
        std::fflush(stdout);
        cur_token = nxt;
    }
    }

    g_prof.report();
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qwen3_run: FAIL — %s\n", e.what());
    return 1;
}
