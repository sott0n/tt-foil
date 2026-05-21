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

struct LayerW {
    tt::foil::op_lib::TensorDesc ln1g, ln2g, qng, kng;
    tt::foil::op_lib::TensorDesc Wq, Wk, Wv, Wo;
    tt::foil::op_lib::TensorDesc Wgate, Wup, Wdown;
};
LayerW load_and_upload_layer(tt::foil::Device& dev, const std::string& d) {
    namespace ol = tt::foil::op_lib;
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
    up(L.ln1g,  gamma_to_tiles(ln1g, kH));
    up(L.ln2g,  gamma_to_tiles(ln2g, kH));
    up(L.qng,   gamma_to_tiles(qng,  kHeadDim));
    up(L.kng,   gamma_to_tiles(kng,  kHeadDim));
    up(L.Wq,    tile2d(Wq,    kH,   kNumQ  * kHeadDim));
    up(L.Wk,    tile2d(Wk,    kH,   kNumKv * kHeadDim));
    up(L.Wv,    tile2d(Wv,    kH,   kNumKv * kHeadDim));
    up(L.Wo,    tile2d(Wo,    kNumQ * kHeadDim, kH));
    up(L.Wgate, tile2d(Wgate, kH,   kFFN));
    up(L.Wup,   tile2d(Wup,   kH,   kFFN));
    up(L.Wdown, tile2d(Wdown, kFFN, kH));
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
    std::fprintf(stderr, "loading embed_tokens (622 MB)...\n");
    auto embed_table = load_bin(mdir + "/embed_tokens.bin",
                                static_cast<std::size_t>(kV) * kH);
    auto final_g = load_bin(mdir + "/final_norm.bin", kH);
    std::fprintf(stderr, "loading lm_head_tiled.bin (622 MB)...\n");
    auto lmhead_tiles = load_bin(mdir + "/lm_head_tiled.bin",
                                 static_cast<std::size_t>(kHt) * kVt * kTileWords);

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
    auto final_g_tiles = gamma_to_tiles(final_g, kH);

    // -----------------------------------------------------------------
    // Open device, upload static tensors.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    ol::TensorDesc T_embed_table;
    T_embed_table.buf = tt::foil::allocate_buffer(
        *dev, tt::foil::BufferLocation::DRAM,
        static_cast<std::size_t>(kV) * kH * 2);
    T_embed_table.num_tiles = 0;
    tt::foil::write_buffer(*dev, *T_embed_table.buf, embed_table.data(),
                           static_cast<std::size_t>(kV) * kH * 2);

    auto T_W_lm = ol::allocate_tensor_dram(*dev, kHt * kVt);
    tt::foil::write_buffer(*dev, *T_W_lm.buf, lmhead_tiles.data(),
                           lmhead_tiles.size() * 2);

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
    auto T_Q  = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K  = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V  = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qn = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn  = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kt    = ol::allocate_tensor_dram(*dev, kNkDt * kSt);
    auto T_proj  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up    = ol::allocate_tensor_dram(*dev, kSt * kFFt);
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

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run  = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };
    auto step = [&] { tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core); };

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
    // Embedding lookup → T_layer_in.
    // -----------------------------------------------------------------
    std::fprintf(stderr, "embedding lookup...\n");
    std::vector<uint32_t> tid_vec(token_ids.begin(), token_ids.end());
    run([&] { return ol::make_embedding(*dev, T_embed_table, tid_vec, kH, T_embed_rm); });
    step();
    std::vector<uint16_t> hidden_rm(kS * kH);
    tt::foil::read_buffer(*dev, *T_embed_rm.buf, hidden_rm.data(), kS * kH * 2);
    upload(T_layer_in, tile2d(hidden_rm, kS, kH));

    // -----------------------------------------------------------------
    // Layer weights.
    // -----------------------------------------------------------------
    std::fprintf(stderr, "loading %u layers...\n", kNumLayers);
    std::vector<LayerW> layers;
    layers.reserve(kNumLayers);
    for (uint32_t i = 0; i < kNumLayers; ++i)
        layers.push_back(load_and_upload_layer(*dev, root + "/layer" + std::to_string(i)));

    // -----------------------------------------------------------------
    // Prefill — populates slot 0 of each layer's KV cache.
    // -----------------------------------------------------------------
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::fprintf(stderr, "  prefill layer %u …\n", li);

        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); }); step();
        run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); }); step();
        run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); }); step();
        run([&] { return ol::make_transpose_2d(*dev, T_Kr, T_Kt, kSt, kNkDt); }); step();

        {
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

        run([&] {
            return ol::make_gqa_fused(*dev, T_Qr, T_Kt, T_V, T_mask, T_attn,
                                      kSt, kDt, kNumQ, kNumKv);
        }); step();
        run([&] { return ol::make_matmul(*dev, T_attn, w.Wo, T_proj, kSt, kNqDt, kHt); }); step();
        run([&] { return ol::make_eltwise_add(*dev, T_layer_in, T_proj, T_xmid); });        step();
        run([&] { return ol::make_rmsnorm(*dev, T_xmid, w.ln2g, T_ynorm, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wgate, T_gate, kSt, kHt, kFFt); }); step();
        run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wup,   T_up,   kSt, kHt, kFFt); }); step();
        run([&] { return ol::make_silu(*dev, T_gate, T_silu); });                              step();
        run([&] { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });                step();
        run([&] { return ol::make_matmul(*dev, T_fused, w.Wdown, T_down, kSt, kFFt, kHt); });  step();
        run([&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_layer_out); });          step();
        std::swap(T_layer_in, T_layer_out);
    }

    // Prefill final norm + lm_head → argmax(row S-1) is the first decode input.
    std::fprintf(stderr, "  prefill final norm + lm_head ...\n");
    run([&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); }); step();
    {
        auto op = ol::make_matmul(*dev, T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
        ol::execute(*dev, op); step();
    }
    std::vector<uint16_t> logits_tiles(static_cast<size_t>(kSt) * kVt * kTileWords);
    tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
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

    for (uint32_t t = 0; t < kNumDecode; ++t) {
        const uint32_t pos      = kS + t;
        const uint32_t valid_kv = pos + 1;
        if (pos >= kStKvDec * kTileH) {
            std::fprintf(stderr, "  decode beyond cache capacity, stopping\n");
            break;
        }
        std::fprintf(stderr, "  decode step %u: pos=%u input=%u\n", t, pos, cur_token);

        std::vector<uint32_t> dec_tid = {cur_token};
        run([&] { return ol::make_embedding(*dev, T_embed_table, dec_tid, kH, T_dembed_rm); });
        step();
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

            run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); }); step();
            run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
            run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
            run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
            run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); }); step();
            run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); }); step();
            run([&] { return ol::make_rope(*dev, T_Qn, T_dcos, T_dsin, T_Qr, kSt, kNumQ,  kDtHalf); }); step();
            run([&] { return ol::make_rope(*dev, T_Kn, T_dcos, T_dsin, T_Kr, kSt, kNumKv, kDtHalf); }); step();

            {
                const uint32_t slot1_r = pos - kS;
                std::vector<uint16_t> Kr_tiles(static_cast<size_t>(kSt) * kNkDt * kTileWords);
                std::vector<uint16_t> V_tiles (static_cast<size_t>(kSt) * kNkDt * kTileWords);
                tt::foil::read_buffer(*dev, *T_Kr.buf, Kr_tiles.data(), Kr_tiles.size() * 2);
                tt::foil::read_buffer(*dev, *T_V.buf,  V_tiles.data(),  V_tiles.size()  * 2);
                auto Kr_rm = untile2d(Kr_tiles, kTileH, kTotalNk);
                auto V_rm  = untile2d(V_tiles,  kTileH, kTotalNk);
                for (uint32_t c = 0; c < kTotalNk; ++c) {
                    cache_K_slot1_rm[li][slot1_r * kTotalNk + c] = Kr_rm[c];
                    cache_V_slot1_rm[li][slot1_r * kTotalNk + c] = V_rm [c];
                }
            }
            {
                auto KT_slot1 = build_slot1_KT_tiles(cache_K_slot1_rm[li]);
                auto V_slot1  = build_slot1_V_tiles (cache_V_slot1_rm[li]);
                tt::foil::write_buffer(*dev, *T_Kt_cache[li].buf, kSlot0Bytes,
                                       KT_slot1.data(), KT_slot1.size() * 2);
                tt::foil::write_buffer(*dev, *T_V_cache[li].buf, kSlot0Bytes,
                                       V_slot1.data(),  V_slot1.size()  * 2);
            }

            run([&] {
                return ol::make_gqa_decode(*dev, T_Qr, T_Kt_cache[li], T_V_cache[li],
                                           T_dmask, T_attn,
                                           kStDec, kStKvDec, kDt, kNumQ, kNumKv);
            }); step();
            run([&] { return ol::make_matmul(*dev, T_attn, w.Wo, T_proj, kSt, kNqDt, kHt); }); step();
            run([&] { return ol::make_eltwise_add(*dev, T_layer_in, T_proj, T_xmid); });        step();
            run([&] { return ol::make_rmsnorm(*dev, T_xmid, w.ln2g, T_ynorm, kSt, kHt, kEps); }); step();
            run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wgate, T_gate, kSt, kHt, kFFt); }); step();
            run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wup,   T_up,   kSt, kHt, kFFt); }); step();
            run([&] { return ol::make_silu(*dev, T_gate, T_silu); });                              step();
            run([&] { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });                step();
            run([&] { return ol::make_matmul(*dev, T_fused, w.Wdown, T_down, kSt, kFFt, kHt); });  step();
            run([&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_layer_out); });          step();
            std::swap(T_layer_in, T_layer_out);
        }

        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); }); step();
        {
            auto op = ol::make_matmul(*dev, T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
            ol::execute(*dev, op); step();
        }
        tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
        auto dec_logits_rm = untile2d(logits_tiles, kS, kV);
        uint32_t nxt = 0;
        float best = -1e30f;
        for (uint32_t v = 0; v < kV; ++v) {
            float lv = bf16_to_f32(dec_logits_rm[v]);
            if (lv > best) { best = lv; nxt = v; }
        }
        std::printf("%u\n", nxt);
        std::fflush(stdout);
        cur_token = nxt;
    }

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qwen3_run: FAIL — %s\n", e.what());
    return 1;
}
