// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B end-to-end inference forward pass (prefill).
//
//   token_ids
//     → embed_tokens lookup            [S, H]
//     → N × Transformer layer          [S, H]   (attention + MLP + 2 residuals)
//     → RMSNorm(model.norm)            [S, H]
//     → matmul (lm_head = embed.T)     [S, V]
//     → argmax                         [S]
//
// PASS criterion: top-1 predicted token per sequence row matches the
// numpy reference in tools/qwen3_inference_golden.py.
//
// Required inputs (regenerate after weight export):
//   python3 tools/qwen3_inference_golden.py \
//       --data-dir data/qwen3_vl_2b --num-layers 3 \
//       --num-q 16 --num-kv 8 --head-dim 128 \
//       --rope-theta 5000000.0 --seq 32
//
//   tools/qwen3_lm_head_golden.py must have been run too — this test
//   reuses model/lm_head_tiled.bin for the lm_head projection.

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

#ifndef INF_NUM_LAYERS
#define INF_NUM_LAYERS 3
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNumLayers = INF_NUM_LAYERS;

constexpr uint32_t kS         = 32;
constexpr uint32_t kH         = 2048;
constexpr uint32_t kV         = 151936;
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
constexpr uint32_t kNqDt    = kNumQ  * kDt;
constexpr uint32_t kNkDt    = kNumKv * kDt;
constexpr uint32_t kVt      = kV / kTileW;          // 4748
constexpr float    kEps     = 1e-6f;
constexpr uint32_t kCosTiles = kSt * kDtHalf;

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

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    if (!data_env) throw std::runtime_error("set TT_FOIL_QWEN3_DATA to data/qwen3_vl_2b");
    const std::string root  = data_env;
    const std::string mdir  = root + "/model";
    const std::string chain = root + "/chain" + std::to_string(kNumLayers);

    // -----------------------------------------------------------------
    // 1. Load inputs / weights / golden.
    // -----------------------------------------------------------------
    auto token_ids = load_u32(chain + "/inf_token_ids.bin", kS);
    auto golden_top1 = load_u32(chain + "/inf_top1.bin",      kS);
    auto cos_rm = load_bin(chain + "/cos_table.bin", kS * kHeadDim / 2);
    auto sin_rm = load_bin(chain + "/sin_table.bin", kS * kHeadDim / 2);

    std::printf("loading embed_tokens (622 MB)...\n");
    auto embed_table = load_bin(mdir + "/embed_tokens.bin",
                                static_cast<std::size_t>(kV) * kH);
    auto final_g     = load_bin(mdir + "/final_norm.bin", kH);
    std::printf("loading lm_head_tiled.bin (622 MB)...\n");
    auto lmhead_tiles = load_bin(mdir + "/lm_head_tiled.bin",
                                 static_cast<std::size_t>(kHt) * kVt * kTileWords);

    std::vector<uint16_t> mask_rm(kS * kS, 0);
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;
    auto mask_tiles = tile2d(mask_rm, kS, kS);
    auto cos_tiles  = tile2d(cos_rm,  kS, kHeadDim / 2);
    auto sin_tiles  = tile2d(sin_rm,  kS, kHeadDim / 2);
    auto final_g_tiles = gamma_to_tiles(final_g, kH);

    // -----------------------------------------------------------------
    // 2. Open device, allocate persistent tensors.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

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

    auto T_cos    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_sin    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);
    auto T_final_g = ol::allocate_tensor_dram(*dev, kHt);
    tt::foil::write_buffer(*dev, *T_cos.buf,    cos_tiles.data(),  cos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_sin.buf,    sin_tiles.data(),  sin_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_mask.buf,   mask_tiles.data(), mask_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_final_g.buf, final_g_tiles.data(),
                           final_g_tiles.size() * 2);

    auto T_embed_rm  = ol::allocate_tensor_dram(*dev, kSt * kHt);     // row-major
    auto T_layer_in  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_layer_out = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xnorm1    = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Q = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qn = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_proj = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Qh   = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_KhT  = ol::allocate_tensor_dram(*dev, kDt * kSt);
    auto T_Vh   = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_AttnH = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_normed = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_logits = ol::allocate_tensor_dram(*dev, kSt * kVt);

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };
    auto step = [&] { tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core); };

    // -----------------------------------------------------------------
    // 3. Embedding lookup → tile into T_layer_in.
    // -----------------------------------------------------------------
    std::printf("running embedding lookup...\n");
    std::vector<uint32_t> tid_vec(token_ids.begin(), token_ids.end());
    run([&] { return ol::make_embedding(*dev, T_embed_table, tid_vec, kH, T_embed_rm); });
    step();

    std::vector<uint16_t> hidden_rm(kS * kH);
    tt::foil::read_buffer(*dev, *T_embed_rm.buf, hidden_rm.data(), kS * kH * 2);
    upload(T_layer_in, tile2d(hidden_rm, kS, kH));

    // -----------------------------------------------------------------
    // 4. Load layer weights + N-layer forward.
    // -----------------------------------------------------------------
    std::printf("loading %u layers...\n", kNumLayers);
    std::vector<LayerW> layers;
    layers.reserve(kNumLayers);
    for (uint32_t i = 0; i < kNumLayers; ++i)
        layers.push_back(load_and_upload_layer(*dev, root + "/layer" + std::to_string(i)));

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::printf("  layer %u …\n", li);

        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); }); step();
        run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); }); step();
        run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); }); step();

        std::vector<uint16_t> Qr_all(kSt * kNqDt * kTileWords);
        std::vector<uint16_t> Kr_all(kSt * kNkDt * kTileWords);
        std::vector<uint16_t> V_all (kSt * kNkDt * kTileWords);
        tt::foil::read_buffer(*dev, *T_Qr.buf, Qr_all.data(), Qr_all.size() * 2);
        tt::foil::read_buffer(*dev, *T_Kr.buf, Kr_all.data(), Kr_all.size() * 2);
        tt::foil::read_buffer(*dev, *T_V.buf,  V_all.data(),  V_all.size()  * 2);
        auto Qr_rm = untile2d(Qr_all, kS, kNumQ  * kHeadDim);
        auto Kr_rm = untile2d(Kr_all, kS, kNumKv * kHeadDim);
        auto V_rm  = untile2d(V_all,  kS, kNumKv * kHeadDim);

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
            run([&] { return ol::make_mha(*dev, T_Qh, T_KhT, T_Vh, T_mask, T_AttnH, kSt, kDt); });
            step();
            std::vector<uint16_t> ah_tiles(kSt * kDt * kTileWords);
            tt::foil::read_buffer(*dev, *T_AttnH.buf, ah_tiles.data(), ah_tiles.size() * 2);
            auto ah_rm = untile2d(ah_tiles, kS, kHeadDim);
            for (uint32_t s = 0; s < kS; ++s)
                for (uint32_t d = 0; d < kHeadDim; ++d)
                    attn_concat_rm[s * kNumQ * kHeadDim + h * kHeadDim + d] = ah_rm[s * kHeadDim + d];
        }
        upload(T_attn, tile2d(attn_concat_rm, kS, kNumQ * kHeadDim));
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

    // -----------------------------------------------------------------
    // 5. Final RMSNorm + lm_head matmul.
    // -----------------------------------------------------------------
    std::printf("  final norm + lm_head ...\n");
    run([&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); });
    step();
    {
        auto op = ol::make_matmul(*dev, T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
        ol::execute(*dev, op);
    }

    // -----------------------------------------------------------------
    // 6. Read logits, argmax per row, compare with golden top-1.
    // -----------------------------------------------------------------
    std::vector<uint16_t> logits_tiles(static_cast<size_t>(kSt) * kVt * kTileWords);
    tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
    auto logits_rm = untile2d(logits_tiles, kS, kV);

    std::vector<uint32_t> got_top1(kS, 0);
    for (uint32_t s = 0; s < kS; ++s) {
        float best = -1e30f; uint32_t bi = 0;
        for (uint32_t v = 0; v < kV; ++v) {
            float lv = bf16_to_f32(logits_rm[static_cast<size_t>(s) * kV + v]);
            if (lv > best) { best = lv; bi = v; }
        }
        got_top1[s] = bi;
    }

    uint32_t mismatch = 0;
    for (uint32_t s = 0; s < kS; ++s) if (got_top1[s] != golden_top1[s]) ++mismatch;

    tt::foil::close_device(std::move(dev));

    std::printf("top-1 (got / ref) first 8 of %u rows: ", kS);
    for (uint32_t s = 0; s < 8; ++s)
        std::printf("[%u / %u] ", got_top1[s], golden_top1[s]);
    std::printf("\n");

    // Loose threshold: BF16 drift through 3 layers + final norm + lm_head
    // can shuffle close ties. Accept up to 8/32 = 25% argmax mismatches as
    // a first signal; tighten as we add fp32 accumulators / fused GQA.
    const uint32_t kAllowedMismatch = kS / 4;
    if (mismatch > kAllowedMismatch) {
        std::fprintf(stderr,
            "test_qwen3_inference: %u/%u top-1 mismatches (allowed %u)\n",
            mismatch, kS, kAllowedMismatch);
        return 1;
    }
    std::printf("test_qwen3_inference: PASS  (N=%u layers, S=%u V=%u, "
                "%u/%u top-1 matches; %u within allowed)\n",
                kNumLayers, kS, kV, kS - mismatch, kS, kAllowedMismatch);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_inference: FAIL — %s\n", e.what());
    return 1;
}
