// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B end-to-end prefill + 1-step decode with a KV cache.
//
// Pipeline (mirrors tools/qwen3_inference_golden.py decode path):
//
//   ┌── prefill (identical to test_qwen3_inference) ──┐
//   │  embed → N × Transformer → final_norm → lm_head │
//   │  per layer: snapshot K_T (post-RoPE, transposed)│
//   │             and V tiles → host KV-cache vectors │
//   └────────── argmax row S-1 → next-token id ───────┘
//                                  │
//                                  ▼
//   ┌── decode step (St_q=1, St_kv=2 cache+new) ──────┐
//   │  embed(next_id) padded to 32-row tile           │
//   │  for each layer:                                │
//   │    RMSNorm → Wq/Wk/Wv → q_norm/k_norm           │
//   │    RoPE@pos=S → transpose K_new                 │
//   │    build cache_KT[..,St_kv=2] (slot0=prefill,   │
//   │                                slot1=new) host  │
//   │    build cache_V [St_kv=2,..] (slot0=prefill,   │
//   │                                slot1=new) host  │
//   │    gqa_decode → Wo → +residual → MLP →+residual │
//   │  final_norm → lm_head → argmax(row 0)           │
//   └────────── compare to decode_top1 golden ────────┘
//
// Required golden (regenerate after weight export):
//   python3 tools/qwen3_inference_golden.py \
//       --data-dir data/qwen3_vl_2b --num-layers <N> \
//       --num-q 16 --num-kv 8 --head-dim 128 \
//       --rope-theta 5000000.0 --seq 32 \
//       --prompt "The capital of Japan is"

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

#ifndef DEC_NUM_LAYERS
#define DEC_NUM_LAYERS 3
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNumLayers = DEC_NUM_LAYERS;

constexpr uint32_t kS         = 32;
constexpr uint32_t kH         = 2048;
constexpr uint32_t kV         = 151936;
constexpr uint32_t kFFN       = 6144;
constexpr uint32_t kNumQ      = 16;
constexpr uint32_t kNumKv     = 8;
constexpr uint32_t kHeadDim   = 128;
constexpr uint32_t kGqaGroups = kNumQ / kNumKv;

constexpr uint32_t kSt      = kS       / kTileH;   // 1 (prefill)
constexpr uint32_t kStDec   = 1;                   // decode Q tile-rows
constexpr uint32_t kStKvDec = 2;                   // decode K/V tile-rows (cache+new)
constexpr uint32_t kHt      = kH       / kTileW;   // 64
constexpr uint32_t kFFt     = kFFN     / kTileW;   // 192
constexpr uint32_t kDt      = kHeadDim / kTileW;   // 4
constexpr uint32_t kDtHalf  = kDt / 2;             // 2
constexpr uint32_t kNqDt    = kNumQ  * kDt;
constexpr uint32_t kNkDt    = kNumKv * kDt;
constexpr uint32_t kVt      = kV / kTileW;          // 4748
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
    for (auto& v : qng) { v = f32_to_bf16(bf16_to_f32(v) * inv_sqrt_d); }

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

    // ---- Load inputs / golden / model tensors ----
    auto token_ids   = load_u32(chain + "/inf_token_ids.bin", kS);
    auto cos_rm      = load_bin(chain + "/cos_table.bin", kS * kHeadDim / 2);
    auto sin_rm      = load_bin(chain + "/sin_table.bin", kS * kHeadDim / 2);
    auto decode_in   = load_u32(chain + "/decode_input.bin", 1);
    auto decode_top1 = load_u32(chain + "/decode_top1.bin",  1);
    auto decode_cos_rm = load_bin(chain + "/decode_cos.bin", kHeadDim / 2);
    auto decode_sin_rm = load_bin(chain + "/decode_sin.bin", kHeadDim / 2);

    std::printf("loading embed_tokens (622 MB)...\n");
    auto embed_table = load_bin(mdir + "/embed_tokens.bin",
                                static_cast<std::size_t>(kV) * kH);
    auto final_g     = load_bin(mdir + "/final_norm.bin", kH);
    std::printf("loading lm_head_tiled.bin (622 MB)...\n");
    auto lmhead_tiles = load_bin(mdir + "/lm_head_tiled.bin",
                                 static_cast<std::size_t>(kHt) * kVt * kTileWords);

    // Prefill causal mask [S, S].
    std::vector<uint16_t> mask_rm(kS * kS, 0);
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;
    auto mask_tiles = tile2d(mask_rm, kS, kS);
    auto cos_tiles  = tile2d(cos_rm,  kS, kHeadDim / 2);
    auto sin_tiles  = tile2d(sin_rm,  kS, kHeadDim / 2);
    auto final_g_tiles = gamma_to_tiles(final_g, kH);

    // Decode mask [St_q=1 tile = 32 rows, St_kv=2 tiles = 64 cols]:
    // only row 0 of Q is the real new token; it attends to cols 0..kS
    // (prefill positions 0..S-1 plus the new position S = kS).
    std::vector<uint16_t> dmask_rm(kTileH * (kStKvDec * kTileW), 0);
    for (uint32_t c = 0; c <= kS; ++c) dmask_rm[0 * (kStKvDec * kTileW) + c] = one_bf16;
    auto dmask_tiles = tile2d(dmask_rm, kTileH, kStKvDec * kTileW);

    // Decode cos/sin tile: row 0 carries the per-dim cos/sin for position S.
    auto build_decode_rope_tile = [&](const std::vector<uint16_t>& row) {
        std::vector<uint16_t> rm(kTileH * (kDtHalf * kTileW), 0);
        for (uint32_t c = 0; c < kHeadDim / 2; ++c) rm[c] = row[c];
        return tile2d(rm, kTileH, kDtHalf * kTileW);
    };
    auto dcos_tiles = build_decode_rope_tile(decode_cos_rm);
    auto dsin_tiles = build_decode_rope_tile(decode_sin_rm);

    // ---- Open device, allocate persistent tensors ----
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

    auto T_cos    = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_sin    = ol::allocate_tensor_dram(*dev, kSt * kDtHalf);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);
    auto T_final_g = ol::allocate_tensor_dram(*dev, kHt);
    tt::foil::write_buffer(*dev, *T_cos.buf,    cos_tiles.data(),  cos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_sin.buf,    sin_tiles.data(),  sin_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_mask.buf,   mask_tiles.data(), mask_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_final_g.buf, final_g_tiles.data(),
                           final_g_tiles.size() * 2);

    // Decode-only RoPE table + mask DRAM tensors.
    auto T_dcos = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dsin = ol::allocate_tensor_dram(*dev, kStDec * kDtHalf);
    auto T_dmask = ol::allocate_tensor_dram(*dev, kStDec * kStKvDec);
    tt::foil::write_buffer(*dev, *T_dcos.buf,  dcos_tiles.data(),  dcos_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_dsin.buf,  dsin_tiles.data(),  dsin_tiles.size() * 2);
    tt::foil::write_buffer(*dev, *T_dmask.buf, dmask_tiles.data(), dmask_tiles.size() * 2);

    auto T_embed_rm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
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
    auto T_Kt   = ol::allocate_tensor_dram(*dev, kNkDt * kSt);
    auto T_proj = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_normed = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_logits = ol::allocate_tensor_dram(*dev, kSt * kVt);

    // Decode-step extra tensors (St_kv=2 caches, St_q=1 attn output).
    auto T_Kt_cache = ol::allocate_tensor_dram(*dev, kNkDt * kStKvDec);
    auto T_V_cache  = ol::allocate_tensor_dram(*dev, kStKvDec * kNkDt);

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };
    auto step = [&] { tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core); };

    // ---- Embedding lookup → T_layer_in ----
    std::printf("running embedding lookup...\n");
    std::vector<uint32_t> tid_vec(token_ids.begin(), token_ids.end());
    run([&] { return ol::make_embedding(*dev, T_embed_table, tid_vec, kH, T_embed_rm); });
    step();

    std::vector<uint16_t> hidden_rm(kS * kH);
    tt::foil::read_buffer(*dev, *T_embed_rm.buf, hidden_rm.data(), kS * kH * 2);
    upload(T_layer_in, tile2d(hidden_rm, kS, kH));

    // ---- Load layer weights ----
    std::printf("loading %u layers...\n", kNumLayers);
    std::vector<LayerW> layers;
    layers.reserve(kNumLayers);
    for (uint32_t i = 0; i < kNumLayers; ++i)
        layers.push_back(load_and_upload_layer(*dev, root + "/layer" + std::to_string(i)));

    // Per-layer KV caches captured during prefill — host-side tile vectors,
    // sized for one prefill slot (kSt = 1 tile-row).
    //   KT slot 0:  num_kv*Dt rows × kSt cols of tiles (= kNkDt * kSt tiles)
    //   V  slot 0:  kSt rows × num_kv*Dt cols of tiles (= kSt * kNkDt tiles)
    std::vector<std::vector<uint16_t>> cache_KT_prefill(kNumLayers);
    std::vector<std::vector<uint16_t>> cache_V_prefill(kNumLayers);

    // -----------------------------------------------------------------
    // Prefill — identical to test_qwen3_inference, plus per-layer KT/V
    // read-back after the K transpose / V matmul.
    // -----------------------------------------------------------------
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::printf("  prefill layer %u …\n", li);

        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); }); step();
        run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); }); step();
        run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); }); step();
        run([&] { return ol::make_transpose_2d(*dev, T_Kr, T_Kt, kSt, kNkDt); }); step();

        // Snapshot prefill KT (= post-RoPE K transposed) and V tiles.
        cache_KT_prefill[li].resize(static_cast<size_t>(kNkDt) * kSt * kTileWords);
        cache_V_prefill[li].resize(static_cast<size_t>(kSt) * kNkDt * kTileWords);
        tt::foil::read_buffer(*dev, *T_Kt.buf, cache_KT_prefill[li].data(),
                              cache_KT_prefill[li].size() * 2);
        tt::foil::read_buffer(*dev, *T_V.buf,  cache_V_prefill[li].data(),
                              cache_V_prefill[li].size() * 2);

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

    // ---- Sanity: prefill argmax(row S-1) must equal decode_input ----
    std::printf("  prefill final norm + lm_head ...\n");
    run([&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); });
    step();
    {
        auto op = ol::make_matmul(*dev, T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
        ol::execute(*dev, op);
        step();
    }
    std::vector<uint16_t> logits_tiles(static_cast<size_t>(kSt) * kVt * kTileWords);
    tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
    auto logits_rm = untile2d(logits_tiles, kS, kV);
    uint32_t last_top1 = 0;
    {
        float best = -1e30f;
        for (uint32_t v = 0; v < kV; ++v) {
            float lv = bf16_to_f32(logits_rm[static_cast<size_t>(kS - 1) * kV + v]);
            if (lv > best) { best = lv; last_top1 = v; }
        }
    }
    std::printf("  prefill last-row top1=%u  (golden decode_input=%u)\n",
                last_top1, decode_in[0]);

    // -----------------------------------------------------------------
    // Decode step.
    // -----------------------------------------------------------------
    // Use the golden decode_input (not last_top1) so the test isolates
    // decode-path numeric error from any prefill drift at the last row.
    std::printf("decode: input token = %u, expected next = %u\n",
                decode_in[0], decode_top1[0]);

    // Embed the new token into a 32-row tile (row 0 = embed, rest 0).
    std::vector<uint32_t> dec_tid = {decode_in[0]};
    auto T_dembed_rm = ol::allocate_tensor_dram(*dev, kHt);  // [1, H] worth of row-major bytes
    run([&] { return ol::make_embedding(*dev, T_embed_table, dec_tid, kH, T_dembed_rm); });
    step();
    std::vector<uint16_t> dec_row(kH);
    tt::foil::read_buffer(*dev, *T_dembed_rm.buf, dec_row.data(), kH * 2);
    std::vector<uint16_t> dec_hidden_rm(kTileH * kH, 0);
    for (uint32_t c = 0; c < kH; ++c) dec_hidden_rm[c] = dec_row[c];  // row 0 only
    upload(T_layer_in, tile2d(dec_hidden_rm, kTileH, kH));

    // Helper: build cache KT [num_kv*Dt, St_kv=2] tiles from (prefill, new).
    // tile (r, s) at flat index r*St_kv + s.
    auto build_cache_KT = [&](const std::vector<uint16_t>& prefill,
                              const std::vector<uint16_t>& new_kt) {
        std::vector<uint16_t> out(static_cast<size_t>(kNkDt) * kStKvDec * kTileWords, 0);
        for (uint32_t r = 0; r < kNkDt; ++r) {
            std::copy_n(prefill.data() + r * kTileWords,        kTileWords,
                        out.data()    + (r * kStKvDec + 0) * kTileWords);
            std::copy_n(new_kt.data() + r * kTileWords,         kTileWords,
                        out.data()    + (r * kStKvDec + 1) * kTileWords);
        }
        return out;
    };
    // V cache [St_kv=2, num_kv*Dt]: slot 0 = prefill tiles, slot 1 = new tiles.
    // Layout is row-major in tile-space so it's just concat.
    auto build_cache_V = [&](const std::vector<uint16_t>& prefill,
                             const std::vector<uint16_t>& new_v) {
        std::vector<uint16_t> out;
        out.reserve(prefill.size() + new_v.size());
        out.insert(out.end(), prefill.begin(), prefill.end());
        out.insert(out.end(), new_v.begin(),    new_v.end());
        return out;
    };

    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::printf("  decode layer %u …\n", li);

        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); }); step();
        run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); }); step();
        run([&] { return ol::make_rope(*dev, T_Qn, T_dcos, T_dsin, T_Qr, kSt, kNumQ,  kDtHalf); }); step();
        run([&] { return ol::make_rope(*dev, T_Kn, T_dcos, T_dsin, T_Kr, kSt, kNumKv, kDtHalf); }); step();
        run([&] { return ol::make_transpose_2d(*dev, T_Kr, T_Kt, kSt, kNkDt); }); step();

        std::vector<uint16_t> new_KT(static_cast<size_t>(kNkDt) * kSt * kTileWords);
        std::vector<uint16_t> new_V (static_cast<size_t>(kSt) * kNkDt * kTileWords);
        tt::foil::read_buffer(*dev, *T_Kt.buf, new_KT.data(), new_KT.size() * 2);
        tt::foil::read_buffer(*dev, *T_V.buf,  new_V.data(),  new_V.size()  * 2);

        upload(T_Kt_cache, build_cache_KT(cache_KT_prefill[li], new_KT));
        upload(T_V_cache,  build_cache_V (cache_V_prefill[li],  new_V));

        run([&] {
            return ol::make_gqa_decode(*dev, T_Qr, T_Kt_cache, T_V_cache, T_dmask, T_attn,
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

    // ---- Decode final norm + lm_head + argmax(row 0) ----
    run([&] { return ol::make_rmsnorm(*dev, T_layer_in, T_final_g, T_normed, kSt, kHt, kEps); });
    step();
    {
        auto op = ol::make_matmul(*dev, T_normed, T_W_lm, T_logits, kSt, kHt, kVt);
        ol::execute(*dev, op);
    }
    tt::foil::read_buffer(*dev, *T_logits.buf, logits_tiles.data(), logits_tiles.size() * 2);
    auto dec_logits_rm = untile2d(logits_tiles, kS, kV);
    uint32_t dec_pred = 0;
    {
        float best = -1e30f;
        for (uint32_t v = 0; v < kV; ++v) {
            float lv = bf16_to_f32(dec_logits_rm[v]);  // row 0 only
            if (lv > best) { best = lv; dec_pred = v; }
        }
    }

    tt::foil::close_device(std::move(dev));

    std::printf("test_qwen3_decode: predicted=%u  golden=%u\n", dec_pred, decode_top1[0]);
    if (dec_pred != decode_top1[0]) {
        std::fprintf(stderr,
            "test_qwen3_decode: FAIL (got %u, expected %u)\n",
            dec_pred, decode_top1[0]);
        return 1;
    }
    std::printf("test_qwen3_decode: PASS  (N=%u layers, input=%u → next=%u)\n",
                kNumLayers, decode_in[0], dec_pred);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_decode: FAIL — %s\n", e.what());
    return 1;
}
