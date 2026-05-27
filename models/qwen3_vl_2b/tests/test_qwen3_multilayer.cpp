// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: N-layer Qwen3-VL-2B Transformer chain on real weights.
//
// Runs a stack of N full layers (each = attention + MLP + 2 residuals),
// passing the output of layer i into layer i+1. Used to verify:
//   • DRAM weight loading for multiple layers (each ~100 MB)
//   • inter-layer residual dataflow & ping-ponging
//   • compound BF16 drift across multiple Transformer layers
//   • reset_l1 lifecycle for a long op chain

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

#ifndef CHAIN_NUM_LAYERS
#define CHAIN_NUM_LAYERS 3
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNumLayers = CHAIN_NUM_LAYERS;

// Qwen3-VL-2B layer geometry.
constexpr uint32_t kS         = 32;
constexpr uint32_t kH         = 2048;
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
constexpr float    kEps     = 1e-6f;
constexpr uint32_t kCosTiles = kSt * kDtHalf;

std::vector<uint16_t> load_bin(const std::string& path, std::size_t nelem) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + path);
    std::vector<uint16_t> v(nelem);
    f.read(reinterpret_cast<char*>(v.data()), nelem * 2);
    if (f.gcount() != static_cast<std::streamsize>(nelem * 2))
        throw std::runtime_error("short read: " + path);
    return v;
}

std::vector<uint16_t> tile2d(const std::vector<uint16_t>& rm,
                             uint32_t Rows, uint32_t Cols) {
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

std::vector<uint16_t> untile2d(const std::vector<uint16_t>& tiles,
                               uint32_t Rows, uint32_t Cols) {
    const uint32_t Rt = Rows / kTileH, Ct = Cols / kTileW;
    std::vector<uint16_t> rm(Rows * Cols, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t idx = 0;
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

std::vector<uint16_t> gamma_to_tiles(const std::vector<uint16_t>& gamma, uint32_t D) {
    std::vector<uint16_t> rm(kTileH * D);
    for (uint32_t r = 0; r < kTileH; ++r)
        for (uint32_t c = 0; c < D; ++c) rm[r * D + c] = gamma[c];
    return tile2d(rm, kTileH, D);
}

// Per-layer DRAM-resident weight set.
struct LayerW {
    tt::foil::op_lib::TensorDesc ln1g, ln2g, qng, kng;
    tt::foil::op_lib::TensorDesc Wq, Wk, Wv, Wo;
    tt::foil::op_lib::TensorDesc Wgate, Wup, Wdown;
};

LayerW load_and_upload_layer(tt::foil::Device& dev, const std::string& dir) {
    namespace ol = tt::foil::op_lib;
    auto load = [&](const char* name, std::size_t nelem) {
        return load_bin(dir + "/" + name, nelem);
    };

    auto ln1g_flat  = load("ln1_gamma.bin", kH);
    auto ln2g_flat  = load("ln2_gamma.bin", kH);
    auto qng_flat   = load("q_norm.bin",    kHeadDim);
    auto kng_flat   = load("k_norm.bin",    kHeadDim);
    auto Wq_rm      = load("W_q.bin",       kH * kNumQ  * kHeadDim);
    auto Wk_rm      = load("W_k.bin",       kH * kNumKv * kHeadDim);
    auto Wv_rm      = load("W_v.bin",       kH * kNumKv * kHeadDim);
    auto Wo_rm      = load("W_o.bin",       kNumQ * kHeadDim * kH);
    auto Wgate_rm   = load("W_gate.bin",    kH * kFFN);
    auto Wup_rm     = load("W_up.bin",      kH * kFFN);
    auto Wdown_rm   = load("W_down.bin",    kFFN * kH);

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
    up(L.ln1g,  gamma_to_tiles(ln1g_flat, kH));
    up(L.ln2g,  gamma_to_tiles(ln2g_flat, kH));
    up(L.qng,   gamma_to_tiles(qng_flat,  kHeadDim));
    up(L.kng,   gamma_to_tiles(kng_flat,  kHeadDim));
    up(L.Wq,    tile2d(Wq_rm,    kH,   kNumQ  * kHeadDim));
    up(L.Wk,    tile2d(Wk_rm,    kH,   kNumKv * kHeadDim));
    up(L.Wv,    tile2d(Wv_rm,    kH,   kNumKv * kHeadDim));
    up(L.Wo,    tile2d(Wo_rm,    kNumQ * kHeadDim, kH));
    up(L.Wgate, tile2d(Wgate_rm, kH,   kFFN));
    up(L.Wup,   tile2d(Wup_rm,   kH,   kFFN));
    up(L.Wdown, tile2d(Wdown_rm, kFFN, kH));
    return L;
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    if (!data_env) throw std::runtime_error("set TT_FOIL_QWEN3_DATA to data/qwen3_vl_2b");
    const std::string root = data_env;
    const std::string chain_dir = root + "/chain" + std::to_string(kNumLayers);

    // -----------------------------------------------------------------
    // 1. Load input, golden, cos/sin (shared across layers).
    // -----------------------------------------------------------------
    auto x_rm      = load_bin(chain_dir + "/chain_input.bin",  kS * kH);
    auto golden_rm = load_bin(chain_dir + "/chain_golden.bin", kS * kH);
    auto cos_rm    = load_bin(chain_dir + "/cos_table.bin",    kS * kHeadDim / 2);
    auto sin_rm    = load_bin(chain_dir + "/sin_table.bin",    kS * kHeadDim / 2);

    auto x_tiles   = tile2d(x_rm,   kS, kH);
    auto cos_tiles = tile2d(cos_rm, kS, kHeadDim / 2);
    auto sin_tiles = tile2d(sin_rm, kS, kHeadDim / 2);

    std::vector<uint16_t> mask_rm(kS * kS, 0);
    const uint16_t one_bf16 = f32_to_bf16(1.0f);
    for (uint32_t i = 0; i < kS; ++i)
        for (uint32_t j = 0; j <= i; ++j) mask_rm[i * kS + j] = one_bf16;
    auto mask_tiles = tile2d(mask_rm, kS, kS);

    // -----------------------------------------------------------------
    // 2. Open device, allocate persistent tensors (shared across layers).
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};
    namespace ol = tt::foil::op_lib;

    auto T_cos    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_sin    = ol::allocate_tensor_dram(*dev, kCosTiles);
    auto T_mask   = ol::allocate_tensor_dram(*dev, kSt * kSt);
    tt::foil::write_buffer(*dev, *T_cos.buf,  cos_tiles.data(),  cos_tiles.size()  * 2);
    tt::foil::write_buffer(*dev, *T_sin.buf,  sin_tiles.data(),  sin_tiles.size()  * 2);
    tt::foil::write_buffer(*dev, *T_mask.buf, mask_tiles.data(), mask_tiles.size() * 2);

    // Ping-pong layer I/O.
    auto T_layer_in  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_layer_out = ol::allocate_tensor_dram(*dev, kSt * kHt);
    tt::foil::write_buffer(*dev, *T_layer_in.buf, x_tiles.data(), x_tiles.size() * 2);

    // Per-layer intermediates (reused across layers).
    auto T_xnorm1 = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Q      = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_K      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_V      = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qn     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kn     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_Qr     = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_Kr     = ol::allocate_tensor_dram(*dev, kSt * kNkDt);
    auto T_attn   = ol::allocate_tensor_dram(*dev, kSt * kNqDt);
    auto T_proj   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_xmid   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_ynorm  = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_gate   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_up     = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_silu   = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_fused  = ol::allocate_tensor_dram(*dev, kSt * kFFt);
    auto T_down   = ol::allocate_tensor_dram(*dev, kSt * kHt);
    auto T_Qh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_KhT    = ol::allocate_tensor_dram(*dev, kDt * kSt);
    auto T_Vh     = ol::allocate_tensor_dram(*dev, kSt * kDt);
    auto T_AttnH  = ol::allocate_tensor_dram(*dev, kSt * kDt);

    // -----------------------------------------------------------------
    // 3. Load all layer weights upfront (~100 MB each in DRAM).
    // -----------------------------------------------------------------
    std::printf("loading %u layers from %s/layer{0..%u}/...\n",
                kNumLayers, root.c_str(), kNumLayers - 1);
    std::vector<LayerW> layers;
    layers.reserve(kNumLayers);
    for (uint32_t i = 0; i < kNumLayers; ++i)
        layers.push_back(load_and_upload_layer(*dev, root + "/layer" + std::to_string(i)));

    auto upload = [&](auto& t, const std::vector<uint16_t>& tiles) {
        tt::foil::write_buffer(*dev, *t.buf, tiles.data(), tiles.size() * 2);
    };
    auto run = [&](auto factory) { auto op = factory(); ol::execute(*dev, op); };
    auto step = [&] { tt::foil::release_kernels(*dev, core); tt::foil::reset_l1(*dev, core); };

    // -----------------------------------------------------------------
    // 4. Forward pass: for each layer i, transform T_layer_in → T_layer_out,
    //    then swap.
    // -----------------------------------------------------------------
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (uint32_t li = 0; li < kNumLayers; ++li) {
        const LayerW& w = layers[li];
        std::printf("  layer %u/%u …\n", li, kNumLayers);

        // x_norm1 = RMSNorm(in)
        run([&] { return ol::make_rmsnorm(*dev, T_layer_in, w.ln1g, T_xnorm1, kSt, kHt, kEps); });
        step();
        // QKV
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wq, T_Q, kSt, kHt, kNqDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wk, T_K, kSt, kHt, kNkDt); }); step();
        run([&] { return ol::make_matmul(*dev, T_xnorm1, w.Wv, T_V, kSt, kHt, kNkDt); }); step();
        // q_norm / k_norm
        run([&] { return ol::make_rmsnorm(*dev, T_Q, w.qng, T_Qn, kSt * kNumQ,  kDt, kEps); });
        step();
        run([&] { return ol::make_rmsnorm(*dev, T_K, w.kng, T_Kn, kSt * kNumKv, kDt, kEps); });
        step();
        // RoPE
        run([&] { return ol::make_rope(*dev, T_Qn, T_cos, T_sin, T_Qr, kSt, kNumQ,  kDtHalf); });
        step();
        run([&] { return ol::make_rope(*dev, T_Kn, T_cos, T_sin, T_Kr, kSt, kNumKv, kDtHalf); });
        step();

        // Per-head GQA host loop.
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
            run([&] {
                return ol::make_mha(*dev, T_Qh, T_KhT, T_Vh, T_mask, T_AttnH, kSt, kDt);
            });
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

        // MLP + residual #2 → write to T_layer_out
        run([&] { return ol::make_rmsnorm(*dev, T_xmid, w.ln2g, T_ynorm, kSt, kHt, kEps); }); step();
        run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wgate, T_gate, kSt, kHt, kFFt); }); step();
        run([&] { return ol::make_matmul(*dev, T_ynorm, w.Wup,   T_up,   kSt, kHt, kFFt); }); step();
        run([&] { return ol::make_silu(*dev, T_gate, T_silu); });                              step();
        run([&] { return ol::make_eltwise_mul(*dev, T_silu, T_up, T_fused); });                step();
        run([&] { return ol::make_matmul(*dev, T_fused, w.Wdown, T_down, kSt, kFFt, kHt); });  step();
        run([&] { return ol::make_eltwise_add(*dev, T_xmid, T_down, T_layer_out); });          step();

        // Swap in ↔ out for the next layer (re-bind shared_ptrs).
        std::swap(T_layer_in, T_layer_out);
    }

    // After the loop, T_layer_in holds the final output (post-swap).
    std::vector<uint16_t> out_tiles(kSt * kHt * kTileWords);
    tt::foil::read_buffer(*dev, *T_layer_in.buf, out_tiles.data(), out_tiles.size() * 2);
    auto out_rm = untile2d(out_tiles, kS, kH);

    // Tolerance: BF16 drift compounds per layer. Single layer was 0.047
    // at tol=0.30; expect roughly linear growth across N layers.
    const float kAbsTol = 0.30f * kNumLayers;
    const float kRelTol = 0.15f;
    uint32_t bad = 0; float worst_abs = 0.0f, worst_rel = 0.0f;
    uint32_t first_bad = kS * kH;
    for (uint32_t i = 0; i < kS * kH; ++i) {
        float got = bf16_to_f32(out_rm[i]);
        float exp = bf16_to_f32(golden_rm[i]);
        float d_abs = std::fabs(got - exp);
        float d_rel = d_abs / std::max(std::fabs(exp), 1e-3f);
        if (d_abs > worst_abs) worst_abs = d_abs;
        if (d_rel > worst_rel) worst_rel = d_rel;
        if (d_abs > kAbsTol && d_rel > kRelTol) {
            if (first_bad == kS * kH) first_bad = i;
            ++bad;
        }
    }

    tt::foil::close_device(std::move(dev));

    if (bad != 0) {
        std::fprintf(stderr,
            "test_qwen3_multilayer: %u/%u mismatches; first at i=%u (row=%u col=%u): "
            "got=%.4f exp=%.4f; worst abs=%.5f rel=%.4f (tol abs=%.5f rel=%.4f)\n",
            bad, kS * kH, first_bad, first_bad / kH, first_bad % kH,
            bf16_to_f32(out_rm[first_bad]), bf16_to_f32(golden_rm[first_bad]),
            worst_abs, worst_rel, kAbsTol, kRelTol);
        return 1;
    }
    std::printf("test_qwen3_multilayer: PASS  (N=%u, S=%u H=%u, worst abs=%.5f rel=%.4f)\n",
                kNumLayers, kS, kH, worst_abs, worst_rel);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_multilayer: FAIL — %s\n", e.what());
    return 1;
}
