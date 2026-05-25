// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// CIFAR-10 ResNet-20 inference end-to-end on a single Tensix core.
// Loads the pretrained akamaster weights + sample image + golden logits
// produced by models/cifar10_resnet20/export.py and runs the full
// forward pass on device. Every multiply/add/ReLU lives on device;
// layout (im2col, tile/untile, option-A zero-padded skip) stays on
// host.
//
// Architecture (akamaster, option-A skip):
//
//   stem      conv 3×3 s=1 (3→16)   + bias + ReLU                → (16, 32, 32)
//   layer1.k  conv 3×3 s=1 (16→16)  + bias + ReLU
//             conv 3×3 s=1 (16→16)  + bias
//             skip = identity
//             ReLU                                                 → (16, 32, 32)
//   layer2.0  conv 3×3 s=2 (16→32)  + bias + ReLU
//             conv 3×3 s=1 (32→32)  + bias
//             skip = subsample-stride-2 then zero-pad channels     → (32, 16, 16)
//             ReLU
//   layer2.{1,2}: basic block at (32, 16, 16) like layer1
//   layer3.0:    downsample like layer2.0 but (32→64, 16→8)
//   layer3.{1,2}: basic block at (64, 8, 8)
//   GAP                                                            → (64,)
//   FC: (10, 64)                                                   → 10 logits
//
// Channel padding: tt-foil's tile is 32×32, so C=16 (layer1, parts of
// layer2.0) gets stored as C=32 with channels 16..31 zeroed; C=3 (RGB)
// likewise pads to C=32. The conv kernels still compute over the
// padded channels — multiplying by zero — and we trust the zeros not
// to bleed into the valid output rows.
//
// Kernel phases (release_kernels between each, see below for sizes):
//   A. Stem + Layer1   : conv_3x3_l1, residual_add_n32, bias_relu_post
//   B. Layer2          : conv_3x3_s2_l2, conv_3x3_l2, residual_add_n8, bias_relu_post
//   C. Layer3          : conv_3x3_s2_l3, conv_3x3_l3, residual_add_n4, bias_relu_post
//   D. Tail            : global_avg_pool, fc, bias_relu_post

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/profiling.h"
#include "cb_config.hpp"
#include "tile_utils.hpp"

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kImageC = 3;
constexpr int kImageH = 32;
constexpr int kImageW = 32;
constexpr int kNumClasses = 10;

// Channel-padded width everywhere (tile-aligned).
constexpr int kPadC16 = 32;     // C=16 layers pad to 32
constexpr int kPadC32 = 32;
constexpr int kPadC64 = 64;

// ---------------------------------------------------------------------------
// Manifest parsing (simple line-oriented format from manifest.txt).
// ---------------------------------------------------------------------------
struct LayerInfo {
    std::string name;
    size_t offset_bytes;
    size_t total_bytes;
    int ndim;
    int shape[4];
};

struct WeightStore {
    std::vector<uint8_t> blob;                          // contents of weights.bin
    std::unordered_map<std::string, LayerInfo> layers;
};

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("can't open " + path);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

WeightStore load_weights(const std::string& data_dir) {
    WeightStore w;
    w.blob = read_file(data_dir + "/weights.bin");

    std::ifstream mf(data_dir + "/manifest.txt");
    if (!mf) throw std::runtime_error("can't open " + data_dir + "/manifest.txt");
    std::string line;
    int num_layers = -1;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (num_layers < 0) { num_layers = std::stoi(line); continue; }
        std::istringstream is(line);
        LayerInfo L;
        is >> L.name >> L.offset_bytes >> L.total_bytes >> L.ndim
           >> L.shape[0] >> L.shape[1] >> L.shape[2] >> L.shape[3];
        w.layers[L.name] = L;
    }
    if (static_cast<int>(w.layers.size()) != num_layers) {
        throw std::runtime_error("manifest layer count mismatch");
    }
    return w;
}

// Read a layer's tensor as a vector of bf16 words (uint16).
std::vector<uint16_t> read_bf16_layer(const WeightStore& w, const std::string& name) {
    auto it = w.layers.find(name);
    if (it == w.layers.end()) throw std::runtime_error("missing layer: " + name);
    const auto& L = it->second;
    std::vector<uint16_t> out(L.total_bytes / 2);
    std::memcpy(out.data(), w.blob.data() + L.offset_bytes, L.total_bytes);
    return out;
}

// ---------------------------------------------------------------------------
// Tile/untile helpers (same shape as the other tt-foil tests).
// ---------------------------------------------------------------------------

void tile_matrix(const std::vector<uint16_t>& m_rm,
                 int rows_t, int cols_t, int col_dim,
                 std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(rows_t) * cols_t * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (int rt = 0; rt < rows_t; ++rt) {
        for (int ct = 0; ct < cols_t; ++ct) {
            for (int r = 0; r < (int)kTileH; ++r)
                for (int c = 0; c < (int)kTileW; ++c)
                    block[r * kTileW + c] =
                        m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

void untile_matrix(const std::vector<uint16_t>& tiles,
                   int rows_t, int cols_t, int col_dim,
                   std::vector<uint16_t>& m_rm) {
    m_rm.assign(static_cast<size_t>(rows_t) * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (int rt = 0; rt < rows_t; ++rt) {
        for (int ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = tiles.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (int r = 0; r < (int)kTileH; ++r)
                for (int c = 0; c < (int)kTileW; ++c)
                    m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Channel padding helpers (Cin → Cin_padded by zero-extending C dim).
// Activations: (C, H, W) row-major; weight: (Cout, Cin, 3, 3).
// ---------------------------------------------------------------------------

void pad_chw(const std::vector<uint16_t>& src, int C, int H, int W, int Cpad,
             std::vector<uint16_t>& dst) {
    dst.assign(static_cast<size_t>(Cpad) * H * W, 0);
    for (int c = 0; c < C; ++c)
        std::memcpy(dst.data() + c * H * W,
                    src.data() + c * H * W, H * W * sizeof(uint16_t));
}

// (Cout, Cin, 3, 3) row-major bf16 weight with Cout-pad and Cin-pad.
void pad_conv_weight(const std::vector<uint16_t>& src,
                     int Cout, int Cin, int K,
                     int CoutPad, int CinPad,
                     std::vector<uint16_t>& dst) {
    dst.assign(static_cast<size_t>(CoutPad) * CinPad * K * K, 0);
    for (int co = 0; co < Cout; ++co)
        for (int ci = 0; ci < Cin; ++ci)
            for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj)
                    dst[((co * CinPad + ci) * K + ki) * K + kj] =
                        src[((co * Cin + ci) * K + ki) * K + kj];
}

// Per-channel bias: pad to Cpad with zeros.
void pad_bias(const std::vector<uint16_t>& src, int C, int Cpad,
              std::vector<uint16_t>& dst) {
    dst.assign(Cpad, 0);
    for (int c = 0; c < C; ++c) dst[c] = src[c];
}

// ---------------------------------------------------------------------------
// 3×3 im2col with arbitrary stride and pad=1.
//   In : (Cpad, Hi, Wi)
//   Out: (Cpad * 9, Ho * Wo) row-major.
// ---------------------------------------------------------------------------
void im2col_3x3(const std::vector<uint16_t>& x_chw,
                int Cpad, int Hi, int Wi, int Ho, int Wo, int stride,
                std::vector<uint16_t>& a) {
    constexpr int kK = 3;
    constexpr int kPad = 1;
    a.assign(static_cast<size_t>(Cpad) * kK * kK * Ho * Wo, 0);
    for (int ci = 0; ci < Cpad; ++ci)
        for (int ki = 0; ki < kK; ++ki)
            for (int kj = 0; kj < kK; ++kj) {
                int row = ci * (kK * kK) + ki * kK + kj;
                for (int ho = 0; ho < Ho; ++ho) {
                    int ih = ho * stride + ki - kPad;
                    if (ih < 0 || ih >= Hi) continue;
                    for (int wo = 0; wo < Wo; ++wo) {
                        int iw = wo * stride + kj - kPad;
                        if (iw < 0 || iw >= Wi) continue;
                        a[row * (Ho * Wo) + ho * Wo + wo] =
                            x_chw[(ci * Hi + ih) * Wi + iw];
                    }
                }
            }
}

// Weight reshape: (CoutPad, CinPad, 3, 3) → (CoutPad, CinPad*9).
void weight_3x3_reshape(const std::vector<uint16_t>& w_cchw,
                        int CoutPad, int CinPad,
                        std::vector<uint16_t>& w_mat) {
    constexpr int kK = 3;
    w_mat.assign(static_cast<size_t>(CoutPad) * CinPad * kK * kK, 0);
    for (int co = 0; co < CoutPad; ++co)
        for (int ci = 0; ci < CinPad; ++ci)
            for (int ki = 0; ki < kK; ++ki)
                for (int kj = 0; kj < kK; ++kj)
                    w_mat[co * (CinPad * kK * kK) + ci * (kK * kK) + ki * kK + kj] =
                        w_cchw[((co * CinPad + ci) * kK + ki) * kK + kj];
}

// (C, H, W) → (H, W, C) for the option-A skip math (we work in CHW
// throughout the activations; the option-A subsample is just a CHW
// stride-2 read followed by a channel-axis pad).
void subsample_stride2_chw(const std::vector<uint16_t>& src,
                           int C, int H, int W,
                           std::vector<uint16_t>& dst) {
    int Ho = H / 2;
    int Wo = W / 2;
    dst.assign(static_cast<size_t>(C) * Ho * Wo, 0);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < Ho; ++h)
            for (int w = 0; w < Wo; ++w)
                dst[(c * Ho + h) * Wo + w] = src[(c * H + h * 2) * W + w * 2];
}

// Option-A channel pad. Akamaster's skip semantics:
//
//   skip[c, h, w] = x_sub[c - pad_each, h, w]   if pad_each ≤ c < pad_each + Cin_real
//                   0                            otherwise
//
// The output buffer is the *tile-padded* width (CoutPad). Anything past
// Cin_real + 2*pad_each up to CoutPad-1 stays zero, which is exactly what
// the residual_add wants for channels the model doesn't use.
void pad_channels_centered(const std::vector<uint16_t>& src,
                           int Cin_real, int H, int W,
                           int pad_each_side, int CoutPad,
                           std::vector<uint16_t>& dst) {
    dst.assign(static_cast<size_t>(CoutPad) * H * W, 0);
    for (int c = 0; c < Cin_real; ++c) {
        int c_new = c + pad_each_side;
        if (c_new >= CoutPad) break;
        std::memcpy(dst.data() + c_new * H * W,
                    src.data() + c * H * W,
                    H * W * sizeof(uint16_t));
    }
}

// ---------------------------------------------------------------------------
// Buffer / lambda factory: a conv stage and a bias_relu_post stage,
// reusable across phases.
// ---------------------------------------------------------------------------

}  // namespace

int main() try {
    const std::string kernel_root = required_env("TT_FOIL_KERNEL_DIR");
    const std::string data_dir    = required_env("TT_FOIL_DATA_DIR");
    const char* dev_env           = std::getenv("TT_FOIL_DEVICE");
    int pcie_index                = dev_env ? std::stoi(dev_env) : 0;

    // ---- Load weights + image + golden -----------------------------
    WeightStore W = load_weights(data_dir);
    auto image_bytes  = read_file(data_dir + "/image.bin");
    auto golden_bytes = read_file(data_dir + "/golden.bin");
    if (image_bytes.size() != size_t(kImageC * kImageH * kImageW * 2))
        throw std::runtime_error("image.bin size unexpected");
    if (golden_bytes.size() != size_t(kNumClasses * 2))
        throw std::runtime_error("golden.bin size unexpected");

    std::vector<uint16_t> img(image_bytes.size() / 2);
    std::memcpy(img.data(), image_bytes.data(), image_bytes.size());
    std::vector<uint16_t> golden(kNumClasses);
    std::memcpy(golden.data(), golden_bytes.data(), golden_bytes.size());

    // ---- Pad input image (3, 32, 32) → (32, 32, 32) -----------------
    std::vector<uint16_t> act;        // current activation in (Cpad, H, W) CHW
    pad_chw(img, kImageC, kImageH, kImageW, kPadC16, act);
    int act_C   = kPadC16;
    int act_H   = kImageH;
    int act_W   = kImageW;

    // ---- Optional stage-by-stage golden comparison ----------------
    const bool debug_stages = std::getenv("TT_FOIL_DEBUG_STAGES") != nullptr;
    auto check_stage = [&](const std::string& stage_name, int C_real, int H, int W) {
        if (!debug_stages) return;
        std::vector<uint8_t> g;
        try {
            g = read_file(data_dir + "/stages/" + stage_name + ".bin");
        } catch (const std::exception&) {
            std::printf("[stage %s] golden missing, skip\n", stage_name.c_str());
            return;
        }
        const size_t expected = static_cast<size_t>(C_real) * H * W * 2;
        if (g.size() != expected) {
            std::printf("[stage %s] golden size mismatch (got %zu, want %zu)\n",
                        stage_name.c_str(), g.size(), expected);
            return;
        }
        std::vector<uint16_t> golden_chw(C_real * H * W);
        std::memcpy(golden_chw.data(), g.data(), g.size());
        float worst = 0.f;
        int worst_idx = -1;
        for (int c = 0; c < C_real; ++c) {
            for (int i = 0; i < H * W; ++i) {
                float dv = bf16_to_f32(act[c * H * W + i]);
                float gv = bf16_to_f32(golden_chw[c * H * W + i]);
                float d = std::fabs(dv - gv);
                if (d > worst) { worst = d; worst_idx = c * H * W + i; }
            }
        }
        float dv = bf16_to_f32(act[worst_idx]);
        float gv = bf16_to_f32(golden_chw[worst_idx]);
        std::printf("[stage %-16s] (C=%d %dx%d) worst abs=%.4f at idx %d (dev=%.4f ref=%.4f)\n",
                    stage_name.c_str(), C_real, H, W, worst, worst_idx, dv, gv);
    };

    // ---- Open device + allocate DRAM scratch -----------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // Largest matmul dims we'll see:
    //   stem/layer1: Mt=1, Kt=9 (Cin=32), Nt=32 (HW=1024 → 32 tiles)
    //   im2col bytes: Kt*Nt = 288 tiles = 576 KB
    //   weight bytes: Mt*Kt = 9 tiles  = 18 KB
    //   output bytes: Mt*Nt = 32 tiles = 64 KB
    // We size DRAM scratch to those maxima so any smaller layer fits.
    constexpr int kMaxKt = 18;     // layer3 Kt
    constexpr int kMaxMt = 2;      // layer3 Mt
    constexpr int kMaxNtStem = 32; // stem/layer1 Nt
    const uint32_t w_bytes_max  = kMaxMt * kMaxKt   * kTileBytes;       // 36 tiles
    const uint32_t a_bytes_max  = kMaxKt * kMaxNtStem * kTileBytes;     // 18*32 = 576 tiles  ← biggest
    const uint32_t y_bytes_max  = kMaxMt * kMaxNtStem * kTileBytes;     // 64 tiles
    // Up to Mt = kPadC64 / 32 = 2 bias tiles for layer3 (Cout=64).
    const uint32_t bias_bytes   = 2 * kTileBytes;
    const uint32_t scaler_bytes = kTileBytes;

    auto buf_W      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes_max, core);
    auto buf_A      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes_max, core);
    auto buf_Y      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes_max, core);
    auto buf_post   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes_max, core);
    auto buf_skip   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes_max, core);
    auto buf_bias_d = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, bias_bytes, core);
    auto buf_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, scaler_bytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    auto noc_of = [&](auto& buf) { return tt::foil::make_noc_dram_addr(*dev, buf->device_addr); };
    uint64_t W_noc      = noc_of(buf_W);
    uint64_t A_noc      = noc_of(buf_A);
    uint64_t Y_noc      = noc_of(buf_Y);
    uint64_t post_noc   = noc_of(buf_post);
    uint64_t skip_noc   = noc_of(buf_skip);
    uint64_t bias_d_noc = noc_of(buf_bias_d);
    uint64_t scaler_noc = noc_of(buf_scaler);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    using R = tt::foil::RiscBinary;
    auto load = [&](const std::string& dir) {
        std::array<R, 5> bins = {{
            {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
            {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
            {R::RiscId::TRISC0, dir + "/compute.trisc0.elf"},
            {R::RiscId::TRISC1, dir + "/compute.trisc1.elf"},
            {R::RiscId::TRISC2, dir + "/compute.trisc2.elf"},
        }};
        return tt::foil::load_kernel(*dev, bins, core);
    };

    std::array<tt::foil::CbConfig, 3> matmul_cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};

    // Kernel handles (rebound per phase).
    std::shared_ptr<tt::foil::Kernel> k_conv_s1, k_conv_s2, k_add, k_bias,
                                       k_gap, k_fc;

    // ---- Helpers ----------------------------------------------------

    // Pack a per-channel bias vector (length Mt * 32 = Cpad) into Mt
    // contiguous 32×32 bf16 tiles, channel values in column 0.
    auto pack_bias_tile = [&](const std::vector<uint16_t>& bias_cpad,
                              int Mt,
                              std::vector<uint16_t>& tiles_out) {
        tiles_out.clear();
        for (int mt = 0; mt < Mt; ++mt) {
            std::vector<uint16_t> rm(kTileH * kTileW, 0);
            for (uint32_t r = 0; r < kTileH; ++r) {
                size_t idx = mt * kTileH + r;
                if (idx < bias_cpad.size())
                    rm[r * kTileW + 0] = bias_cpad[idx];
            }
            tt::foil::test::row_major_to_tile(rm.data(), tiles_out);
        }
    };

    // Run a matmul-shaped conv:
    //   W tiled as (Mt, Kt) at column dim = Kt*32, A tiled as (Kt, Nt)
    //   at column dim = Nt*32. Output tiled as (Mt, Nt) at column dim
    //   HW = Nt*32. Returns the post-conv activation in CHW layout
    //   shape (Mt*32, HW_out).
    auto run_conv = [&](tt::foil::Kernel& k_conv,
                        int Mt, int Kt, int Nt,
                        int CinPad, int Hi, int Wi, int Ho, int Wo, int stride,
                        const std::vector<uint16_t>& w_cchw,    // (CoutPad, CinPad, 3, 3)
                        const std::vector<uint16_t>& x_chw,     // (CinPad, Hi, Wi)
                        std::vector<uint16_t>& y_chw_out) {
        const int CoutPad = Mt * (int)kTileH;
        const int HWout   = Ho * Wo;
        std::vector<uint16_t> w_mat, w_tiles, a_mat, a_tiles;
        weight_3x3_reshape(w_cchw, CoutPad, CinPad, w_mat);
        const int Kdim = CinPad * 3 * 3;
        tile_matrix(w_mat, Mt, Kt, Kdim, w_tiles);
        im2col_3x3(x_chw, CinPad, Hi, Wi, Ho, Wo, stride, a_mat);
        tile_matrix(a_mat, Kt, Nt, HWout, a_tiles);

        const uint32_t wb = static_cast<uint32_t>(Mt * Kt) * kTileBytes;
        const uint32_t ab = static_cast<uint32_t>(Kt * Nt) * kTileBytes;
        const uint32_t yb = static_cast<uint32_t>(Mt * Nt) * kTileBytes;
        tt::foil::write_buffer(*dev, *buf_W, w_tiles.data(), wb);
        tt::foil::write_buffer(*dev, *buf_A, a_tiles.data(), ab);
        std::vector<uint8_t> zero(yb, 0);
        tt::foil::write_buffer(*dev, *buf_Y, zero.data(), yb);

        std::array<uint32_t, 7> rab = {
            lo(W_noc), hi(W_noc), lo(A_noc), hi(A_noc),
            (uint32_t)Mt, (uint32_t)Kt, (uint32_t)Nt,
        };
        std::array<uint32_t, 3> ran = { lo(Y_noc), hi(Y_noc),
                                        (uint32_t)(Mt * Nt) };
        tt::foil::set_runtime_args(*dev, k_conv, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, k_conv, R::RiscId::NCRISC, ran);
        tt::foil::register_cbs(*dev, k_conv, matmul_cbs);
        tt::foil::execute(*dev, k_conv);

        std::vector<uint16_t> y_tiles(static_cast<size_t>(Mt) * Nt * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_Y, y_tiles.data(), yb);
        untile_matrix(y_tiles, Mt, Nt, HWout, y_chw_out);
    };

    // Run bias_relu_post on an (Mt*32, HW)-shaped CHW activation,
    // returning the result in-place. Uses buf_Y as input staging and
    // buf_post as output staging. bias_cpad must hold Mt*32 entries
    // — channel biases for each Mt row of the tile grid.
    auto run_bias_relu = [&](tt::foil::Kernel& k,
                             std::vector<uint16_t>& chw_inout,
                             int Mt, int Nt, int HW,
                             const std::vector<uint16_t>& bias_cpad,
                             uint32_t relu_enable) {
        const int n_tiles = Mt * Nt;
        const uint32_t bytes = n_tiles * kTileBytes;
        std::vector<uint16_t> in_tiles;
        tile_matrix(chw_inout, Mt, Nt, HW, in_tiles);
        tt::foil::write_buffer(*dev, *buf_Y, in_tiles.data(), bytes);

        std::vector<uint16_t> bt;
        pack_bias_tile(bias_cpad, Mt, bt);
        tt::foil::write_buffer(*dev, *buf_bias_d, bt.data(),
                               Mt * kTileBytes);

        std::vector<uint8_t> zero(bytes, 0);
        tt::foil::write_buffer(*dev, *buf_post, zero.data(), bytes);

        std::array<uint32_t, 6> rab = {
            lo(Y_noc),      hi(Y_noc),
            lo(bias_d_noc), hi(bias_d_noc),
            (uint32_t)Mt, (uint32_t)Nt,
        };
        std::array<uint32_t, 3> ran = { lo(post_noc), hi(post_noc),
                                        (uint32_t)n_tiles };
        std::array<uint32_t, 2> rac = { (uint32_t)n_tiles, relu_enable };
        tt::foil::set_runtime_args(*dev, k, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::NCRISC, ran);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC0, rac);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC1, rac);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::TRISC2, rac);
        tt::foil::register_cbs(*dev, k, matmul_cbs);
        tt::foil::execute(*dev, k);

        std::vector<uint16_t> out_tiles(n_tiles * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, out_tiles.data(), bytes);
        untile_matrix(out_tiles, Mt, Nt, HW, chw_inout);
    };

    // Run residual_add: y_chw = a_chw + b_chw on device.
    auto run_add = [&](tt::foil::Kernel& k,
                       const std::vector<uint16_t>& a_chw,
                       const std::vector<uint16_t>& b_chw,
                       int Mt, int Nt, int HW,
                       std::vector<uint16_t>& y_chw) {
        const int n_tiles = Mt * Nt;
        const uint32_t bytes = n_tiles * kTileBytes;
        std::vector<uint16_t> a_tiles, b_tiles;
        tile_matrix(a_chw, Mt, Nt, HW, a_tiles);
        tile_matrix(b_chw, Mt, Nt, HW, b_tiles);
        tt::foil::write_buffer(*dev, *buf_Y,    a_tiles.data(), bytes);
        tt::foil::write_buffer(*dev, *buf_skip, b_tiles.data(), bytes);
        std::vector<uint8_t> zero(bytes, 0);
        tt::foil::write_buffer(*dev, *buf_post, zero.data(), bytes);

        std::array<uint32_t, 4> rab = {
            lo(Y_noc),    hi(Y_noc),
            lo(skip_noc), hi(skip_noc),
        };
        std::array<uint32_t, 2> ran = { lo(post_noc), hi(post_noc) };
        tt::foil::set_runtime_args(*dev, k, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, k, R::RiscId::NCRISC, ran);
        tt::foil::register_cbs(*dev, k, matmul_cbs);
        tt::foil::execute(*dev, k);

        std::vector<uint16_t> y_tiles(n_tiles * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, y_tiles.data(), bytes);
        untile_matrix(y_tiles, Mt, Nt, HW, y_chw);
    };

    // One basic block (or downsample block with option-A skip): two
    // convs, optional skip-shape change, residual add, ReLU. The
    // calling phase has the right conv kernels loaded.
    //
    // `conv1_kernel` may be the stride-2 variant for the first block of
    // a layer; `Cin_in_pad` is the padded input channels and
    // `Cin_out_pad` the post-conv1 channels (== Cout_block).
    auto run_block = [&](tt::foil::Kernel& conv1_k, tt::foil::Kernel& conv2_k,
                         tt::foil::Kernel& add_k, tt::foil::Kernel& bias_k,
                         int Mt_out, int Kt1, int Kt2, int Nt_out,
                         int CinPad, int CoutPad,
                         int Cin_real, int Cout_real,
                         int Hi, int Wi, int Ho, int Wo, int stride_first,
                         bool downsample,
                         const std::vector<uint16_t>& w1_padded,
                         const std::vector<uint16_t>& b1_padded,
                         const std::vector<uint16_t>& w2_padded,
                         const std::vector<uint16_t>& b2_padded) {
        const int HWout = Ho * Wo;

        // Stage 1: conv1 + bias + ReLU
        std::vector<uint16_t> t1;
        run_conv(conv1_k,
                 Mt_out, Kt1, Nt_out,
                 CinPad, Hi, Wi, Ho, Wo, stride_first,
                 w1_padded, act, t1);
        run_bias_relu(bias_k, t1, Mt_out, Nt_out, HWout, b1_padded, 1);

        // Stage 2: conv2 + bias (no ReLU yet)
        std::vector<uint16_t> t2;
        run_conv(conv2_k,
                 Mt_out, Kt2, Nt_out,
                 CoutPad, Ho, Wo, Ho, Wo, /*stride=*/1,
                 w2_padded, t1, t2);
        run_bias_relu(bias_k, t2, Mt_out, Nt_out, HWout, b2_padded, 0);

        // Stage 3: prep skip path
        std::vector<uint16_t> skip;
        if (!downsample) {
            // Identity skip = act (the original block input).
            skip = act;
        } else {
            // Option A (akamaster): subsample stride-2 in spatial, then
            // option-A channel pad with `pad_each = (Cout_real-Cin_real)/2`
            // zeros on each side of the real channels. Real channels land
            // at indices [pad_each, pad_each + Cin_real); the rest of
            // the (CoutPad, Ho, Wo) layout stays zero.
            std::vector<uint16_t> sub;
            subsample_stride2_chw(act, CinPad, Hi, Wi, sub);
            int pad_each = (Cout_real - Cin_real) / 2;
            pad_channels_centered(sub, Cin_real, Ho, Wo,
                                  pad_each, CoutPad, skip);
        }

        // Stage 4: residual add (t2 + skip) → buf_post
        std::vector<uint16_t> y;
        run_add(add_k, t2, skip, Mt_out, Nt_out, HWout, y);

        // Stage 5: ReLU (bias=0, relu=1)
        std::vector<uint16_t> zero_bias(Mt_out * kTileH, 0);
        run_bias_relu(bias_k, y, Mt_out, Nt_out, HWout, zero_bias, 1);

        // Update current activation.
        act   = std::move(y);
        act_C = CoutPad;
        act_H = Ho;
        act_W = Wo;
    };

    // ================================================================
    // Phase A — Stem + Layer1
    // Both use conv_3x3_l1 (Mt=1 Kt=9 Nt=32) and residual_add_n32.
    // ================================================================
    { TT_FOIL_ZONE("Phase_A_stem_layer1");
    k_conv_s1 = load(kernel_root + "/conv_3x3_l1");
    k_add     = load(kernel_root + "/residual_add_n32");
    k_bias    = load(kernel_root + "/bias_relu_post");

    // Stem: conv 3×3 s=1 (3→16, 32×32). Cin=3 padded to 32, Cout=16 padded to 32.
    {
        auto w_raw = read_bf16_layer(W, "stem.conv.w");      // (16, 3, 3, 3)
        auto b_raw = read_bf16_layer(W, "stem.bn.b");        // (16,)
        std::vector<uint16_t> w_p, b_p;
        pad_conv_weight(w_raw, /*Cout=*/16, /*Cin=*/3, /*K=*/3,
                        /*CoutPad=*/kPadC16, /*CinPad=*/kPadC16, w_p);
        pad_bias(b_raw, 16, kPadC16, b_p);

        std::vector<uint16_t> y;
        run_conv(*k_conv_s1, /*Mt=*/1, /*Kt=*/9, /*Nt=*/32,
                 /*CinPad=*/kPadC16, /*Hi=*/32, /*Wi=*/32,
                 /*Ho=*/32, /*Wo=*/32, /*stride=*/1,
                 w_p, act, y);
        run_bias_relu(*k_bias, y, /*Mt=*/1, /*Nt=*/32, /*HW=*/1024, b_p, 1);
        act   = std::move(y);
        act_C = kPadC16;
        act_H = 32;
        act_W = 32;
        check_stage("post_stem", 16, 32, 32);
    }

    // Layer 1: 3 basic blocks at (16-padded, 32×32). Each uses
    // conv_3x3_l1 for both convs.
    for (int b = 0; b < 3; ++b) {
        std::string base = "layer1." + std::to_string(b);
        auto w1 = read_bf16_layer(W, base + ".conv1.w");
        auto b1 = read_bf16_layer(W, base + ".bn1.b");
        auto w2 = read_bf16_layer(W, base + ".conv2.w");
        auto b2 = read_bf16_layer(W, base + ".bn2.b");
        std::vector<uint16_t> w1p, b1p, w2p, b2p;
        pad_conv_weight(w1, 16, 16, 3, kPadC16, kPadC16, w1p);
        pad_conv_weight(w2, 16, 16, 3, kPadC16, kPadC16, w2p);
        pad_bias(b1, 16, kPadC16, b1p);
        pad_bias(b2, 16, kPadC16, b2p);
        run_block(*k_conv_s1, *k_conv_s1, *k_add, *k_bias,
                  /*Mt_out=*/1, /*Kt1=*/9, /*Kt2=*/9, /*Nt_out=*/32,
                  /*CinPad=*/kPadC16, /*CoutPad=*/kPadC16,
                  /*Cin_real=*/16, /*Cout_real=*/16,
                  32, 32, 32, 32, /*stride_first=*/1, /*downsample=*/false,
                  w1p, b1p, w2p, b2p);
        check_stage("post_layer1." + std::to_string(b), 16, 32, 32);
    }

    }  // end Phase A
    // ================================================================
    // Phase B — Layer2 (3 blocks; first is downsample)
    // ================================================================
    { TT_FOIL_ZONE("Phase_B_layer2");
    k_conv_s1.reset(); k_add.reset(); k_bias.reset();
    tt::foil::release_kernels(*dev, core);

    k_conv_s2 = load(kernel_root + "/conv_3x3_s2_l2");      // Mt=1 Kt=9 Nt=8
    k_conv_s1 = load(kernel_root + "/conv_3x3_l2");         // Mt=1 Kt=9 Nt=8
    k_add     = load(kernel_root + "/residual_add_n8");
    k_bias    = load(kernel_root + "/bias_relu_post");

    {
        // layer2.0 — downsample (16→32 channels, 32×32 → 16×16 spatial)
        auto w1 = read_bf16_layer(W, "layer2.0.conv1.w");
        auto b1 = read_bf16_layer(W, "layer2.0.bn1.b");
        auto w2 = read_bf16_layer(W, "layer2.0.conv2.w");
        auto b2 = read_bf16_layer(W, "layer2.0.bn2.b");
        std::vector<uint16_t> w1p, b1p, w2p, b2p;
        // conv1: Cin=16 padded to 32, Cout=32 → 32
        pad_conv_weight(w1, 32, 16, 3, kPadC32, kPadC16, w1p);
        // conv2: Cin=32, Cout=32 (no padding needed but use 32/32)
        pad_conv_weight(w2, 32, 32, 3, kPadC32, kPadC32, w2p);
        pad_bias(b1, 32, kPadC32, b1p);
        pad_bias(b2, 32, kPadC32, b2p);

        run_block(*k_conv_s2, *k_conv_s1, *k_add, *k_bias,
                  /*Mt_out=*/1, /*Kt1=*/9, /*Kt2=*/9, /*Nt_out=*/8,
                  /*CinPad=*/kPadC16, /*CoutPad=*/kPadC32,
                  /*Cin_real=*/16, /*Cout_real=*/32,
                  /*Hi=*/32, /*Wi=*/32, /*Ho=*/16, /*Wo=*/16,
                  /*stride_first=*/2, /*downsample=*/true,
                  w1p, b1p, w2p, b2p);
        check_stage("post_layer2.0", 32, 16, 16);
    }

    for (int b = 1; b <= 2; ++b) {
        std::string base = "layer2." + std::to_string(b);
        auto w1 = read_bf16_layer(W, base + ".conv1.w");
        auto b1 = read_bf16_layer(W, base + ".bn1.b");
        auto w2 = read_bf16_layer(W, base + ".conv2.w");
        auto b2 = read_bf16_layer(W, base + ".bn2.b");
        std::vector<uint16_t> w1p, b1p, w2p, b2p;
        pad_conv_weight(w1, 32, 32, 3, kPadC32, kPadC32, w1p);
        pad_conv_weight(w2, 32, 32, 3, kPadC32, kPadC32, w2p);
        pad_bias(b1, 32, kPadC32, b1p);
        pad_bias(b2, 32, kPadC32, b2p);
        run_block(*k_conv_s1, *k_conv_s1, *k_add, *k_bias,
                  /*Mt_out=*/1, /*Kt1=*/9, /*Kt2=*/9, /*Nt_out=*/8,
                  /*CinPad=*/kPadC32, /*CoutPad=*/kPadC32,
                  /*Cin_real=*/32, /*Cout_real=*/32,
                  16, 16, 16, 16, /*stride_first=*/1, /*downsample=*/false,
                  w1p, b1p, w2p, b2p);
        check_stage("post_layer2." + std::to_string(b), 32, 16, 16);
    }

    }  // end Phase B
    // ================================================================
    // Phase C — Layer3 (3 blocks; first is downsample)
    // ================================================================
    { TT_FOIL_ZONE("Phase_C_layer3");
    k_conv_s2.reset(); k_conv_s1.reset(); k_add.reset(); k_bias.reset();
    tt::foil::release_kernels(*dev, core);

    k_conv_s2 = load(kernel_root + "/conv_3x3_s2_l3");      // Mt=2 Kt=9 Nt=2
    k_conv_s1 = load(kernel_root + "/conv_3x3_l3");         // Mt=2 Kt=18 Nt=2
    k_add     = load(kernel_root + "/residual_add_n4");
    k_bias    = load(kernel_root + "/bias_relu_post");

    {
        // layer3.0 — downsample (32→64 channels, 16×16 → 8×8 spatial)
        auto w1 = read_bf16_layer(W, "layer3.0.conv1.w");
        auto b1 = read_bf16_layer(W, "layer3.0.bn1.b");
        auto w2 = read_bf16_layer(W, "layer3.0.conv2.w");
        auto b2 = read_bf16_layer(W, "layer3.0.bn2.b");
        std::vector<uint16_t> w1p, b1p, w2p, b2p;
        pad_conv_weight(w1, 64, 32, 3, kPadC64, kPadC32, w1p);
        pad_conv_weight(w2, 64, 64, 3, kPadC64, kPadC64, w2p);
        pad_bias(b1, 64, kPadC64, b1p);
        pad_bias(b2, 64, kPadC64, b2p);

        run_block(*k_conv_s2, *k_conv_s1, *k_add, *k_bias,
                  /*Mt_out=*/2, /*Kt1=*/9, /*Kt2=*/18, /*Nt_out=*/2,
                  /*CinPad=*/kPadC32, /*CoutPad=*/kPadC64,
                  /*Cin_real=*/32, /*Cout_real=*/64,
                  /*Hi=*/16, /*Wi=*/16, /*Ho=*/8, /*Wo=*/8,
                  /*stride_first=*/2, /*downsample=*/true,
                  w1p, b1p, w2p, b2p);
        check_stage("post_layer3.0", 64, 8, 8);
    }

    for (int b = 1; b <= 2; ++b) {
        std::string base = "layer3." + std::to_string(b);
        auto w1 = read_bf16_layer(W, base + ".conv1.w");
        auto b1 = read_bf16_layer(W, base + ".bn1.b");
        auto w2 = read_bf16_layer(W, base + ".conv2.w");
        auto b2 = read_bf16_layer(W, base + ".bn2.b");
        std::vector<uint16_t> w1p, b1p, w2p, b2p;
        pad_conv_weight(w1, 64, 64, 3, kPadC64, kPadC64, w1p);
        pad_conv_weight(w2, 64, 64, 3, kPadC64, kPadC64, w2p);
        pad_bias(b1, 64, kPadC64, b1p);
        pad_bias(b2, 64, kPadC64, b2p);
        run_block(*k_conv_s1, *k_conv_s1, *k_add, *k_bias,
                  /*Mt_out=*/2, /*Kt1=*/18, /*Kt2=*/18, /*Nt_out=*/2,
                  /*CinPad=*/kPadC64, /*CoutPad=*/kPadC64,
                  /*Cin_real=*/64, /*Cout_real=*/64,
                  8, 8, 8, 8, /*stride_first=*/1, /*downsample=*/false,
                  w1p, b1p, w2p, b2p);
        check_stage("post_layer3." + std::to_string(b), 64, 8, 8);
    }

    }  // end Phase C
    // ================================================================
    // Phase D — Tail: GAP + FC + bias
    // ================================================================
    // logits_padded outlives the Phase D zone so the host-side argmax
    // below (which we explicitly do NOT want inside the phase profile)
    // can read it.
    std::vector<uint16_t> logits_padded(kTileH, 0);
    { TT_FOIL_ZONE("Phase_D_tail");
    k_conv_s2.reset(); k_conv_s1.reset(); k_add.reset(); k_bias.reset();
    tt::foil::release_kernels(*dev, core);

    k_gap  = load(kernel_root + "/global_avg_pool");
    k_fc   = load(kernel_root + "/fc");
    k_bias = load(kernel_root + "/bias_relu_post");

    // act is now (64, 8, 8) → tile as Mt=2, Nt=2 with HW=64. Run GAP
    // twice, once per Mt slice (the stock global_avg_pool kernel
    // emits exactly one output tile).
    std::vector<uint16_t> gap_tiles(2 * kTileWords, 0);
    {
        const int Mt_act = 2, Nt_act = 2;
        const int HW = 64;
        std::vector<uint16_t> act_tiles;
        tile_matrix(act, Mt_act, Nt_act, HW, act_tiles);

        // Scaler tile = 1/HW across the whole tile.
        std::vector<uint16_t> scaler_rm(kTileH * kTileW,
            f32_to_bf16(1.0f / static_cast<float>(HW)));
        std::vector<uint16_t> scaler_tiles;
        tt::foil::test::row_major_to_tile(scaler_rm.data(), scaler_tiles);
        tt::foil::write_buffer(*dev, *buf_scaler, scaler_tiles.data(), kTileBytes);

        const uint32_t per_mt_in_bytes = Nt_act * kTileBytes;
        for (int mt = 0; mt < Mt_act; ++mt) {
            // Stage this Mt's input tiles at the start of buf_A.
            tt::foil::write_buffer(*dev, *buf_A,
                act_tiles.data() + mt * Nt_act * kTileWords, per_mt_in_bytes);
            std::vector<uint8_t> zero(kTileBytes, 0);
            tt::foil::write_buffer(*dev, *buf_post, zero.data(), kTileBytes);

            std::array<uint32_t, 5> rab = {
                lo(A_noc),      hi(A_noc),
                lo(scaler_noc), hi(scaler_noc),
                (uint32_t)Nt_act,
            };
            std::array<uint32_t, 2> ran = { lo(post_noc), hi(post_noc) };
            std::array<uint32_t, 1> rac = { (uint32_t)Nt_act };
            tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::BRISC,  rab);
            tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::NCRISC, ran);
            tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC0, rac);
            tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC1, rac);
            tt::foil::set_runtime_args(*dev, *k_gap, R::RiscId::TRISC2, rac);
            tt::foil::register_cbs(*dev, *k_gap, matmul_cbs);
            tt::foil::execute(*dev, *k_gap);

            tt::foil::read_buffer(*dev, *buf_post,
                gap_tiles.data() + mt * kTileWords, kTileBytes);
        }
    }

    // FC matmul: W shape (10, 64) padded to (32, 64) = (Mt=1, Kt=2).
    // X is the 2-tile column vector formed by stacking gap_tiles
    // vertically: rows 0..63 of column 0 carry the GAP means, others
    // zero. Output (32, 32) — only column 0, rows 0..9 are the real
    // logits.
    {
        auto wfc = read_bf16_layer(W, "fc.w");                  // (10, 64)
        auto bfc = read_bf16_layer(W, "fc.b");                  // (10,)
        // Pad weight to (32, 64).
        std::vector<uint16_t> Wmat(32 * 64, 0);
        for (int r = 0; r < 10; ++r)
            for (int c = 0; c < 64; ++c)
                Wmat[r * 64 + c] = wfc[r * 64 + c];

        std::vector<uint16_t> w_tiles;
        tile_matrix(Wmat, /*rows_t=*/1, /*cols_t=*/2, /*col_dim=*/64, w_tiles);

        // X operand: gap_tiles is already in the right "channel value
        // in column 0" layout; we just stack the two channel-slice
        // tiles as Kt=2.
        const uint32_t Wbytes = 2 * kTileBytes;
        tt::foil::write_buffer(*dev, *buf_W,    w_tiles.data(),    Wbytes);
        tt::foil::write_buffer(*dev, *buf_A,    gap_tiles.data(),  2 * kTileBytes);
        std::vector<uint8_t> zero(kTileBytes, 0);
        tt::foil::write_buffer(*dev, *buf_Y, zero.data(), kTileBytes);

        std::array<uint32_t, 7> rab = {
            lo(W_noc), hi(W_noc), lo(A_noc), hi(A_noc),
            /*Mt=*/1u, /*Kt=*/2u, /*Nt=*/1u,
        };
        std::array<uint32_t, 3> ran = { lo(Y_noc), hi(Y_noc), 1u };
        tt::foil::set_runtime_args(*dev, *k_fc, R::RiscId::BRISC,  rab);
        tt::foil::set_runtime_args(*dev, *k_fc, R::RiscId::NCRISC, ran);
        tt::foil::register_cbs(*dev, *k_fc, matmul_cbs);
        tt::foil::execute(*dev, *k_fc);

        // FC bias add (relu=0, Mt=1, Nt=1).
        std::vector<uint16_t> b_padded(kTileH, 0);
        for (int k = 0; k < 10; ++k) b_padded[k] = bfc[k];
        std::vector<uint16_t> bt;
        pack_bias_tile(b_padded, /*Mt=*/1, bt);
        tt::foil::write_buffer(*dev, *buf_bias_d, bt.data(), kTileBytes);
        tt::foil::write_buffer(*dev, *buf_post, zero.data(), kTileBytes);

        std::array<uint32_t, 6> brb = {
            lo(Y_noc),      hi(Y_noc),
            lo(bias_d_noc), hi(bias_d_noc),
            1u, 1u,  // Mt=1, Nt=1
        };
        std::array<uint32_t, 3> brn = { lo(post_noc), hi(post_noc), 1u };
        std::array<uint32_t, 2> brc = { 1u, /*relu=*/0u };
        tt::foil::set_runtime_args(*dev, *k_bias, R::RiscId::BRISC,  brb);
        tt::foil::set_runtime_args(*dev, *k_bias, R::RiscId::NCRISC, brn);
        tt::foil::set_runtime_args(*dev, *k_bias, R::RiscId::TRISC0, brc);
        tt::foil::set_runtime_args(*dev, *k_bias, R::RiscId::TRISC1, brc);
        tt::foil::set_runtime_args(*dev, *k_bias, R::RiscId::TRISC2, brc);
        tt::foil::register_cbs(*dev, *k_bias, matmul_cbs);
        tt::foil::execute(*dev, *k_bias);

        // Read tile, extract column 0 rows 0..9 as logits.
        std::vector<uint16_t> out_tile(kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_post, out_tile.data(), kTileBytes);
        std::vector<uint16_t> out_rm(kTileH * kTileW);
        tt::foil::test::tile_to_row_major(out_tile.data(), out_rm.data());
        for (int k = 0; k < kNumClasses; ++k)
            logits_padded[k] = out_rm[k * kTileW + 0];
    }
    }  // end Phase D

    // ---- Compare to golden + report argmax -------------------------
    std::vector<float> dev_logits(kNumClasses), ref_logits(kNumClasses);
    for (int k = 0; k < kNumClasses; ++k) {
        dev_logits[k] = bf16_to_f32(logits_padded[k]);
        ref_logits[k] = bf16_to_f32(golden[k]);
    }
    int argmax_dev = 0, argmax_ref = 0;
    for (int k = 1; k < kNumClasses; ++k) {
        if (dev_logits[k] > dev_logits[argmax_dev]) argmax_dev = k;
        if (ref_logits[k] > ref_logits[argmax_ref]) argmax_ref = k;
    }

    float worst_abs = 0.f;
    for (int k = 0; k < kNumClasses; ++k)
        worst_abs = std::max(worst_abs, std::fabs(dev_logits[k] - ref_logits[k]));

    static const char* kClassNames[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    };

    std::printf("test_cifar10_resnet20: dev_logits = [");
    for (int k = 0; k < kNumClasses; ++k) std::printf(" %.3f", dev_logits[k]);
    std::printf(" ]\n");
    std::printf("                       ref_logits = [");
    for (int k = 0; k < kNumClasses; ++k) std::printf(" %.3f", ref_logits[k]);
    std::printf(" ]\n");
    std::printf("                       worst abs = %.4f\n", worst_abs);

    bool argmax_match = (argmax_dev == argmax_ref);
    std::printf("test_cifar10_resnet20: %s  argmax dev=%d (%s) ref=%d (%s)\n",
                argmax_match ? "PASS" : "FAIL (argmax mismatch)",
                argmax_dev, kClassNames[argmax_dev],
                argmax_ref, kClassNames[argmax_ref]);

    tt::foil::close_device(std::move(dev));
    return argmax_match ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_cifar10_resnet20: FAIL — %s\n", e.what());
    return 1;
}
