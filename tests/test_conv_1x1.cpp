// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v7-1: bf16 1×1 convolution implemented as a matmul.
//
// A 1×1 conv with N=1 batch is:
//   Y[c_out, h, w] = sum_{c_in} W[c_out, c_in] * X[c_in, h, w]
//
// Flatten X to shape (C_in, H*W) row-major and use W as (C_out, C_in)
// row-major. Then Y_flat = W @ X_flat has shape (C_out, H*W) and is
// the same matmul kernel we already use for matmul_dram. The test:
//
//   1. Generate host tensors X (C_in, H, W) and W (C_out, C_in).
//   2. Compute reference conv on the host in fp32.
//   3. Reshape X CHW → matrix (C_in, H*W) row-major (no value change,
//      just stride/layout interpretation).
//   4. Tile both matrices and write to DRAM.
//   5. Run matmul_dram-style kernel; output is (C_out, H*W) tiled.
//   6. Untile + reshape back to (C_out, H, W); compare with reference.
//
// Tile alignment requirements (all multiples of 32):
//   C_in   = MM_KT * 32
//   C_out  = MM_MT * 32
//   H * W  = MM_NT * 32

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

#ifndef MM_MT
#define MM_MT 2
#endif
#ifndef MM_KT
#define MM_KT 2
#endif
#ifndef MM_NT
#define MM_NT 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kMt    = MM_MT;
constexpr uint32_t kKt    = MM_KT;
constexpr uint32_t kNt    = MM_NT;
constexpr uint32_t kCout  = kMt * kTileH;       // M = C_out
constexpr uint32_t kCin   = kKt * kTileW;       // K = C_in
constexpr uint32_t kHW    = kNt * kTileW;       // N = H * W

// Derive (H, W) from kHW. Use H=8 as a fixed factor and W = kHW / 8 so
// the test scales with MM_NT without manual edits. kHW is always a
// multiple of 32 (one tile), so dividing by 8 is integer-safe.
constexpr uint32_t kH = 8;
constexpr uint32_t kW = kHW / kH;
static_assert(kH * kW == kHW, "H must divide MM_NT * 32");

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Reference conv (also reused to make the matmul reference): X (CHW),
// W (C_out, C_in) → Y (C_out, H, W), fp32 accumulate, bf16 round.
void conv_1x1_reference(const std::vector<uint16_t>& x_chw,
                        const std::vector<uint16_t>& w,
                        std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * kH * kW, 0);
    for (uint32_t co = 0; co < kCout; ++co) {
        for (uint32_t h = 0; h < kH; ++h) {
            for (uint32_t w_ = 0; w_ < kW; ++w_) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kCin; ++ci) {
                    float xv = bf16_to_f32(x_chw[(ci * kH + h) * kW + w_]);
                    float wv = bf16_to_f32(w[co * kCin + ci]);
                    acc += xv * wv;
                }
                y_chw[(co * kH + h) * kW + w_] = f32_to_bf16(acc);
            }
        }
    }
}

// CHW → (C, H*W) matrix view: just flatten the H, W dims. Bytes are
// unchanged in memory if we treat the host buffer as row-major
// (C, H*W). This helper makes that explicit for readability.
void chw_to_matrix(const std::vector<uint16_t>& x_chw, std::vector<uint16_t>& m) {
    m.resize(kCin * kHW);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t i = 0; i < kHW; ++i)
            m[ci * kHW + i] = x_chw[ci * kHW + i];   // identity copy
}

// Inverse for the output: (C_out, H*W) → CHW (also identity layout).
void matrix_to_chw(const std::vector<uint16_t>& m, std::vector<uint16_t>& y_chw) {
    y_chw.resize(kCout * kHW);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t i = 0; i < kHW; ++i)
            y_chw[co * kHW + i] = m[co * kHW + i];
}

void tile_stream(const std::vector<uint16_t>& m_rm, uint32_t rows_t, uint32_t cols_t,
                 uint32_t row_dim, uint32_t col_dim,
                 std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(rows_t) * cols_t * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt) {
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] =
                        m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
            (void)row_dim;
        }
    }
}

void untile_stream(const std::vector<uint16_t>& tiles, uint32_t rows_t, uint32_t cols_t,
                   uint32_t col_dim, std::vector<uint16_t>& m_rm) {
    m_rm.assign(rows_t * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt) {
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = tiles.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t w_bytes   = kMt * kKt * kTileBytes;
    const uint32_t x_bytes   = kKt * kNt * kTileBytes;
    const uint32_t y_bytes   = kMt * kNt * kTileBytes;

    // ---- Host tensors -----------------------------------------------
    // X in CHW (C_in, H, W). W in (C_out, C_in). Same value formula
    // as the matmul tests so we stay in the regime where 1 bf16 ULP
    // worst case is the expectation.
    std::vector<uint16_t> x_chw(kCin * kH * kW);
    std::vector<uint16_t> w_mat(kCout * kCin);
    for (uint32_t ci = 0; ci < kCin; ++ci) {
        for (uint32_t h = 0; h < kH; ++h) {
            for (uint32_t w_ = 0; w_ < kW; ++w_) {
                uint32_t idx = (ci * kH + h) * kW + w_;
                x_chw[idx] = f32_to_bf16(0.005f * static_cast<float>(
                    (ci + 1) + ((h * kW + w_) % 7)));
            }
        }
    }
    for (uint32_t co = 0; co < kCout; ++co) {
        for (uint32_t ci = 0; ci < kCin; ++ci) {
            w_mat[co * kCin + ci] = f32_to_bf16(0.005f * static_cast<float>(
                (co + 1) + (ci % 5)));
        }
    }

    // ---- Reference --------------------------------------------------
    std::vector<uint16_t> y_ref_chw;
    conv_1x1_reference(x_chw, w_mat, y_ref_chw);

    // ---- Reshape + tile for the matmul kernel -----------------------
    // The matmul kernel computes C = A @ B where A is (M, K) and
    // B is (K, N). We feed:
    //   A = W   (C_out × C_in)   → Mt × Kt tiles
    //   B = X'  (C_in × H*W)     → Kt × Nt tiles, where X' is the
    //                              CHW→matrix reshape of X.
    std::vector<uint16_t> x_mat;     // (C_in, H*W)
    chw_to_matrix(x_chw, x_mat);

    std::vector<uint16_t> a_stream;  // W tiles, Mt × Kt
    std::vector<uint16_t> b_stream;  // X' tiles, Kt × Nt
    tile_stream(w_mat, /*rows_t=*/kMt, /*cols_t=*/kKt,
                /*row_dim=*/kCout, /*col_dim=*/kCin, a_stream);
    tile_stream(x_mat, /*rows_t=*/kKt, /*cols_t=*/kNt,
                /*row_dim=*/kCin, /*col_dim=*/kHW, b_stream);

    // ---- Device -----------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes, core);
    auto buf_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes, core);
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, core);

    auto buf_cb_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), w_bytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), x_bytes);
    std::vector<uint8_t> zero(y_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), y_bytes);

    uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, buf_a  ->device_addr);
    uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, buf_b  ->device_addr);
    uint64_t out_noc = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    std::array<uint32_t, 7> ra_brisc = {
        lo(a_noc), hi(a_noc),
        lo(b_noc), hi(b_noc),
        kMt, kKt, kNt,
    };
    std::array<uint32_t, 3> ra_ncrisc = {
        lo(out_noc), hi(out_noc), kMt * kNt,
    };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> y_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), y_bytes);

    std::vector<uint16_t> y_mat;     // (C_out, H*W) row-major
    untile_stream(y_tiles, /*rows_t=*/kMt, /*cols_t=*/kNt,
                  /*col_dim=*/kHW, y_mat);

    std::vector<uint16_t> y_dev_chw;
    matrix_to_chw(y_mat, y_dev_chw);

    // ---- Compare ----------------------------------------------------
    const float kAbsTol = 0.05f;
    const float kRelTol = 0.01f;
    uint32_t bad = 0;
    uint32_t first_bad = static_cast<uint32_t>(y_dev_chw.size());
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (uint32_t i = 0; i < y_dev_chw.size(); ++i) {
        float got = bf16_to_f32(y_dev_chw[i]);
        float exp = bf16_to_f32(y_ref_chw[i]);
        float d   = std::fabs(got - exp);
        float ref = std::fabs(exp);
        float tol = std::max(kAbsTol, kRelTol * ref);
        if (d > worst_abs) worst_abs = d;
        if (ref > 0.f && (d / ref) > worst_rel) worst_rel = d / ref;
        if (d > tol) {
            if (first_bad == y_dev_chw.size()) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_conv_1x1(C_in=%u C_out=%u H=%u W=%u): %u/%zu mismatches; "
            "first at idx %u: got=%.5f expected=%.5f, worst abs=%.5f, "
            "worst rel=%.4f%%\n",
            kCin, kCout, kH, kW, bad, y_dev_chw.size(), first_bad,
            bf16_to_f32(y_dev_chw[first_bad]),
            bf16_to_f32(y_ref_chw[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_conv_1x1: FAIL");
        return 1;
    }

    std::printf("test_conv_1x1: PASS  (C_in=%u C_out=%u H=%u W=%u, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kCin, kCout, kH, kW, worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_conv_1x1: FAIL — %s\n", e.what());
    return 1;
}
