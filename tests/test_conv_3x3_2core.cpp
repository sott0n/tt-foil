// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v7-3: 2-core sharded 3×3 convolution (pad=1, stride=1, N=1).
//
// The output's H axis is split between two cores. Each core handles
// H_local = H_global / 2 output rows. For 3×3 stride=1 conv, output
// row h depends on input rows {h-1, h, h+1}. So core 1's first
// output row needs core 0's last input row (halo). Conversely,
// core 0's last output row needs core 1's first input row.
//
// **Halo handling**: done entirely on the host. Each core's im2col
// matrix is built directly against the global input X with the
// per-core output row range. Where im2col reaches into the neighbor's
// rows it naturally picks them up because the host owns the full X.
// The device-side kernels are bit-identical to conv_3x3 / matmul_dram.
//
// Per-core tile shape:
//   MM_MT     = ceil(C_out         / 32)   M (output channels)
//   MM_KT     = ceil(C_in * 9      / 32)   K (im2col rows)
//   MM_NT     = ceil(H_local * W   / 32)   N (per-core output pixels)
// Global Nt   = 2 * MM_NT, H_global = 2 * H_local.

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
#define MM_MT 1
#endif
#ifndef MM_KT
#define MM_KT 9
#endif
#ifndef MM_NT
#define MM_NT 2     // per-core
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kMt        = MM_MT;
constexpr uint32_t kKt        = MM_KT;
constexpr uint32_t kNtPerCore = MM_NT;
constexpr uint32_t kCores     = 2;

constexpr uint32_t kCout    = kMt * kTileH;
constexpr uint32_t kKdim    = kKt * kTileW;            // C_in * 9
constexpr uint32_t kHWlocal = kNtPerCore * kTileW;     // per-core H_local * W

constexpr uint32_t kKH = 3;
constexpr uint32_t kKW = 3;
constexpr uint32_t kPad = 1;
constexpr uint32_t kCin = kKdim / (kKH * kKW);
static_assert(kCin * kKH * kKW == kKdim,
              "MM_KT*32 must equal C_in*9");

// Pick W and derive H_local from kHWlocal. W=8 keeps the spatial
// extent visible. With default kNtPerCore=2 → kHWlocal=64 → H_local=8.
constexpr uint32_t kW       = 8;
constexpr uint32_t kHlocal  = kHWlocal / kW;
constexpr uint32_t kH       = kHlocal * kCores;        // global H
static_assert(kHlocal * kW == kHWlocal, "W must divide kHWlocal");

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Reference 3×3 conv on the GLOBAL tensors.
void conv3x3_reference(const std::vector<uint16_t>& x_chw,
                       const std::vector<uint16_t>& w_cchw,
                       std::vector<uint16_t>& y_chw) {
    y_chw.assign(kCout * kH * kW, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t h = 0; h < kH; ++h)
            for (uint32_t w_ = 0; w_ < kW; ++w_) {
                float acc = 0.0f;
                for (uint32_t ci = 0; ci < kCin; ++ci)
                    for (uint32_t ki = 0; ki < kKH; ++ki)
                        for (uint32_t kj = 0; kj < kKW; ++kj) {
                            int ih = static_cast<int>(h + ki) - static_cast<int>(kPad);
                            int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                            if (ih < 0 || ih >= static_cast<int>(kH) ||
                                iw < 0 || iw >= static_cast<int>(kW)) continue;
                            float xv = bf16_to_f32(x_chw[(ci * kH + ih) * kW + iw]);
                            float wv = bf16_to_f32(
                                w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj]);
                            acc += xv * wv;
                        }
                y_chw[(co * kH + h) * kW + w_] = f32_to_bf16(acc);
            }
}

// Per-core im2col: builds the (C_in*9, H_local*W) matrix for output
// rows h_start .. h_start+H_local on the GLOBAL X (no slicing). When
// the convolution reaches into the neighbor's rows it picks up the
// halo automatically.
void im2col_local(const std::vector<uint16_t>& x_chw, uint32_t h_start,
                  std::vector<uint16_t>& a) {
    a.assign(kKdim * kHWlocal, 0);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t ki = 0; ki < kKH; ++ki)
            for (uint32_t kj = 0; kj < kKW; ++kj) {
                uint32_t row = ci * (kKH * kKW) + ki * kKW + kj;
                for (uint32_t hl = 0; hl < kHlocal; ++hl) {
                    int ih = static_cast<int>(h_start + hl + ki) -
                             static_cast<int>(kPad);
                    if (ih < 0 || ih >= static_cast<int>(kH)) continue;
                    for (uint32_t w_ = 0; w_ < kW; ++w_) {
                        int iw = static_cast<int>(w_ + kj) - static_cast<int>(kPad);
                        if (iw < 0 || iw >= static_cast<int>(kW)) continue;
                        uint32_t col = hl * kW + w_;
                        a[row * kHWlocal + col] = x_chw[(ci * kH + ih) * kW + iw];
                    }
                }
            }
}

void weight_reshape(const std::vector<uint16_t>& w_cchw,
                    std::vector<uint16_t>& w_mat) {
    w_mat.assign(kCout * kKdim, 0);
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj) {
                    uint32_t col = ci * (kKH * kKW) + ki * kKW + kj;
                    w_mat[co * kKdim + col] =
                        w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj];
                }
}

void tile_matrix(const std::vector<uint16_t>& m_rm,
                 uint32_t rows_t, uint32_t cols_t, uint32_t col_dim,
                 std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(rows_t) * cols_t * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] =
                        m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}

void untile_matrix(const std::vector<uint16_t>& tiles,
                   uint32_t rows_t, uint32_t cols_t, uint32_t col_dim,
                   std::vector<uint16_t>& m_rm) {
    m_rm.assign(rows_t * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = tiles.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    m_rm[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t w_bytes   = kMt        * kKt        * kTileBytes;
    const uint32_t a_bytes   = kKt        * kNtPerCore * kTileBytes;
    const uint32_t y_bytes   = kMt        * kNtPerCore * kTileBytes;

    // ---- Host tensors -----------------------------------------------
    std::vector<uint16_t> x_chw(kCin * kH * kW);
    std::vector<uint16_t> w_cchw(kCout * kCin * kKH * kKW);
    for (uint32_t ci = 0; ci < kCin; ++ci)
        for (uint32_t h = 0; h < kH; ++h)
            for (uint32_t w_ = 0; w_ < kW; ++w_)
                x_chw[(ci * kH + h) * kW + w_] = f32_to_bf16(0.005f *
                    static_cast<float>((ci + 1) + ((h * kW + w_) % 7)));
    for (uint32_t co = 0; co < kCout; ++co)
        for (uint32_t ci = 0; ci < kCin; ++ci)
            for (uint32_t ki = 0; ki < kKH; ++ki)
                for (uint32_t kj = 0; kj < kKW; ++kj)
                    w_cchw[((co * kCin + ci) * kKH + ki) * kKW + kj] =
                        f32_to_bf16(0.005f *
                            static_cast<float>((co + 1) + (ci % 5) + (ki * 3 + kj)));

    // ---- Reference --------------------------------------------------
    std::vector<uint16_t> y_ref_chw;
    conv3x3_reference(x_chw, w_cchw, y_ref_chw);

    // ---- Per-core im2col + shared weight reshape -------------------
    std::array<std::vector<uint16_t>, kCores> a_per_core;
    for (uint32_t cid = 0; cid < kCores; ++cid)
        im2col_local(x_chw, cid * kHlocal, a_per_core[cid]);

    std::vector<uint16_t> w_mat;
    weight_reshape(w_cchw, w_mat);

    std::vector<uint16_t> w_tiles;
    tile_matrix(w_mat, kMt, kKt, kKdim, w_tiles);

    std::array<std::vector<uint16_t>, kCores> a_tiles_per_core;
    for (uint32_t cid = 0; cid < kCores; ++cid)
        tile_matrix(a_per_core[cid], kKt, kNtPerCore, kHWlocal, a_tiles_per_core[cid]);

    // ---- Device + 2 cores ------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {0, 1}});
    std::array<tt::foil::CoreCoord, kCores> cores = {{ {0, 0}, {0, 1} }};

    // Weight buffer is shared; per-core im2col + output live in their
    // own DRAM allocations so the kernel sees a contiguous Kt × Nt
    // stream for the A side and a contiguous Mt × Nt for the output.
    auto buf_w = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, w_bytes, cores[0]);
    std::array<std::shared_ptr<tt::foil::Buffer>, kCores> buf_a_per_core, buf_out_per_core;
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        buf_a_per_core  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes, cores[cid]);
        buf_out_per_core[cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, y_bytes, cores[cid]);
    }

    tt::foil::write_buffer(*dev, *buf_w, w_tiles.data(), w_bytes);
    std::vector<uint8_t> zero(y_bytes, 0);
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        tt::foil::write_buffer(*dev, *buf_a_per_core[cid],
                               a_tiles_per_core[cid].data(), a_bytes);
        tt::foil::write_buffer(*dev, *buf_out_per_core[cid], zero.data(), y_bytes);
    }

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};

    std::array<std::shared_ptr<tt::foil::Buffer>, kCores> cb_a, cb_b, cb_out;
    std::array<std::shared_ptr<tt::foil::Kernel>, kCores> kernels;

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    for (uint32_t cid = 0; cid < kCores; ++cid) {
        cb_a  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        cb_b  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        cb_out[cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        kernels[cid] = tt::foil::load_kernel(*dev, bins, cores[cid]);

        std::array<tt::foil::CbConfig, 3> cbs = {{
            {0,  cb_a  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {1,  cb_b  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {16, cb_out[cid]->device_addr, kTileBytes, 1, kTileBytes},
        }};
        tt::foil::register_cbs(*dev, *kernels[cid], cbs);

        uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, buf_w           ->device_addr);
        uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, buf_a_per_core  [cid]->device_addr);
        uint64_t out_noc = tt::foil::make_noc_dram_addr(*dev, buf_out_per_core[cid]->device_addr);

        std::array<uint32_t, 7> ra_brisc = {
            lo(a_noc), hi(a_noc),
            lo(b_noc), hi(b_noc),
            kMt, kKt, kNtPerCore,
        };
        std::array<uint32_t, 3> ra_ncrisc = {
            lo(out_noc), hi(out_noc), kMt * kNtPerCore,
        };
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::NCRISC, ra_ncrisc);
    }

    tt::foil::execute(*dev, {kernels[0].get(), kernels[1].get()});

    // ---- Read back + stitch ----------------------------------------
    std::array<std::vector<uint16_t>, kCores> y_per_core_mat;
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        std::vector<uint16_t> y_tiles(kMt * kNtPerCore * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_out_per_core[cid], y_tiles.data(), y_bytes);
        untile_matrix(y_tiles, kMt, kNtPerCore, kHWlocal, y_per_core_mat[cid]);
    }

    // Stitch per-core Y (each is C_out × H_local*W row-major in
    // identity CHW interpretation) into global C_out × H*W.
    std::vector<uint16_t> y_dev_chw(kCout * kH * kW, 0);
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        for (uint32_t co = 0; co < kCout; ++co)
            for (uint32_t hl = 0; hl < kHlocal; ++hl)
                for (uint32_t w_ = 0; w_ < kW; ++w_) {
                    uint32_t gh = cid * kHlocal + hl;
                    y_dev_chw[(co * kH + gh) * kW + w_] =
                        y_per_core_mat[cid][co * kHWlocal + hl * kW + w_];
                }
    }

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
            "test_conv_3x3_2core(C_in=%u C_out=%u H=%u W=%u, 2 cores, "
            "H_local=%u): %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            kCin, kCout, kH, kW, kHlocal,
            bad, y_dev_chw.size(), first_bad,
            bf16_to_f32(y_dev_chw[first_bad]),
            bf16_to_f32(y_ref_chw[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_conv_3x3_2core: FAIL");
        return 1;
    }

    std::printf("test_conv_3x3_2core: PASS  (C_in=%u C_out=%u H=%u W=%u, "
                "2 cores H_local=%u, pad=1 stride=1, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kCin, kCout, kH, kW, kHlocal, worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_conv_3x3_2core: FAIL — %s\n", e.what());
    return 1;
}
