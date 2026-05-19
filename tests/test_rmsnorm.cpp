// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: RMSNorm  y = x * (1/sqrt(mean(x²)+eps)) * gamma
//
// Kernel CB layout:
//   0  cb_inp        (x, depth=Wt)
//   1  cb_reduce     (scaler=1/H, depth=1, persistent)
//   2  cb_gamma      (gamma weights replicated to 32 rows, depth=Wt, persistent)
//   3  cb_eps        (eps, depth=1, persistent)
//   4  cb_x2        (x², depth=Wt, per row)
//   5  cb_var        (E[x²], depth=1)
//   6  cb_recip_sqrt (1/sqrt(var+eps), depth=1)
//   7  cb_x_normed   (x*scale, depth=Wt)
//  16  cb_out        (output, depth=1)
//
// Usage:
//   TT_FOIL_KERNEL_DIR=ops/rmsnorm/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_rmsnorm

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

// Test dimensions: NCHt token tile-rows × Wt hidden tiles
#ifndef RMS_NCHT
#define RMS_NCHT 2
#endif
#ifndef RMS_WT
#define RMS_WT 4
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNCHt  = RMS_NCHT;
constexpr uint32_t kWt    = RMS_WT;
constexpr uint32_t kH     = kWt * kTileW;        // hidden dim
constexpr uint32_t kNr    = kNCHt * kTileH;       // total tokens
constexpr uint32_t kElems = kNr * kH;             // total elements

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Convert row-major x[kNr, kH] to tile-format stream: kNCHt * kWt tiles
void tile_stream_2d(const std::vector<uint16_t>& src_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kNCHt) * kWt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < kNCHt; ++rt) {
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = src_rm[(rt * kTileH + r) * kH + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

// Convert tile-format stream back to row-major x[kNr, kH]
void untile_stream_2d(const std::vector<uint16_t>& tiles, std::vector<uint16_t>& dst_rm) {
    dst_rm.assign(kElems, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    uint32_t tile_idx = 0;
    for (uint32_t rt = 0; rt < kNCHt; ++rt) {
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            const uint16_t* tile = tiles.data() + tile_idx * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    dst_rm[(rt * kTileH + r) * kH + ct * kTileW + c] = block[r * kTileW + c];
            ++tile_idx;
        }
    }
}

// gamma_rm[kNr, kH]: all kNr rows identical, each row = gamma[kH]
// → for mul_tiles (elementwise), gamma is replicated to all 32 token rows per tile
void make_gamma_tiles(const std::vector<uint16_t>& gamma, std::vector<uint16_t>& out) {
    // Expand gamma to kNr rows (all same)
    std::vector<uint16_t> gamma_rm(kNr * kH);
    for (uint32_t r = 0; r < kNr; ++r)
        for (uint32_t c = 0; c < kH; ++c)
            gamma_rm[r * kH + c] = gamma[c];
    tile_stream_2d(gamma_rm, out);
}

// Constant tile: all kTileH * kTileW elements = val.
// NOTE: row_major_to_tile APPENDS kTileWords to `out` starting at out.size().
// Pass an empty vector so the resulting tile lives at offset 0.
void make_const_tile(float val, std::vector<uint16_t>& out) {
    uint16_t bf16_val = f32_to_bf16(val);
    out.clear();
    std::vector<uint16_t> block(kTileH * kTileW, bf16_val);
    tt::foil::test::row_major_to_tile(block.data(), out);
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    constexpr float kEps = 1e-5f;

    // Build input x: values in range [-2, 2]
    std::vector<uint16_t> x_rm(kElems);
    for (uint32_t i = 0; i < kElems; ++i) {
        float v = -2.0f + 4.0f * static_cast<float>(i % 256) / 255.0f;
        x_rm[i] = f32_to_bf16(v);
    }

    // Gamma = 1.0 for easy verification
    std::vector<uint16_t> gamma(kH, f32_to_bf16(1.0f));

    // Compute reference output on host
    std::vector<uint16_t> ref_rm(kElems);
    for (uint32_t t = 0; t < kNr; ++t) {
        float sum_sq = 0.0f;
        for (uint32_t c = 0; c < kH; ++c) {
            float x = bf16_to_f32(x_rm[t * kH + c]);
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(kH) + kEps);
        float inv_rms = 1.0f / rms;
        for (uint32_t c = 0; c < kH; ++c) {
            float x = bf16_to_f32(x_rm[t * kH + c]);
            float g = bf16_to_f32(gamma[c]);
            ref_rm[t * kH + c] = f32_to_bf16(x * inv_rms * g);
        }
    }

    // Tile x
    std::vector<uint16_t> x_tiles;
    tile_stream_2d(x_rm, x_tiles);
    const uint32_t x_bytes = kNCHt * kWt * kTileBytes;

    // Tile gamma (replicated to all token rows)
    std::vector<uint16_t> gamma_tiles;
    make_gamma_tiles(gamma, gamma_tiles);
    const uint32_t gamma_bytes = kWt * kTileBytes;

    // Constant tiles
    std::vector<uint16_t> scaler_tile, eps_tile;
    make_const_tile(1.0f / static_cast<float>(kH), scaler_tile);
    make_const_tile(kEps, eps_tile);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // DRAM buffers
    auto buf_x       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes,        core);
    auto buf_gamma   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, gamma_bytes,     core);
    auto buf_scaler  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes,      core);
    auto buf_eps     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes,      core);
    auto buf_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes,         core);

    // L1 CB buffers
    const uint32_t l1_wt = kWt * kTileBytes;  // size for depth=Wt CBs
    auto l1_inp        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,     core);
    auto l1_reduce     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_gamma      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,     core);
    auto l1_eps        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_x2         = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,     core);
    auto l1_var        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_recip_sqrt = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_x_normed   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,     core);
    auto l1_out        = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    // Write inputs
    tt::foil::write_buffer(*dev, *buf_x,      x_tiles.data(),       x_bytes);
    tt::foil::write_buffer(*dev, *buf_gamma,  gamma_tiles.data(),   gamma_bytes);
    tt::foil::write_buffer(*dev, *buf_scaler, scaler_tile.data(),   kTileBytes);
    tt::foil::write_buffer(*dev, *buf_eps,    eps_tile.data(),      kTileBytes);
    {
        std::vector<uint8_t> zero(x_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), x_bytes);
    }

    uint64_t x_noc      = tt::foil::make_noc_dram_addr(*dev, buf_x->device_addr);
    uint64_t gamma_noc  = tt::foil::make_noc_dram_addr(*dev, buf_gamma->device_addr);
    uint64_t scaler_noc = tt::foil::make_noc_dram_addr(*dev, buf_scaler->device_addr);
    uint64_t eps_noc    = tt::foil::make_noc_dram_addr(*dev, buf_eps->device_addr);
    uint64_t dst_noc    = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/rmsnorm.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/rmsnorm.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/rmsnorm.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // CB registration
    std::array<tt::foil::CbConfig, 9> cbs = {{
        {0,  l1_inp->device_addr,        kTileBytes, kWt, kTileBytes},  // cb_inp, depth=Wt
        {1,  l1_reduce->device_addr,     kTileBytes, 1,   kTileBytes},  // cb_reduce
        {2,  l1_gamma->device_addr,      kTileBytes, kWt, kTileBytes},  // cb_gamma, depth=Wt
        {3,  l1_eps->device_addr,        kTileBytes, 1,   kTileBytes},  // cb_eps
        {4,  l1_x2->device_addr,         kTileBytes, kWt, kTileBytes},  // cb_x2, depth=Wt
        {5,  l1_var->device_addr,        kTileBytes, 1,   kTileBytes},  // cb_var
        {6,  l1_recip_sqrt->device_addr, kTileBytes, 1,   kTileBytes},  // cb_recip_sqrt
        {7,  l1_x_normed->device_addr,   kTileBytes, kWt, kTileBytes},  // cb_x_normed, depth=Wt
        {16, l1_out->device_addr,        kTileBytes, 1,   kTileBytes},  // cb_out
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    // Runtime args
    std::array<uint32_t, 10> ra_brisc = {
        static_cast<uint32_t>(x_noc & 0xffffffffu),
        static_cast<uint32_t>(x_noc >> 32),
        static_cast<uint32_t>(gamma_noc & 0xffffffffu),
        static_cast<uint32_t>(gamma_noc >> 32),
        static_cast<uint32_t>(scaler_noc & 0xffffffffu),
        static_cast<uint32_t>(scaler_noc >> 32),
        static_cast<uint32_t>(eps_noc & 0xffffffffu),
        static_cast<uint32_t>(eps_noc >> 32),
        kNCHt, kWt,
    };
    std::array<uint32_t, 2> ra_trisc = {kNCHt, kWt};
    std::array<uint32_t, 4> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu),
        static_cast<uint32_t>(dst_noc >> 32),
        kNCHt, kWt,
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    // Read back and verify
    const uint32_t total_tiles = kNCHt * kWt;
    std::vector<uint16_t> out_tiles(static_cast<size_t>(total_tiles) * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, out_tiles.data(), x_bytes);

    std::vector<uint16_t> out_rm;
    untile_stream_2d(out_tiles, out_rm);

    const float kAbsTol = 0.02f;
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
        uint32_t t_bad = first_bad / kH;
        uint32_t c_bad = first_bad % kH;
        std::fprintf(stderr,
            "test_rmsnorm: %u/%u mismatches; first at token=%u col=%u: "
            "x=%.4f got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, kElems, t_bad, c_bad,
            bf16_to_f32(x_rm[first_bad]),
            bf16_to_f32(out_rm[first_bad]),
            bf16_to_f32(ref_rm[first_bad]), worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_rmsnorm: FAIL");
        return 1;
    }

    std::printf("test_rmsnorm: PASS  (NCHt=%u, Wt=%u, H=%u, Nr=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kNCHt, kWt, kH, kNr, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_rmsnorm: FAIL — %s\n", e.what());
    return 1;
}
