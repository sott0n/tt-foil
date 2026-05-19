// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Softmax (per-row, minimum scope)
//   y[r, c] = exp(x[r, c]) / sum_c'(exp(x[r, c']))
// No max subtraction, no causal mask.
//
// Kernel CB layout:
//   0  cb_inp     (x, depth=Wt)
//   1  cb_reduce  (BF16(1.0) tile for SUM reduce, depth=1 persistent)
//   2  cb_exp     (exp(x), depth=Wt)
//   3  cb_sum     (per-row sum, depth=1)
//   4  cb_recip   (1/sum, depth=1)
//  16  cb_out     (output, depth=1)

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

#ifndef SOFTMAX_NCHT
#define SOFTMAX_NCHT 2
#endif
#ifndef SOFTMAX_WT
#define SOFTMAX_WT 4
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNCHt  = SOFTMAX_NCHT;
constexpr uint32_t kWt    = SOFTMAX_WT;
constexpr uint32_t kW     = kWt * kTileW;         // softmax axis length
constexpr uint32_t kNr    = kNCHt * kTileH;        // total rows
constexpr uint32_t kElems = kNr * kW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Convert row-major x[kNr, kW] → tile stream
void tile_stream_2d(const std::vector<uint16_t>& src_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kNCHt) * kWt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < kNCHt; ++rt) {
        for (uint32_t ct = 0; ct < kWt; ++ct) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = src_rm[(rt * kTileH + r) * kW + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

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
                    dst_rm[(rt * kTileH + r) * kW + ct * kTileW + c] = block[r * kTileW + c];
            ++tile_idx;
        }
    }
}

// Constant tile (all kTileH*kTileW elements = val).
// row_major_to_tile APPENDS; pass empty vector.
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

    // Input x: small bounded range (no max-subtract, so keep |x| small to avoid
    // exp overflow / massive precision loss in BF16).
    std::vector<uint16_t> x_rm(kElems);
    for (uint32_t r = 0; r < kNr; ++r) {
        for (uint32_t c = 0; c < kW; ++c) {
            // Per-row distinct values in [-1, +1]
            float v = -1.0f + 2.0f * static_cast<float>(c) / static_cast<float>(kW - 1);
            v += 0.05f * static_cast<float>(r % 16);   // small row variation
            x_rm[r * kW + c] = f32_to_bf16(v);
        }
    }

    // Host reference softmax (per row)
    std::vector<uint16_t> ref_rm(kElems);
    for (uint32_t r = 0; r < kNr; ++r) {
        // sum of exp(x)
        double s = 0.0;
        for (uint32_t c = 0; c < kW; ++c) s += std::exp(bf16_to_f32(x_rm[r * kW + c]));
        for (uint32_t c = 0; c < kW; ++c) {
            float y = static_cast<float>(std::exp(bf16_to_f32(x_rm[r * kW + c])) / s);
            ref_rm[r * kW + c] = f32_to_bf16(y);
        }
    }

    std::vector<uint16_t> x_tiles;
    tile_stream_2d(x_rm, x_tiles);
    const uint32_t x_bytes = kNCHt * kWt * kTileBytes;

    std::vector<uint16_t> scaler_tile;
    make_const_tile(1.0f, scaler_tile);   // scaler=1.0 → exact SUM reduce

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_x      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes,     core);
    auto buf_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTileBytes, core);
    auto buf_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, x_bytes,     core);

    const uint32_t l1_wt = kWt * kTileBytes;
    auto l1_inp    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    auto l1_reduce = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_exp    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    auto l1_sum    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_recip  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto l1_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_x,      x_tiles.data(),     x_bytes);
    tt::foil::write_buffer(*dev, *buf_scaler, scaler_tile.data(), kTileBytes);
    {
        std::vector<uint8_t> zero(x_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), x_bytes);
    }

    uint64_t x_noc      = tt::foil::make_noc_dram_addr(*dev, buf_x->device_addr);
    uint64_t scaler_noc = tt::foil::make_noc_dram_addr(*dev, buf_scaler->device_addr);
    uint64_t dst_noc    = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/softmax.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/softmax.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/softmax.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // fifo_size must equal num_pages * page_size (see src/cb_config.hpp).
    std::array<tt::foil::CbConfig, 6> cbs = {{
        {0,  l1_inp->device_addr,    l1_wt,      kWt, kTileBytes},
        {1,  l1_reduce->device_addr, kTileBytes, 1,   kTileBytes},
        {2,  l1_exp->device_addr,    l1_wt,      kWt, kTileBytes},
        {3,  l1_sum->device_addr,    kTileBytes, 1,   kTileBytes},
        {4,  l1_recip->device_addr,  kTileBytes, 1,   kTileBytes},
        {16, l1_out->device_addr,    kTileBytes, 1,   kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 6> ra_brisc = {
        static_cast<uint32_t>(x_noc & 0xffffffffu),  static_cast<uint32_t>(x_noc >> 32),
        static_cast<uint32_t>(scaler_noc & 0xffffffffu), static_cast<uint32_t>(scaler_noc >> 32),
        kNCHt, kWt,
    };
    std::array<uint32_t, 2> ra_trisc = {kNCHt, kWt};
    std::array<uint32_t, 4> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu), static_cast<uint32_t>(dst_noc >> 32),
        kNCHt, kWt,
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

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
        uint32_t r_bad = first_bad / kW;
        uint32_t c_bad = first_bad % kW;
        std::fprintf(stderr,
            "test_softmax: %u/%u mismatches; first at row=%u col=%u: "
            "x=%.4f got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, kElems, r_bad, c_bad,
            bf16_to_f32(x_rm[first_bad]),
            bf16_to_f32(out_rm[first_bad]),
            bf16_to_f32(ref_rm[first_bad]), worst);
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_softmax: FAIL\n");
        return 1;
    }

    // Per-row sum sanity check
    for (uint32_t r = 0; r < kNr && r < 4; ++r) {
        double s = 0.0;
        for (uint32_t c = 0; c < kW; ++c) s += bf16_to_f32(out_rm[r * kW + c]);
        std::printf("test_softmax: row=%u sum=%.4f (target=1.0)\n", r, s);
    }

    std::printf("test_softmax: PASS  (NCHt=%u, Wt=%u, W=%u, Nr=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kNCHt, kWt, kW, kNr, worst, kAbsTol);

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_softmax: FAIL — %s\n", e.what());
    return 1;
}
