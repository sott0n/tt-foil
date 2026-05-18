// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for global_avg_pool — reduce-sum across Nt input tiles with
// a 1/HW scaler so the device returns the per-channel mean. Input is
// shaped (C=32, HW=Nt×32), output is one (C=32, padded to 32) tile
// whose column 0 holds the means.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"
#include "tile_utils.hpp"

#ifndef GAP_NT_TEST
#define GAP_NT_TEST 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kC  = 32;
constexpr uint32_t kNt = GAP_NT_TEST;
constexpr uint32_t kHW = kNt * kTileW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

void tile_row_major(const std::vector<uint16_t>& m,
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
                        m[(rt * kTileH + r) * col_dim + ct * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Input (C, HW) row-major ----------------------------------
    std::mt19937 rng(0xa9a9);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> x_rm(kC * kHW);
    for (auto& v : x_rm) v = f32_to_bf16(u(rng));

    // Reference: per-channel mean.
    std::vector<uint16_t> ref(kC);
    for (uint32_t c = 0; c < kC; ++c) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < kHW; ++i) acc += bf16_to_f32(x_rm[c * kHW + i]);
        ref[c] = f32_to_bf16(acc / static_cast<float>(kHW));
    }

    // Scaler tile filled with 1/HW (bf16).
    const float scale = 1.0f / static_cast<float>(kHW);
    std::vector<uint16_t> scaler_rm(kTileH * kTileW, f32_to_bf16(scale));

    std::vector<uint16_t> x_tiles, scaler_tiles;
    tile_row_major(x_rm, /*rows_t=*/1, /*cols_t=*/kNt, kHW, x_tiles);
    tile_row_major(scaler_rm, 1, 1, kTileW, scaler_tiles);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    const uint32_t in_bytes     = kNt * kTileBytes;
    const uint32_t scaler_bytes = kTileBytes;
    const uint32_t out_bytes    = kTileBytes;

    auto buf_in     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, in_bytes,     core);
    auto buf_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, scaler_bytes, core);
    auto buf_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, out_bytes,    core);

    auto buf_cb_in     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_scaler = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_in,     x_tiles.data(),      in_bytes);
    tt::foil::write_buffer(*dev, *buf_scaler, scaler_tiles.data(), scaler_bytes);
    std::vector<uint8_t> zero(out_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), out_bytes);

    uint64_t in_noc     = tt::foil::make_noc_dram_addr(*dev, buf_in    ->device_addr);
    uint64_t scaler_noc = tt::foil::make_noc_dram_addr(*dev, buf_scaler->device_addr);
    uint64_t out_noc    = tt::foil::make_noc_dram_addr(*dev, buf_out   ->device_addr);

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
        {0,  buf_cb_in    ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_scaler->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out   ->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    std::array<uint32_t, 5> ra_brisc = {
        lo(in_noc),     hi(in_noc),
        lo(scaler_noc), hi(scaler_noc),
        kNt,
    };
    std::array<uint32_t, 2> ra_ncrisc = { lo(out_noc), hi(out_noc) };
    std::array<uint32_t, 1> ra_compute = { kNt };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_compute);

    tt::foil::execute(*dev, *kernel);

    // Read back the single output tile and pull column 0.
    std::vector<uint16_t> y_tiles(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), out_bytes);
    std::vector<uint16_t> y_block(kTileH * kTileW);
    tt::foil::test::tile_to_row_major(y_tiles.data(), y_block.data());

    std::vector<uint16_t> y_dev(kC);
    for (uint32_t c = 0; c < kC; ++c) y_dev[c] = y_block[c * kTileW + 0];

    // ---- Compare ---------------------------------------------------
    // Reduce-sum accumulates ~64 bf16 values per channel; the bf16
    // intermediate rounding gives a typical worst abs of a few thousandths.
    const float kAbsTol = 0.05f;
    uint32_t bad = 0, first_bad = kC;
    float worst = 0.f;
    for (uint32_t c = 0; c < kC; ++c) {
        float d = std::fabs(bf16_to_f32(y_dev[c]) - bf16_to_f32(ref[c]));
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kC) first_bad = c;
            ++bad;
        }
    }

    if (bad) {
        std::fprintf(stderr,
            "test_global_avg_pool: %u/%u mismatches; first ch %u: "
            "got=%.5f exp=%.5f worst=%.5f\n",
            bad, kC, first_bad,
            bf16_to_f32(y_dev[first_bad]),
            bf16_to_f32(ref[first_bad]), worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_global_avg_pool: FAIL");
        return 1;
    }

    std::printf("test_global_avg_pool: PASS  (C=%u HW=%u, worst abs=%.5f)\n",
                kC, kHW, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_global_avg_pool: FAIL — %s\n", e.what());
    return 1;
}
