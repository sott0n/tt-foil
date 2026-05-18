// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for bias_relu_post — drives the kernel with N input tiles
// laid out as (C=32, HW=Nt×32), a per-channel bias of length 32, and a
// runtime relu_enable flag. Compares each pixel against a host
// reference. We run two flavours back-to-back on the same kernel
// program: relu_enable=0 ("bias only") and relu_enable=1 ("bias +
// ReLU"). Both must match.

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

#ifndef BR_NT_TEST
#define BR_NT_TEST 4
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kC   = 32;
constexpr uint32_t kNt  = BR_NT_TEST;
constexpr uint32_t kHW  = kNt * kTileW;       // spatial slots per channel
constexpr uint32_t kElems = kC * kHW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Take a (C, HW) row-major matrix and tile it as Mt=1, Nt tiles (each
// 32×32). Same shape layout used elsewhere for conv outputs.
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

void untile_row_major(const std::vector<uint16_t>& tiles,
                      uint32_t rows_t, uint32_t cols_t, uint32_t col_dim,
                      std::vector<uint16_t>& m) {
    m.assign(rows_t * kTileH * col_dim, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t rt = 0; rt < rows_t; ++rt)
        for (uint32_t ct = 0; ct < cols_t; ++ct) {
            const uint16_t* tile = tiles.data() + (rt * cols_t + ct) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    m[(rt * kTileH + r) * col_dim + ct * kTileW + c] =
                        block[r * kTileW + c];
        }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Inputs ----------------------------------------------------
    std::mt19937 rng(0xb1a5);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> x_rm(kElems);
    for (auto& v : x_rm) v = f32_to_bf16(u(rng));

    // Per-channel bias laid out into a single 32×32 tile, column 0:
    // bias_tile[r, 0] = bias for channel r; cols 1..31 ignored.
    std::vector<float> bias(kC);
    for (auto& v : bias) v = 0.5f * u(rng);
    std::vector<uint16_t> bias_tile_rm(kTileH * kTileW, 0);
    for (uint32_t r = 0; r < kC; ++r)
        bias_tile_rm[r * kTileW + 0] = f32_to_bf16(bias[r]);

    std::vector<uint16_t> x_tiles, bias_tiles;
    tile_row_major(x_rm, 1, kNt, kHW, x_tiles);
    tile_row_major(bias_tile_rm, 1, 1, kTileW, bias_tiles);

    // ---- Reference for both relu flavours ---------------------------
    auto ref = [&](bool relu) {
        std::vector<uint16_t> y(kElems);
        for (uint32_t c = 0; c < kC; ++c)
            for (uint32_t i = 0; i < kHW; ++i) {
                float v = bf16_to_f32(x_rm[c * kHW + i]) + bias[c];
                if (relu && v < 0.f) v = 0.f;
                y[c * kHW + i] = f32_to_bf16(v);
            }
        return y;
    };
    auto y_ref_noact  = ref(false);
    auto y_ref_relu   = ref(true);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    const uint32_t in_bytes   = kNt * kTileBytes;
    const uint32_t bias_bytes = kTileBytes;
    const uint32_t out_bytes  = kNt * kTileBytes;

    auto buf_in   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, in_bytes,   core);
    auto buf_bias = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, bias_bytes, core);
    auto buf_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, out_bytes,  core);

    auto buf_cb_in   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_bias = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_in,   x_tiles.data(),    in_bytes);
    tt::foil::write_buffer(*dev, *buf_bias, bias_tiles.data(), bias_bytes);

    uint64_t in_noc   = tt::foil::make_noc_dram_addr(*dev, buf_in  ->device_addr);
    uint64_t bias_noc = tt::foil::make_noc_dram_addr(*dev, buf_bias->device_addr);
    uint64_t out_noc  = tt::foil::make_noc_dram_addr(*dev, buf_out ->device_addr);

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
        {0,  buf_cb_in  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  buf_cb_bias->device_addr, kTileBytes, 1, kTileBytes},
        {16, buf_cb_out ->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    auto run_one = [&](uint32_t relu_enable, std::vector<uint16_t>& y_dev) {
        std::vector<uint8_t> zero(out_bytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), out_bytes);

        std::array<uint32_t, 5> ra_brisc = {
            lo(in_noc), hi(in_noc), lo(bias_noc), hi(bias_noc), kNt
        };
        std::array<uint32_t, 3> ra_ncrisc = { lo(out_noc), hi(out_noc), kNt };
        std::array<uint32_t, 2> ra_compute = { kNt, relu_enable };

        tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);
        tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_compute);
        tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_compute);
        tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_compute);
        tt::foil::execute(*dev, *kernel);

        std::vector<uint16_t> y_tiles(kNt * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), out_bytes);
        untile_row_major(y_tiles, 1, kNt, kHW, y_dev);
    };

    auto compare = [&](const std::vector<uint16_t>& dev,
                       const std::vector<uint16_t>& ref_v,
                       const char* label) -> bool {
        const float kAbsTol = 0.05f;
        uint32_t bad = 0, first_bad = kElems;
        float worst = 0.f;
        for (uint32_t i = 0; i < kElems; ++i) {
            float d = std::fabs(bf16_to_f32(dev[i]) - bf16_to_f32(ref_v[i]));
            if (d > worst) worst = d;
            if (d > kAbsTol) {
                if (first_bad == kElems) first_bad = i;
                ++bad;
            }
        }
        if (bad) {
            std::fprintf(stderr,
                "[%s] %u/%u mismatches; first idx %u: got=%.5f exp=%.5f worst=%.5f\n",
                label, bad, kElems, first_bad,
                bf16_to_f32(dev[first_bad]),
                bf16_to_f32(ref_v[first_bad]), worst);
            return false;
        }
        std::printf("  %-12s worst abs=%.5f (tol %.2f) PASS\n",
                    label, worst, kAbsTol);
        return true;
    };

    std::vector<uint16_t> y_noact, y_relu;
    run_one(0, y_noact);
    run_one(1, y_relu);

    bool ok = true;
    ok &= compare(y_noact, y_ref_noact, "bias only");
    ok &= compare(y_relu,  y_ref_relu,  "bias+ReLU");

    if (!ok) {
        tt::foil::close_device(std::move(dev));
        std::puts("test_bias_relu_post: FAIL");
        return 1;
    }
    std::printf("test_bias_relu_post: PASS  (C=%u, Nt=%u)\n", kC, kNt);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_bias_relu_post: FAIL — %s\n", e.what());
    return 1;
}
