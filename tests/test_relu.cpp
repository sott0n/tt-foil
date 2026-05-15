// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5.5: bf16 ReLU through the 5-RISC SFPU pipeline.
//
// One 32×32 tile in, one tile out. Host writes a mix of positive and
// negative bf16 values; device should return max(x, 0) for each.

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

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}
}

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Host input: even rows positive, odd rows negative; column adds a
    // small ramp so we get a wide variety of magnitudes (~ ±15).
    std::vector<uint16_t> x_rm(kTileH * kTileW);
    for (uint32_t r = 0; r < kTileH; ++r) {
        for (uint32_t c = 0; c < kTileW; ++c) {
            float sign = (r & 1u) ? -1.0f : 1.0f;
            x_rm[r * kTileW + c] = f32_to_bf16(sign * (static_cast<float>(c) + 0.5f));
        }
    }
    std::vector<uint16_t> x_tile;
    tt::foil::test::row_major_to_tile(x_rm.data(), x_tile);

    // Reference: max(x, 0) element-wise, bf16.
    std::vector<uint16_t> y_ref_rm(kTileH * kTileW);
    for (uint32_t i = 0; i < kTileH * kTileW; ++i) {
        float v = bf16_to_f32(x_rm[i]);
        y_ref_rm[i] = f32_to_bf16(v < 0.0f ? 0.0f : v);
    }

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_in     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_in  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_in, x_tile.data(), kTileBytes);
    std::vector<uint16_t> zero(kTileWords, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), kTileBytes);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 2> cbs = {{
        {0,  buf_cb_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 1> ra_brisc  = {static_cast<uint32_t>(buf_in->device_addr)};
    std::array<uint32_t, 1> ra_ncrisc = {static_cast<uint32_t>(buf_out->device_addr)};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> y_tile(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tile.data(), kTileBytes);
    std::vector<uint16_t> y_dev_rm(kTileH * kTileW);
    tt::foil::test::tile_to_row_major(y_tile.data(), y_dev_rm.data());

    // ReLU is exact for bf16 inputs (no rounding in max(x, 0)). Tolerate
    // 0 abs diff — anything else means the SFPU op misfired or DST
    // copy/pack corrupted a value.
    uint32_t bad = 0, first_bad = kTileH * kTileW;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kTileH * kTileW; ++i) {
        float got = bf16_to_f32(y_dev_rm[i]);
        float exp = bf16_to_f32(y_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d != 0.0f) {
            if (first_bad == kTileH * kTileW) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_relu: %u/1024 mismatches; first at idx %u (row=%u col=%u): "
            "input=%.4f got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, first_bad, first_bad / kTileW, first_bad % kTileW,
            bf16_to_f32(x_rm[first_bad]),
            bf16_to_f32(y_dev_rm[first_bad]),
            bf16_to_f32(y_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_relu: FAIL");
        return 1;
    }

    std::printf("test_relu: PASS  (worst abs diff=%.5f)\n", worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_relu: FAIL — %s\n", e.what());
    return 1;
}
