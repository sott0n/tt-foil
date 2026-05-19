// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: SiLU eltwise unary via the op_lib eltwise_unary kernel.
//
// Reads NT bf16 tiles from DRAM, applies SiLU(x) = x*sigmoid(x) on-device,
// writes results back to DRAM, and compares against a host f32 reference.
//
// Runtime args layout:
//   BRISC:  arg[0..1] = src NOC addr (lo, hi), arg[2] = num_tiles
//   TRISC*: arg[0]    = num_tiles
//   NCRISC: arg[0..1] = dst NOC addr (lo, hi), arg[2] = num_tiles
//
// Usage:
//   TT_FOIL_KERNEL_DIR=ops/eltwise_unary/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_silu

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

#ifndef SILU_NT
#define SILU_NT 8
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNt    = SILU_NT;
constexpr uint32_t kElems = kNt * kTileH * kTileW;
constexpr uint32_t kBytes = kNt * kTileBytes;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

void tile_stream(const std::vector<uint16_t>& src_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kNt) * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                block[r * kTileW + c] = src_rm[t * (kTileH * kTileW) + r * kTileW + c];
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

void untile_stream(const std::vector<uint16_t>& tiles, std::vector<uint16_t>& dst_rm) {
    dst_rm.assign(kElems, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        const uint16_t* tile = tiles.data() + t * kTileWords;
        tt::foil::test::tile_to_row_major(tile, block.data());
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                dst_rm[t * (kTileH * kTileW) + r * kTileW + c] = block[r * kTileW + c];
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Input: values in [-4, 4] spread across tiles
    std::vector<uint16_t> x_rm(kElems);
    std::vector<uint16_t> y_ref_rm(kElems);
    for (uint32_t t = 0; t < kNt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                uint32_t idx = t * (kTileH * kTileW) + r * kTileW + c;
                float v = -4.0f + 8.0f * static_cast<float>(idx) / static_cast<float>(kElems - 1);
                x_rm[idx]     = f32_to_bf16(v);
                y_ref_rm[idx] = f32_to_bf16(silu_ref(bf16_to_f32(x_rm[idx])));
            }
        }
    }

    std::vector<uint16_t> x_stream;
    tile_stream(x_rm, x_stream);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_in    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes,     core);
    auto buf_out   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes,     core);
    auto buf_cb_in = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);
    auto buf_cb_out= tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_in, x_stream.data(), kBytes);
    std::vector<uint8_t> zero(kBytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), kBytes);

    uint64_t src_noc = tt::foil::make_noc_dram_addr(*dev, buf_in->device_addr);
    uint64_t dst_noc = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/silu.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/silu.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/silu.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 2> cbs = {{
        {0,  buf_cb_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 3> ra_brisc = {
        static_cast<uint32_t>(src_noc & 0xffffffffu),
        static_cast<uint32_t>(src_noc >> 32),
        kNt,
    };
    std::array<uint32_t, 1> ra_trisc = {kNt};
    std::array<uint32_t, 3> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu),
        static_cast<uint32_t>(dst_noc >> 32),
        kNt,
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> y_tiles(static_cast<size_t>(kNt) * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), kBytes);
    std::vector<uint16_t> y_dev_rm;
    untile_stream(y_tiles, y_dev_rm);

    const float kAbsTol = 0.02f;
    uint32_t bad = 0, first_bad = kElems;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float got = bf16_to_f32(y_dev_rm[i]);
        float exp = bf16_to_f32(y_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kElems) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        uint32_t r = first_bad / kTileW, c = first_bad % kTileW;
        std::fprintf(stderr,
            "test_silu: %u/%u mismatches; first at idx %u (r=%u c=%u): "
            "x=%.4f got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, kElems, first_bad, r, c,
            bf16_to_f32(x_rm[first_bad]),
            bf16_to_f32(y_dev_rm[first_bad]),
            bf16_to_f32(y_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_silu: FAIL");
        return 1;
    }

    std::printf("test_silu: PASS  (NT=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kNt, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_silu: FAIL — %s\n", e.what());
    return 1;
}
