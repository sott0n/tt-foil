// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile bf16 eltwise add through the 5-RISC pipeline, with A, B,
// and C all backed by DRAM. The building block for ResNet's skip-
// connection add (Y = Conv2(...) + X), but useful in isolation as the
// first tile-loop eltwise example.
//
// RA_NT tiles per stream. Each 32×32 bf16 tile is 2 KB, so e.g. RA_NT=8
// covers 16 KB per side. Host fills A and B with distinct fp32 patterns
// rounded to bf16, runs the device add, and compares to a host-fp32
// reference rounded once.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/residual_add/prebuilt \
//   TT_FOIL_DEVICE=0 ./test_residual_add

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

#ifndef RA_NT
#define RA_NT 8
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kNt    = RA_NT;
constexpr uint32_t kElems = kNt * kTileH * kTileW;
constexpr uint32_t kBytes = kNt * kTileBytes;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Tile the per-tile row-major blocks into the device's tile layout, one
// 32×32 block at a time. Each tile is independent (no spatial
// stitching), so this is just `RA_NT` calls into row_major_to_tile.
void tile_stream(const std::vector<uint16_t>& src_rm,
                 std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kNt) * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                block[r * kTileW + c] =
                    src_rm[t * (kTileH * kTileW) + r * kTileW + c];
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

void untile_stream(const std::vector<uint16_t>& tiles,
                   std::vector<uint16_t>& dst_rm) {
    dst_rm.assign(kElems, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        const uint16_t* tile = tiles.data() + t * kTileWords;
        tt::foil::test::tile_to_row_major(tile, block.data());
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                dst_rm[t * (kTileH * kTileW) + r * kTileW + c] =
                    block[r * kTileW + c];
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Build A and B with distinct patterns: A varies by tile-index +
    // row, B varies by tile-index + col. Gives a 3-D pattern that's
    // easy to eyeball in a hex dump when something goes wrong.
    std::vector<uint16_t> a_rm(kElems), b_rm(kElems), c_ref_rm(kElems);
    for (uint32_t t = 0; t < kNt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                uint32_t idx = t * (kTileH * kTileW) + r * kTileW + c;
                a_rm[idx] = f32_to_bf16(0.01f * static_cast<float>(r + 1) +
                                        0.1f  * static_cast<float>(t));
                b_rm[idx] = f32_to_bf16(0.01f * static_cast<float>(c + 1) +
                                        0.01f * static_cast<float>(t));
                c_ref_rm[idx] =
                    f32_to_bf16(bf16_to_f32(a_rm[idx]) +
                                bf16_to_f32(b_rm[idx]));
            }
        }
    }

    std::vector<uint16_t> a_stream, b_stream;
    tile_stream(a_rm, a_stream);
    tile_stream(b_rm, b_stream);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes,     core);
    auto buf_b       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes,     core);
    auto buf_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes,     core);
    auto buf_cb_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);
    auto buf_cb_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), kBytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), kBytes);
    std::vector<uint8_t> zero(kBytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), kBytes);

    uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, buf_a->device_addr);
    uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, buf_b->device_addr);
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
        {0,  buf_cb_a->device_addr,   kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b->device_addr,   kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 4> ra_brisc = {
        static_cast<uint32_t>(a_noc & 0xffffffffu),
        static_cast<uint32_t>(a_noc >> 32),
        static_cast<uint32_t>(b_noc & 0xffffffffu),
        static_cast<uint32_t>(b_noc >> 32),
    };
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(out_noc & 0xffffffffu),
        static_cast<uint32_t>(out_noc >> 32),
    };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> c_tiles(static_cast<size_t>(kNt) * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tiles.data(), kBytes);
    std::vector<uint16_t> c_dev_rm;
    untile_stream(c_tiles, c_dev_rm);

    // Same fp16-DST-rounding tolerance story as test_add_tiles.
    const float kAbsTol = 0.05f;
    uint32_t bad = 0, first_bad = kElems;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kElems; ++i) {
        float got = bf16_to_f32(c_dev_rm[i]);
        float exp = bf16_to_f32(c_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kElems) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_residual_add: %u/%u mismatches; first at idx %u: "
            "a=%.5f b=%.5f got=%.5f expected=%.5f, worst abs diff=%.5f\n",
            bad, kElems, first_bad,
            bf16_to_f32(a_rm[first_bad]),
            bf16_to_f32(b_rm[first_bad]),
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_residual_add: FAIL");
        return 1;
    }

    std::printf("test_residual_add: PASS  (RA_NT=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kNt, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_residual_add: FAIL — %s\n", e.what());
    return 1;
}
