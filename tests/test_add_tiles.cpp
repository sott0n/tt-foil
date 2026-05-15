// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5.6: bf16 eltwise add through the 5-RISC pipeline.
//
// Two 32×32 bf16 tiles in (CB c_0, CB c_1), one tile out (CB c_16),
// add_tiles dispatched on the matrix engine. Host fills A and B with
// distinct patterns and asserts C == bf16(fp32(A)+fp32(B)) elementwise.

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

    // A varies with row, B varies with col, so the sum has a clear 2-D
    // pattern that's easy to inspect when something goes wrong.
    std::vector<uint16_t> a_rm(kTileH * kTileW), b_rm(kTileH * kTileW);
    for (uint32_t r = 0; r < kTileH; ++r) {
        for (uint32_t c = 0; c < kTileW; ++c) {
            a_rm[r * kTileW + c] = f32_to_bf16(0.1f * static_cast<float>(r + 1));
            b_rm[r * kTileW + c] = f32_to_bf16(0.01f * static_cast<float>(c + 1));
        }
    }

    std::vector<uint16_t> a_tile, b_tile;
    tt::foil::test::row_major_to_tile(a_rm.data(), a_tile);
    tt::foil::test::row_major_to_tile(b_rm.data(), b_tile);

    // Reference: fp32 sum then bf16 round.
    std::vector<uint16_t> c_ref_rm(kTileH * kTileW);
    for (uint32_t i = 0; i < kTileH * kTileW; ++i) {
        c_ref_rm[i] = f32_to_bf16(bf16_to_f32(a_rm[i]) + bf16_to_f32(b_rm[i]));
    }

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_b       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, a_tile.data(), kTileBytes);
    tt::foil::write_buffer(*dev, *buf_b, b_tile.data(), kTileBytes);
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

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a->device_addr,   kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b->device_addr,   kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 2> ra_brisc  = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
    };
    std::array<uint32_t, 1> ra_ncrisc = {static_cast<uint32_t>(buf_out->device_addr)};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> c_tile(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tile.data(), kTileBytes);
    std::vector<uint16_t> c_dev_rm(kTileH * kTileW);
    tt::foil::test::tile_to_row_major(c_tile.data(), c_dev_rm.data());

    // DST regs run fp16-ish with DST_ACCUM_MODE=false, so the device
    // path effectively does add-then-round-to-fp16, then-round-to-bf16
    // on pack.  The host reference rounds once (fp32→bf16).  Worst case
    // is 1 bf16 ULP at the result magnitude (~0.016 here); allow 0.05
    // to be safe, same as matmul_1tile.
    const float kAbsTol = 0.05f;
    uint32_t bad = 0, first_bad = kTileH * kTileW;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kTileH * kTileW; ++i) {
        float got = bf16_to_f32(c_dev_rm[i]);
        float exp = bf16_to_f32(c_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kTileH * kTileW) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_add_tiles: %u/1024 mismatches; first at idx %u (row=%u col=%u): "
            "a=%.5f b=%.5f got=%.5f expected=%.5f, worst abs diff=%.5f\n",
            bad, first_bad, first_bad / kTileW, first_bad % kTileW,
            bf16_to_f32(a_rm[first_bad]), bf16_to_f32(b_rm[first_bad]),
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_add_tiles: FAIL");
        return 1;
    }

    std::printf("test_add_tiles: PASS  (worst abs diff=%.5f, tol=%.5f)\n",
                worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_add_tiles: FAIL — %s\n", e.what());
    return 1;
}
