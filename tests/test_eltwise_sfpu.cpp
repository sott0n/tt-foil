// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end test: run tt-metal's `programming_examples/eltwise_sfpu`
// kernels on the tt-foil runtime, **unchanged**.
//
// This is the first kernel-as-is integration: examples/eltwise_sfpu/
// kernels/{reader,writer,compute}.cpp are byte-identical copies of
// the tt-metal sources. The reader/writer use the modern
// `TensorAccessor` / `experimental::Noc` API, which resolves
// DRAM-interleaved page reads via the `dram_bank_to_noc_xy[]`
// table firmware copies out of `BANK_TO_NOC_SCRATCH` at boot. The
// table is populated by src/bank_tables_init.cpp; until then it was
// zero-filled and any TensorAccessor read landed on NOC (0, 0).
//
// Pipeline:
//   BRISC  (reader)  : DRAM page i → CB c_0   (TensorAccessor)
//   TRISC0 (UNPACK)  ┐
//   TRISC1 (MATH)    │ copy CB c_0 → DST regs, exp_tile(), pack → CB c_16
//   TRISC2 (PACK)    ┘
//   NCRISC (writer)  : CB c_16 → DRAM page i  (TensorAccessor)
//
// Single tile (page 0 → bank 0 → DRAM channel 0), so the existing
// tt-foil channel-0 allocator + write_buffer(BufferLocation::DRAM)
// are sufficient. Multi-tile interleaved cases would require fanning
// host writes across all 8 channels — a follow-up.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/eltwise_sfpu/prebuilt \
//   TT_FOIL_DEVICE=0 ./test_eltwise_sfpu

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

namespace tf = tt::foil;
namespace tut = tt::foil::test;

namespace {

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    const int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    constexpr uint32_t kNumTiles = 1;
    constexpr uint32_t kBytes = kNumTiles * tut::kTileBytes;

    auto dev = tf::open_device(pcie_index, "", {{0, 0}});
    tf::CoreCoord core{0, 0};

    auto dram_in  = tf::allocate_buffer(*dev, tf::BufferLocation::DRAM, kBytes);
    auto dram_out = tf::allocate_buffer(*dev, tf::BufferLocation::DRAM, kBytes);
    auto cb_in    = tf::allocate_buffer(*dev, tf::BufferLocation::L1, tut::kTileBytes, core);
    auto cb_out   = tf::allocate_buffer(*dev, tf::BufferLocation::L1, tut::kTileBytes, core);

    // Build an input tile in row-major float space, then tile it. Keep
    // values in [-2, 2] so exp(x) ∈ [~0.135, ~7.39] fits comfortably in
    // bf16 dynamic range.
    std::vector<uint16_t> rm(tut::kTileH * tut::kTileW);
    for (uint32_t r = 0; r < tut::kTileH; ++r) {
        for (uint32_t c = 0; c < tut::kTileW; ++c) {
            float v = -2.0f + 4.0f * static_cast<float>(r * tut::kTileW + c) /
                              static_cast<float>(tut::kTileH * tut::kTileW - 1);
            rm[r * tut::kTileW + c] = tut::f32_to_bf16(v);
        }
    }
    std::vector<uint16_t> tiled;  // must start empty: row_major_to_tile APPENDS.
    tut::row_major_to_tile(rm.data(), tiled);
    if (tiled.size() != tut::kTileWords) {
        throw std::runtime_error("row_major_to_tile size mismatch");
    }

    tf::write_buffer(*dev, *dram_in, tiled.data(), kBytes);

    std::vector<uint16_t> zero(tut::kTileWords, 0);
    tf::write_buffer(*dev, *dram_out, zero.data(), kBytes);

    using R = tf::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tf::load_kernel(*dev, bins, core);

    std::array<tf::CbConfig, 2> cbs = {{
        {0,  cb_in->device_addr,  tut::kTileBytes, 1, tut::kTileBytes},
        {16, cb_out->device_addr, tut::kTileBytes, 1, tut::kTileBytes},
    }};
    tf::register_cbs(*dev, *kernel, cbs);

    // Reader: (src_addr, n_tiles); writer: (dst_addr, n_tiles).
    // The bank_base_address passed to TensorAccessor is just the
    // DRAM buffer's device_addr — bank_to_dram_offset[] is 0 (see
    // src/bank_tables_init.cpp), so the kernel's resolution
    //   final = dram_bank_to_noc_xy[noc][page%8] << 36
    //         | (bank_base + (page/8) * 2048)
    // lands on the same byte that write_buffer wrote.
    const std::array<uint32_t, 2> ra_brisc  = {
        static_cast<uint32_t>(dram_in->device_addr),  kNumTiles};
    const std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(dram_out->device_addr), kNumTiles};
    const std::array<uint32_t, 1> ra_trisc  = {kNumTiles};
    tf::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);

    tf::execute(*dev, *kernel);

    std::vector<uint16_t> got_tiled(tut::kTileWords);
    tf::read_buffer(*dev, *dram_out, got_tiled.data(), kBytes);

    std::vector<uint16_t> got_rm(tut::kTileH * tut::kTileW);
    tut::tile_to_row_major(got_tiled.data(), got_rm.data());

    // exp(x) error in bf16: ~0.5% relative due to mantissa, plus SFPU
    // approximation. The reference eltwise_sfpu in tt-metal targets
    // ~1% — match.
    constexpr float kAbsTol = 0.05f;
    constexpr float kRelTol = 0.02f;
    uint32_t bad = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (uint32_t i = 0; i < tut::kTileH * tut::kTileW; ++i) {
        const float in_f  = tut::bf16_to_f32(rm[i]);
        const float got_f = tut::bf16_to_f32(got_rm[i]);
        const float ref   = std::exp(in_f);
        const float ae    = std::abs(got_f - ref);
        const float re    = ae / std::max(0.001f, std::abs(ref));
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
        if (ae > kAbsTol && re > kRelTol) {
            if (bad < 4) {
                std::fprintf(stderr,
                    "test_eltwise_sfpu: idx %u: in=%.4f got=%.4f ref=%.4f abs_err=%.4f rel_err=%.4f\n",
                    i, in_f, got_f, ref, ae, re);
            }
            ++bad;
        }
    }

    std::printf("test_eltwise_sfpu: %u elems, max_abs_err=%.4f max_rel_err=%.4f\n",
        tut::kTileH * tut::kTileW, max_abs, max_rel);
    if (bad > 0) {
        std::fprintf(stderr, "test_eltwise_sfpu: FAIL — %u/%u mismatches\n",
            bad, tut::kTileH * tut::kTileW);
        return 1;
    }
    std::puts("test_eltwise_sfpu: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_eltwise_sfpu: FAIL — %s\n", e.what());
    return 1;
}
