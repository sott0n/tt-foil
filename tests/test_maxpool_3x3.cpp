// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// 3×3 stride-2 pad-1 maxpool — the operator that immediately follows
// the 7×7 stem conv in ResNet (and shrinks 112×112 → 56×56 on the real
// ImageNet input). Algorithm mirrors maxpool_2x2 but with 9 window
// positions instead of 4.
//
// For each output position (oh, ow), the 9 input positions are
//     ih = 2*oh + di - 1     wi = 2*ow + dj - 1     di, dj ∈ {0, 1, 2}
// Positions that fall outside [0, H_in) × [0, W_in) are padding; the
// host fills the corresponding stream slot with a very negative
// sentinel so SFPU binary_max ignores them.
//
// Test shape: H_in = W_in = 16, C = 32, H_out = W_out = 8.
// The output laid out as (H_out*W_out, C) = (64, 32) is exactly
// Nt = 2 bf16 tiles per channel column.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
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

constexpr uint32_t kH    = 16;
constexpr uint32_t kW    = 16;
constexpr uint32_t kC    = 32;
constexpr uint32_t kPad  = 1;
constexpr uint32_t kStride = 2;
constexpr uint32_t kHout = (kH + 2 * kPad - 3) / kStride + 1;   // 8
constexpr uint32_t kWout = (kW + 2 * kPad - 3) / kStride + 1;   // 8
constexpr uint32_t kNt   = (kHout * kWout) / kTileH;            // 2 tiles

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

void rm_matrix_to_tiles(const std::vector<uint16_t>& rm, uint32_t rows,
                        std::vector<uint16_t>& out) {
    if (rows % kTileH != 0) throw std::runtime_error("rows%32!=0");
    const uint32_t nt = rows / kTileH;
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                block[r * kTileW + c] = rm[(t * kTileH + r) * kTileW + c];
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

}  // namespace

int main() try {
    static_assert(kC == kTileW, "this test assumes C fits in one channel tile");

    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Host: random (H, W, C) input + 9 window-position streams ----
    std::mt19937 rng(0xa1f00d);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> x_f32(kH * kW * kC);
    for (auto& v : x_f32) v = dist(rng);
    std::vector<uint16_t> x_bf16(x_f32.size());
    for (size_t i = 0; i < x_f32.size(); ++i) x_bf16[i] = f32_to_bf16(x_f32[i]);

    auto idx = [&](uint32_t h, uint32_t w, uint32_t c) {
        return (h * kW + w) * kC + c;
    };

    // Sentinel used in place of padded positions so binary_max ignores
    // them. bf16 represents ±3.4e38; -1e30 is well below any real input.
    const uint16_t kNegInf = f32_to_bf16(-1.0e30f);

    std::array<std::vector<uint16_t>, 9> streams_rm;
    for (auto& s : streams_rm) s.assign(kHout * kWout * kC, kNegInf);

    for (uint32_t s = 0; s < 9; ++s) {
        int di = static_cast<int>(s / 3);
        int dj = static_cast<int>(s % 3);
        for (uint32_t oh = 0; oh < kHout; ++oh) {
            int ih = static_cast<int>(oh) * static_cast<int>(kStride) + di -
                     static_cast<int>(kPad);
            if (ih < 0 || ih >= static_cast<int>(kH)) continue;
            for (uint32_t ow = 0; ow < kWout; ++ow) {
                int iw = static_cast<int>(ow) * static_cast<int>(kStride) + dj -
                         static_cast<int>(kPad);
                if (iw < 0 || iw >= static_cast<int>(kW)) continue;
                for (uint32_t c = 0; c < kC; ++c) {
                    streams_rm[s][(oh * kWout + ow) * kC + c] =
                        x_bf16[idx(static_cast<uint32_t>(ih),
                                   static_cast<uint32_t>(iw), c)];
                }
            }
        }
    }

    // Tile each stream.
    std::array<std::vector<uint16_t>, 9> streams_tiles;
    for (uint32_t s = 0; s < 9; ++s)
        rm_matrix_to_tiles(streams_rm[s], kHout * kWout, streams_tiles[s]);

    // Reference: max over 3×3 window with -inf for padded slots.
    std::vector<uint16_t> y_ref_rm(kHout * kWout * kC);
    for (uint32_t oh = 0; oh < kHout; ++oh)
        for (uint32_t ow = 0; ow < kWout; ++ow)
            for (uint32_t c = 0; c < kC; ++c) {
                float m = -std::numeric_limits<float>::infinity();
                for (int di = 0; di < 3; ++di)
                    for (int dj = 0; dj < 3; ++dj) {
                        int ih = static_cast<int>(oh) * static_cast<int>(kStride) +
                                 di - static_cast<int>(kPad);
                        int iw = static_cast<int>(ow) * static_cast<int>(kStride) +
                                 dj - static_cast<int>(kPad);
                        if (ih < 0 || ih >= static_cast<int>(kH) ||
                            iw < 0 || iw >= static_cast<int>(kW)) continue;
                        float v = bf16_to_f32(x_bf16[idx(static_cast<uint32_t>(ih),
                                                         static_cast<uint32_t>(iw),
                                                         c)]);
                        if (v > m) m = v;
                    }
                y_ref_rm[(oh * kWout + ow) * kC + c] = f32_to_bf16(m);
            }

    // ---- Device ---------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    const uint32_t stream_bytes = kNt * kTileBytes;
    std::array<std::shared_ptr<tt::foil::Buffer>, 9> buf_in;
    for (uint32_t s = 0; s < 9; ++s)
        buf_in[s] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);

    std::array<std::shared_ptr<tt::foil::Buffer>, 10> cb_bufs;
    for (uint32_t i = 0; i < 10; ++i)
        cb_bufs[i] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    for (uint32_t s = 0; s < 9; ++s)
        tt::foil::write_buffer(*dev, *buf_in[s], streams_tiles[s].data(), stream_bytes);
    std::vector<uint16_t> zero(stream_bytes / 2, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), stream_bytes);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 10> cbs = {{
        {0,  cb_bufs[0]->device_addr, kTileBytes, 1, kTileBytes},
        {1,  cb_bufs[1]->device_addr, kTileBytes, 1, kTileBytes},
        {2,  cb_bufs[2]->device_addr, kTileBytes, 1, kTileBytes},
        {3,  cb_bufs[3]->device_addr, kTileBytes, 1, kTileBytes},
        {4,  cb_bufs[4]->device_addr, kTileBytes, 1, kTileBytes},
        {5,  cb_bufs[5]->device_addr, kTileBytes, 1, kTileBytes},
        {6,  cb_bufs[6]->device_addr, kTileBytes, 1, kTileBytes},
        {7,  cb_bufs[7]->device_addr, kTileBytes, 1, kTileBytes},
        {8,  cb_bufs[8]->device_addr, kTileBytes, 1, kTileBytes},
        {16, cb_bufs[9]->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 10> ra_brisc = {
        static_cast<uint32_t>(buf_in[0]->device_addr),
        static_cast<uint32_t>(buf_in[1]->device_addr),
        static_cast<uint32_t>(buf_in[2]->device_addr),
        static_cast<uint32_t>(buf_in[3]->device_addr),
        static_cast<uint32_t>(buf_in[4]->device_addr),
        static_cast<uint32_t>(buf_in[5]->device_addr),
        static_cast<uint32_t>(buf_in[6]->device_addr),
        static_cast<uint32_t>(buf_in[7]->device_addr),
        static_cast<uint32_t>(buf_in[8]->device_addr),
        kNt,
    };
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(buf_out->device_addr), kNt
    };
    std::array<uint32_t, 1> ra_compute = {kNt};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_compute);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> y_tiles(stream_bytes / 2, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), stream_bytes);

    std::vector<uint16_t> y_dev_rm(kHout * kWout * kC);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        tt::foil::test::tile_to_row_major(y_tiles.data() + t * kTileWords, block.data());
        for (uint32_t r = 0; r < kTileH; ++r)
            for (uint32_t c = 0; c < kTileW; ++c)
                y_dev_rm[(t * kTileH + r) * kTileW + c] = block[r * kTileW + c];
    }

    // Maxpool over bf16 is exact (selection only, no arithmetic), so
    // bit-for-bit match against the reference.
    uint32_t bad = 0, first_bad = static_cast<uint32_t>(y_dev_rm.size());
    float worst = 0.0f;
    for (size_t i = 0; i < y_dev_rm.size(); ++i) {
        float got = bf16_to_f32(y_dev_rm[i]);
        float exp = bf16_to_f32(y_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d != 0.0f) {
            if (first_bad == y_dev_rm.size()) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_maxpool_3x3: %u/%zu mismatches; first at idx %u: "
            "got=%.5f expected=%.5f, worst=%.5f\n",
            bad, y_dev_rm.size(), first_bad,
            bf16_to_f32(y_dev_rm[first_bad]),
            bf16_to_f32(y_ref_rm[first_bad]), worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_maxpool_3x3: FAIL");
        return 1;
    }

    std::printf("test_maxpool_3x3: PASS  "
                "(C=%u %ux%u→%ux%u, K=3 pad=1 stride=2)\n",
                kC, kH, kW, kHout, kWout);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_maxpool_3x3: FAIL — %s\n", e.what());
    return 1;
}
