// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v8.1: 2×2 stride-2 maxpool over a (H, W, C) bf16 tensor.
//
// We pick H = W = 16 and C = 32 so the output is (8, 8, 32) — exactly
// MM_NT = 2 tiles when we lay out the output as a (Hout*Wout, C) matrix
// (64 rows × 32 cols). The host gathers each pool window position into a
// separate (Hout*Wout, C) tile stream and the compute kernel reduces the
// 4 streams pairwise with the SFPU binary_max op.

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

namespace {
using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kH    = 16;
constexpr uint32_t kW    = 16;
constexpr uint32_t kC    = 32;          // == kTileW; one tile wide in channels
constexpr uint32_t kHout = kH / 2;      // 8
constexpr uint32_t kWout = kW / 2;      // 8
constexpr uint32_t kNt   = (kHout * kWout) / kTileH;  // 64/32 = 2 output tiles

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Convert a (Nt*kTileH rows, kTileW cols) row-major bf16 matrix into Nt
// 32×32 face-laid-out tiles, appended sequentially to `out`.
void rm_matrix_to_tiles(const std::vector<uint16_t>& rm, uint32_t rows,
                        std::vector<uint16_t>& out) {
    if (rows % kTileH != 0) throw std::runtime_error("rows%32!=0");
    const uint32_t nt = rows / kTileH;
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                block[r * kTileW + c] = rm[(t * kTileH + r) * kTileW + c];
            }
        }
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}
}  // namespace

int main() try {
    static_assert(kC == kTileW, "this test assumes C fits in one channel tile");

    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Host: build random (H, W, C) input + 4 window-position streams ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> x_f32(kH * kW * kC);
    for (auto& v : x_f32) v = dist(rng);
    std::vector<uint16_t> x_bf16(x_f32.size());
    for (size_t i = 0; i < x_f32.size(); ++i) x_bf16[i] = f32_to_bf16(x_f32[i]);

    auto idx = [&](uint32_t h, uint32_t w, uint32_t c) {
        return (h * kW + w) * kC + c;
    };

    // 4 streams, each shaped (Hout*Wout, C) row-major.
    std::array<std::vector<uint16_t>, 4> streams_rm;
    for (auto& s : streams_rm) s.resize(kHout * kWout * kC);

    for (uint32_t s = 0; s < 4; ++s) {
        uint32_t di = s / 2;
        uint32_t dj = s % 2;
        for (uint32_t oh = 0; oh < kHout; ++oh) {
            for (uint32_t ow = 0; ow < kWout; ++ow) {
                uint32_t ih = 2 * oh + di;
                uint32_t iw = 2 * ow + dj;
                for (uint32_t c = 0; c < kC; ++c) {
                    streams_rm[s][(oh * kWout + ow) * kC + c] = x_bf16[idx(ih, iw, c)];
                }
            }
        }
    }

    // Tile-lay-out each stream.
    std::array<std::vector<uint16_t>, 4> streams_tiles;
    for (uint32_t s = 0; s < 4; ++s) {
        rm_matrix_to_tiles(streams_rm[s], kHout * kWout, streams_tiles[s]);
    }

    // Reference: out[oh, ow, c] = max over (di,dj) of x[2*oh+di, 2*ow+dj, c].
    std::vector<uint16_t> y_ref_rm(kHout * kWout * kC);
    for (uint32_t oh = 0; oh < kHout; ++oh) {
        for (uint32_t ow = 0; ow < kWout; ++ow) {
            for (uint32_t c = 0; c < kC; ++c) {
                float m = -INFINITY;
                for (uint32_t di = 0; di < 2; ++di) {
                    for (uint32_t dj = 0; dj < 2; ++dj) {
                        float v = bf16_to_f32(x_bf16[idx(2 * oh + di, 2 * ow + dj, c)]);
                        if (v > m) m = v;
                    }
                }
                y_ref_rm[(oh * kWout + ow) * kC + c] = f32_to_bf16(m);
            }
        }
    }

    // ---- Device: open, allocate buffers, run ----
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    const uint32_t stream_bytes = kNt * kTileBytes;
    std::array<std::shared_ptr<tt::foil::Buffer>, 4> buf_in;
    for (uint32_t s = 0; s < 4; ++s) {
        buf_in[s] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);
    }
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, stream_bytes, core);

    // One CB slot per input + one for output (single-buffered ring works
    // because compute pops/pushes per tile).
    std::array<std::shared_ptr<tt::foil::Buffer>, 5> cb_bufs;
    for (uint32_t i = 0; i < 5; ++i) {
        cb_bufs[i] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    }

    for (uint32_t s = 0; s < 4; ++s) {
        tt::foil::write_buffer(*dev, *buf_in[s], streams_tiles[s].data(), stream_bytes);
    }
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

    std::array<tt::foil::CbConfig, 5> cbs = {{
        {0,  cb_bufs[0]->device_addr, kTileBytes, 1, kTileBytes},
        {1,  cb_bufs[1]->device_addr, kTileBytes, 1, kTileBytes},
        {2,  cb_bufs[2]->device_addr, kTileBytes, 1, kTileBytes},
        {3,  cb_bufs[3]->device_addr, kTileBytes, 1, kTileBytes},
        {16, cb_bufs[4]->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 5> ra_brisc = {
        static_cast<uint32_t>(buf_in[0]->device_addr),
        static_cast<uint32_t>(buf_in[1]->device_addr),
        static_cast<uint32_t>(buf_in[2]->device_addr),
        static_cast<uint32_t>(buf_in[3]->device_addr),
        kNt,
    };
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(buf_out->device_addr),
        kNt,
    };
    std::array<uint32_t, 1> ra_compute = {kNt};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_compute);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_compute);

    tt::foil::execute(*dev, *kernel);

    // ---- Read back + de-tile ----
    std::vector<uint16_t> y_tiles(stream_bytes / 2, 0);
    tt::foil::read_buffer(*dev, *buf_out, y_tiles.data(), stream_bytes);

    std::vector<uint16_t> y_dev_rm(kHout * kWout * kC);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t t = 0; t < kNt; ++t) {
        tt::foil::test::tile_to_row_major(y_tiles.data() + t * kTileWords, block.data());
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                y_dev_rm[(t * kTileH + r) * kTileW + c] = block[r * kTileW + c];
            }
        }
    }

    // Maxpool over bf16 is exact (no arithmetic, just selection) IF
    // tie-breaking and rounding never round at all. Since we round the
    // *input* through bf16 once before both ref and device, comparing
    // bit-for-bit is safe.
    uint32_t bad = 0, first_bad = y_dev_rm.size();
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
            "test_maxpool_2x2: %u/%zu mismatches; first at idx %u: "
            "got=%.4f expected=%.4f, worst abs diff=%.5f\n",
            bad, y_dev_rm.size(), first_bad,
            bf16_to_f32(y_dev_rm[first_bad]),
            bf16_to_f32(y_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_maxpool_2x2: FAIL");
        return 1;
    }

    std::printf("test_maxpool_2x2: PASS  (Hout*Wout=%u, C=%u, Nt=%u, worst diff=%.5f)\n",
                kHout * kWout, kC, kNt, worst);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_maxpool_2x2: FAIL — %s\n", e.what());
    return 1;
}
