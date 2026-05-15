// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5-1: single-tile bf16 matmul through all 5 Tensix RISCs.
//
// Pipeline (mirrors tile_copy + 2nd input + matmul_tiles):
//   BRISC  (reader)  : L1 a_buf → CB c_0, L1 b_buf → CB c_1
//   TRISC0/1/2       : mm_init → matmul_tiles → pack to CB c_16
//   NCRISC (writer)  : CB c_16 → L1 out_buf
//
// One bf16 32×32 matrix on each input, one bf16 32×32 result. Host fills
// A and B with deterministic patterns, computes a reference matmul in
// float, quantises to bf16, and compares against device output with a
// small absolute tolerance to absorb HiFi4 rounding.
//
// Tile layout: tt-metal bf16 tiles are 4 faces of 16×16 each:
//   face 0 = rows[0..16),  cols[0..16)
//   face 1 = rows[0..16),  cols[16..32)
//   face 2 = rows[16..32), cols[0..16)
//   face 3 = rows[16..32), cols[16..32)
// Within a face, elements are row-major. This is the layout matmul_tiles
// expects on UNPACK and produces on PACK.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/matmul_1tile/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_matmul_1tile

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"

namespace {

constexpr uint32_t kTileH = 32;
constexpr uint32_t kTileW = 32;
constexpr uint32_t kFaceH = 16;
constexpr uint32_t kFaceW = 16;
constexpr uint32_t kTileBytes = kTileH * kTileW * 2;   // bf16
constexpr uint32_t kTileWords = kTileBytes / sizeof(uint16_t);

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// bf16 = top 16 bits of float32, round-to-nearest-even.
uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even on the truncated 16 LSBs.
    uint32_t lsb     = (u >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    u += rounding;
    return static_cast<uint16_t>(u >> 16);
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Convert a 32×32 row-major bf16 matrix into the 4-face tile layout
// matmul_tiles expects. `rm` is row-major (row * 32 + col); `tile` is the
// tt-metal tile layout described in the file header.
void row_major_to_tile(const std::vector<uint16_t>& rm, std::vector<uint16_t>& tile) {
    tile.resize(kTileWords);
    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t row_off = (face / 2) * kFaceH;
        uint32_t col_off = (face % 2) * kFaceW;
        for (uint32_t r = 0; r < kFaceH; ++r) {
            for (uint32_t c = 0; c < kFaceW; ++c) {
                uint32_t src = (row_off + r) * kTileW + (col_off + c);
                uint32_t dst = face * (kFaceH * kFaceW) + r * kFaceW + c;
                tile[dst] = rm[src];
            }
        }
    }
}

void tile_to_row_major(const std::vector<uint16_t>& tile, std::vector<uint16_t>& rm) {
    rm.resize(kTileWords);
    for (uint32_t face = 0; face < 4; ++face) {
        uint32_t row_off = (face / 2) * kFaceH;
        uint32_t col_off = (face % 2) * kFaceW;
        for (uint32_t r = 0; r < kFaceH; ++r) {
            for (uint32_t c = 0; c < kFaceW; ++c) {
                uint32_t src = face * (kFaceH * kFaceW) + r * kFaceW + c;
                uint32_t dst = (row_off + r) * kTileW + (col_off + c);
                rm[dst] = tile[src];
            }
        }
    }
}

// Reference: C = A * B in float, with each accumulation rounded to bf16
// (closest fidelity model to what HiFi4 produces is fp32 accum then bf16
// round at the end — that's what we use, and we set tolerance accordingly).
void matmul_reference(const std::vector<uint16_t>& a_rm,
                      const std::vector<uint16_t>& b_rm,
                      std::vector<uint16_t>& c_rm) {
    c_rm.assign(kTileWords, 0);
    for (uint32_t i = 0; i < kTileH; ++i) {
        for (uint32_t j = 0; j < kTileW; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < kTileW; ++k) {
                acc += bf16_to_f32(a_rm[i * kTileW + k]) *
                       bf16_to_f32(b_rm[k * kTileW + j]);
            }
            c_rm[i * kTileW + j] = f32_to_bf16(acc);
        }
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // ---- Host inputs ---------------------------------------------------
    // A: small ramp, B: small ramp on the other axis, both bf16. Small
    // enough that K=32 sums stay well within bf16's representable range.
    std::vector<uint16_t> a_rm(kTileWords);
    std::vector<uint16_t> b_rm(kTileWords);
    for (uint32_t r = 0; r < kTileH; ++r) {
        for (uint32_t c = 0; c < kTileW; ++c) {
            a_rm[r * kTileW + c] = f32_to_bf16(0.01f * static_cast<float>(r + 1));
            b_rm[r * kTileW + c] = f32_to_bf16(0.01f * static_cast<float>(c + 1));
        }
    }

    std::vector<uint16_t> a_tile, b_tile;
    row_major_to_tile(a_rm, a_tile);
    row_major_to_tile(b_rm, b_tile);

    // ---- Reference -----------------------------------------------------
    std::vector<uint16_t> c_ref_rm;
    matmul_reference(a_rm, b_rm, c_ref_rm);

    // ---- Device --------------------------------------------------------
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

    std::vector<uint16_t> c_dev_rm;
    tile_to_row_major(c_tile, c_dev_rm);

    // ---- Compare -------------------------------------------------------
    // HiFi4 bf16 matmul accumulates in DST regs.  With
    // DST_ACCUM_MODE=false (our v5-1 setting), DST is fp16-ish — so
    // intermediate rounds are coarser than the fp32 reference and a few
    // ULPs of drift are expected.  Worst observed on initial v5-1 bring-up
    // was 0.03125 (1 bf16 ULP at the ~1.0 magnitude range).  Tolerance
    // 0.05 covers that with margin; v5-2 will revisit with
    // DST_ACCUM_MODE=true once we add K-loop accumulation.
    const float kAbsTol = 0.05f;
    uint32_t bad = 0;
    uint32_t first_bad = kTileWords;
    float worst = 0.0f;
    for (uint32_t i = 0; i < kTileWords; ++i) {
        float got = bf16_to_f32(c_dev_rm[i]);
        float exp = bf16_to_f32(c_ref_rm[i]);
        float d = std::fabs(got - exp);
        if (d > worst) worst = d;
        if (d > kAbsTol) {
            if (first_bad == kTileWords) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_matmul_1tile: %u/%u mismatches; first at idx %u "
            "(row=%u col=%u): got=%.5f expected=%.5f, worst abs diff=%.5f\n",
            bad, kTileWords, first_bad, first_bad / kTileW, first_bad % kTileW,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_1tile: FAIL");
        return 1;
    }

    std::printf("test_matmul_1tile: PASS  (worst abs diff=%.5f, tol=%.5f)\n",
                worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_1tile: FAIL — %s\n", e.what());
    return 1;
}
