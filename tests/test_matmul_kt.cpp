// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5-2: K-loop bf16 matmul. M = N = 32 (one output tile),
// K = Kt × 32 (Kt tiles in the reduction dim). DST_ACCUM_MODE=true so
// Kt partial products sum exactly in fp32 DST registers, then a single
// pack rounds the final result to bf16.
//
// Pipeline:
//   BRISC  (reader)  : pushes Kt A tiles + Kt B tiles into CB c_0 / c_1
//   TRISC0/1/2       : mm_init → for kt in Kt: matmul_tiles(...,0) → pack
//   NCRISC (writer)  : drains CB c_16 to L1 out_buf
//
// Both the kernel ELFs and this test bake Kt at compile time. Default is
// Kt=4 (so K = 128).  To rebuild for a different Kt:
//   MM_KT=8 ./examples/matmul_kt/build_kernels.sh
//   cmake --build build -DMM_KT=8 --target test_matmul_kt
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/matmul_kt/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_matmul_kt

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

#ifndef MM_KT
#define MM_KT 4
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kKt = MM_KT;
constexpr uint32_t kK  = kKt * kTileW;   // total K dimension

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// Reference: C = A * B in fp32, then bf16-round at the end.  This matches
// the device pipeline well when DST_ACCUM_MODE=true (fp32 accumulation in
// DST regs).  A is row-major M×K (32×kK), B is row-major K×N (kK×32).
void matmul_reference(const std::vector<uint16_t>& a_rm,
                      const std::vector<uint16_t>& b_rm,
                      std::vector<uint16_t>& c_rm) {
    c_rm.assign(kTileWords, 0);
    for (uint32_t i = 0; i < kTileH; ++i) {
        for (uint32_t j = 0; j < kTileW; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < kK; ++k) {
                acc += bf16_to_f32(a_rm[i * kK + k]) *
                       bf16_to_f32(b_rm[k * kTileW + j]);
            }
            c_rm[i * kTileW + j] = f32_to_bf16(acc);
        }
    }
}

// Tile A (row-major M×K, M=32, K=kKt*32) into a stream of kKt 32×32 tiles
// laid out in K-tile order: tile 0 is A[:, 0:32], tile 1 is A[:, 32:64], …
void tile_A_stream(const std::vector<uint16_t>& a_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                block[r * kTileW + c] = a_rm[r * kK + kt * kTileW + c];
            }
        }
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

// Tile B (row-major K×N, K=kKt*32, N=32) into a stream of kKt 32×32 tiles
// in K-tile order: tile 0 is B[0:32, :], tile 1 is B[32:64, :], …
void tile_B_stream(const std::vector<uint16_t>& b_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        for (uint32_t r = 0; r < kTileH; ++r) {
            for (uint32_t c = 0; c < kTileW; ++c) {
                block[r * kTileW + c] = b_rm[(kt * kTileH + r) * kTileW + c];
            }
        }
        tt::foil::test::row_major_to_tile(block.data(), out);
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t a_bytes  = kKt * kTileBytes;
    const uint32_t b_bytes  = kKt * kTileBytes;

    // ---- Host inputs ---------------------------------------------------
    // A: small magnitude, varies with row+k; B: varies with k+col.
    // Total magnitude after K=kK multiplies-add stays under ~10 so bf16
    // accuracy is OK (we'd see saturation noise above ~256).
    std::vector<uint16_t> a_rm(kTileH * kK);
    std::vector<uint16_t> b_rm(kK * kTileW);
    for (uint32_t r = 0; r < kTileH; ++r) {
        for (uint32_t k = 0; k < kK; ++k) {
            a_rm[r * kK + k] = f32_to_bf16(0.005f * static_cast<float>((r + 1) * 1u + (k % 7)));
        }
    }
    for (uint32_t k = 0; k < kK; ++k) {
        for (uint32_t c = 0; c < kTileW; ++c) {
            b_rm[k * kTileW + c] = f32_to_bf16(0.005f * static_cast<float>((c + 1) * 1u + (k % 5)));
        }
    }

    std::vector<uint16_t> a_stream, b_stream;
    tile_A_stream(a_rm, a_stream);
    tile_B_stream(b_rm, b_stream);

    // ---- Reference -----------------------------------------------------
    std::vector<uint16_t> c_ref_rm;
    matmul_reference(a_rm, b_rm, c_ref_rm);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, a_bytes,     core);
    auto buf_b       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, b_bytes,     core);
    auto buf_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), a_bytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), b_bytes);
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

    // CBs hold 1 tile each — reader pushes a tile, compute consumes it,
    // reader pushes the next one. Backpressure naturally serialises.
    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  buf_cb_a->device_addr,   kTileBytes, 1, kTileBytes},
        {1,  buf_cb_b->device_addr,   kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 3> ra_brisc  = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        kKt,
    };
    std::array<uint32_t, 1> ra_ncrisc = {static_cast<uint32_t>(buf_out->device_addr)};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> c_tile(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tile.data(), kTileBytes);

    std::vector<uint16_t> c_dev_rm(kTileWords);
    tt::foil::test::tile_to_row_major(c_tile.data(), c_dev_rm.data());

    // ---- Compare -------------------------------------------------------
    // With fp32 DST accum, the device output should match the fp32 host
    // reference within ~1 bf16 ULP at the result magnitude.  Our inputs
    // give results in roughly [0.1, 5.0] range; bf16 ULP there is up to
    // ~0.04.  Tolerance 0.05 keeps margin without hiding real bugs.
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
            "test_matmul_kt(Kt=%u): %u/%u mismatches; first at idx %u "
            "(row=%u col=%u): got=%.5f expected=%.5f, worst abs diff=%.5f\n",
            kKt, bad, kTileWords, first_bad,
            first_bad / kTileW, first_bad % kTileW,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_kt: FAIL");
        return 1;
    }

    std::printf("test_matmul_kt: PASS  (Kt=%u, K=%u, worst abs diff=%.5f, tol=%.5f)\n",
                kKt, kK, worst, kAbsTol);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_kt: FAIL — %s\n", e.what());
    return 1;
}
