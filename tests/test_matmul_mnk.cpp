// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5-3: general Mt×Kt × Kt×Nt → Mt×Nt single-core matmul.
//
// Tile counts are baked in at build time via -DMM_MT/MM_KT/MM_NT (the
// CMake target mirrors the values into the test). Defaults: Mt=2,
// Kt=4, Nt=2 → 64×128 × 128×64 = 64×64.
//
// Pipeline (single Tensix core, 5 RISCs):
//   BRISC  reader  : pushes A[mt,kt] + B[kt,nt] in (mt,nt,kt) order
//   TRISC  compute : Mt×Nt iterations, each K-loop accumulates into DST[0]
//                    with DST_ACCUM_MODE=true, then pack→push
//   NCRISC writer  : drains Mt*Nt output tiles to out_buf

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

#ifndef MM_MT
#define MM_MT 2
#endif
#ifndef MM_KT
#define MM_KT 4
#endif
#ifndef MM_NT
#define MM_NT 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kMt = MM_MT;
constexpr uint32_t kKt = MM_KT;
constexpr uint32_t kNt = MM_NT;
constexpr uint32_t kM  = kMt * kTileH;
constexpr uint32_t kK  = kKt * kTileW;
constexpr uint32_t kN  = kNt * kTileW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

// C = A * B, fp32 accumulate, final bf16 round (matches DST_ACCUM_MODE=true).
void matmul_reference(const std::vector<uint16_t>& a_rm,
                      const std::vector<uint16_t>& b_rm,
                      std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    for (uint32_t i = 0; i < kM; ++i) {
        for (uint32_t j = 0; j < kN; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < kK; ++k) {
                acc += bf16_to_f32(a_rm[i * kK + k]) *
                       bf16_to_f32(b_rm[k * kN + j]);
            }
            c_rm[i * kN + j] = f32_to_bf16(acc);
        }
    }
}

// Tile A (row-major M×K) into a stream of Mt*Kt tiles, tile-row-major:
// stream index = mt * Kt + kt, where tile = A[mt*32..(mt+1)*32, kt*32..(kt+1)*32].
void tile_A_stream(const std::vector<uint16_t>& a_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kMt) * kKt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMt; ++mt) {
        for (uint32_t kt = 0; kt < kKt; ++kt) {
            for (uint32_t r = 0; r < kTileH; ++r) {
                for (uint32_t c = 0; c < kTileW; ++c) {
                    block[r * kTileW + c] = a_rm[(mt * kTileH + r) * kK + kt * kTileW + c];
                }
            }
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

// Tile B (row-major K×N) into a stream of Kt*Nt tiles, tile-row-major:
// stream index = kt * Nt + nt.
void tile_B_stream(const std::vector<uint16_t>& b_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kNt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            for (uint32_t r = 0; r < kTileH; ++r) {
                for (uint32_t c = 0; c < kTileW; ++c) {
                    block[r * kTileW + c] = b_rm[(kt * kTileH + r) * kN + nt * kTileW + c];
                }
            }
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

// Detile output stream (Mt*Nt tiles in (mt,nt) row-major order) to row-major M×N.
void untile_C_stream(const std::vector<uint16_t>& c_tiles, std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMt; ++mt) {
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            const uint16_t* tile = c_tiles.data() + (mt * kNt + nt) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r) {
                for (uint32_t c = 0; c < kTileW; ++c) {
                    c_rm[(mt * kTileH + r) * kN + nt * kTileW + c] = block[r * kTileW + c];
                }
            }
        }
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t a_bytes   = kMt * kKt * kTileBytes;
    const uint32_t b_bytes   = kKt * kNt * kTileBytes;
    const uint32_t out_bytes = kMt * kNt * kTileBytes;

    // ---- Host inputs: small magnitudes so K=kK partial sums stay
    //      well within bf16 representable range.
    std::vector<uint16_t> a_rm(kM * kK);
    std::vector<uint16_t> b_rm(kK * kN);
    for (uint32_t r = 0; r < kM; ++r) {
        for (uint32_t k = 0; k < kK; ++k) {
            a_rm[r * kK + k] = f32_to_bf16(0.005f * static_cast<float>((r + 1) + (k % 7)));
        }
    }
    for (uint32_t k = 0; k < kK; ++k) {
        for (uint32_t c = 0; c < kN; ++c) {
            b_rm[k * kN + c] = f32_to_bf16(0.005f * static_cast<float>((c + 1) + (k % 5)));
        }
    }

    std::vector<uint16_t> a_stream, b_stream;
    tile_A_stream(a_rm, a_stream);
    tile_B_stream(b_rm, b_stream);

    std::vector<uint16_t> c_ref_rm;
    matmul_reference(a_rm, b_rm, c_ref_rm);

    // ---- Device --------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_a       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, a_bytes,    core);
    auto buf_b       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, b_bytes,    core);
    auto buf_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, out_bytes,  core);
    auto buf_cb_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), a_bytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), b_bytes);
    std::vector<uint16_t> zero(out_bytes / 2, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), out_bytes);

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

    std::array<uint32_t, 5> ra_brisc = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        kMt, kKt, kNt,
    };
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(buf_out->device_addr),
        kMt * kNt,
    };
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> c_tiles(kMt * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tiles.data(), out_bytes);

    std::vector<uint16_t> c_dev_rm;
    untile_C_stream(c_tiles, c_dev_rm);

    // Mixed abs + rel tolerance. bf16 ULP is ~max(2^-7 * |val|, smallest
    // subnormal), so the tighter of (abs 0.05, rel 1%) covers both
    // small-output configs (Mt=Nt=1, |C|≈1) and larger ones where
    // |C| can be tens. With DST_ACCUM_MODE=true we expect <= 1 bf16 ULP.
    const float kAbsTol = 0.05f;
    const float kRelTol = 0.01f;   // 1% — 1.3 bf16 ULPs at any magnitude
    uint32_t bad = 0;
    uint32_t first_bad = static_cast<uint32_t>(c_dev_rm.size());
    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    for (uint32_t i = 0; i < c_dev_rm.size(); ++i) {
        float got = bf16_to_f32(c_dev_rm[i]);
        float exp = bf16_to_f32(c_ref_rm[i]);
        float d   = std::fabs(got - exp);
        float ref = std::fabs(exp);
        float tol = std::max(kAbsTol, kRelTol * ref);
        if (d > worst_abs) worst_abs = d;
        if (ref > 0.f && (d / ref) > worst_rel) worst_rel = d / ref;
        if (d > tol) {
            if (first_bad == c_dev_rm.size()) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::fprintf(stderr,
            "test_matmul_mnk(Mt=%u Kt=%u Nt=%u): %u/%zu mismatches; "
            "first at idx %u (row=%u col=%u): got=%.5f expected=%.5f, "
            "worst abs=%.5f, worst rel=%.4f%%\n",
            kMt, kKt, kNt, bad, c_dev_rm.size(), first_bad,
            first_bad / kN, first_bad % kN,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_mnk: FAIL");
        return 1;
    }

    std::printf("test_matmul_mnk: PASS  (Mt=%u Kt=%u Nt=%u, M=%u N=%u K=%u, "
                "worst abs=%.5f, worst rel=%.4f%%, tol=max(%.2f, %.0f%%))\n",
                kMt, kKt, kNt, kM, kN, kK, worst_abs, worst_rel * 100.0f,
                kAbsTol, kRelTol * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_mnk: FAIL — %s\n", e.what());
    return 1;
}
