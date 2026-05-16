// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v6-1: row-sharded DRAM matmul across two Tensix cores.
//
// Global problem: A (Mt_global × Kt) tiles, B (Kt × Nt) tiles, C
// (Mt_global × Nt) tiles, all in DRAM. We split M between two cores:
// core 0 owns A[0:Mt_per_core] and C[0:Mt_per_core]; core 1 owns
// A[Mt_per_core:2*Mt_per_core] and C[Mt_per_core:2*Mt_per_core].
// Both cores read the full B.
//
// Each core runs the matmul_dram kernels (built under
// examples/matmul_2core/) — same source, same MM_MT/KT/NT compile-time
// constants. The per-core slice height MM_MT is baked at build time;
// the test computes the global Mt as 2 × MM_MT.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/matmul_2core/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_matmul_2core

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
#define MM_MT 1  // per-core slice height
#endif
#ifndef MM_KT
#define MM_KT 4
#endif
#ifndef MM_NT
#define MM_NT 2
#endif
#ifndef NUM_CORES
#define NUM_CORES 2
#endif

namespace {

using tt::foil::test::kTileH;
using tt::foil::test::kTileW;
using tt::foil::test::kTileBytes;
using tt::foil::test::kTileWords;
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kMtPerCore = MM_MT;
constexpr uint32_t kKt        = MM_KT;
constexpr uint32_t kNt        = MM_NT;
constexpr uint32_t kCores     = NUM_CORES;
constexpr uint32_t kMtGlobal  = kMtPerCore * kCores;
constexpr uint32_t kM         = kMtGlobal  * kTileH;
constexpr uint32_t kK         = kKt        * kTileW;
constexpr uint32_t kN         = kNt        * kTileW;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

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

// Tile A (row-major M×K) into a stream of Mt_global*Kt tiles, tile-row-major
// (stream idx = mt * Kt + kt).
void tile_A_stream(const std::vector<uint16_t>& a_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kMtGlobal) * kKt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMtGlobal; ++mt) {
        for (uint32_t kt = 0; kt < kKt; ++kt) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = a_rm[(mt * kTileH + r) * kK + kt * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

void tile_B_stream(const std::vector<uint16_t>& b_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kNt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = b_rm[(kt * kTileH + r) * kN + nt * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

// Detile output stream (Mt_global × Nt tiles) into row-major M × N.
void untile_C_stream(const std::vector<uint16_t>& c_tiles, std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMtGlobal; ++mt) {
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            const uint16_t* tile = c_tiles.data() + (mt * kNt + nt) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    c_rm[(mt * kTileH + r) * kN + nt * kTileW + c] = block[r * kTileW + c];
        }
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t a_bytes   = kMtGlobal * kKt * kTileBytes;
    const uint32_t b_bytes   = kKt * kNt * kTileBytes;
    const uint32_t out_bytes = kMtGlobal * kNt * kTileBytes;

    std::vector<uint16_t> a_rm(kM * kK);
    std::vector<uint16_t> b_rm(kK * kN);
    for (uint32_t r = 0; r < kM; ++r)
        for (uint32_t k = 0; k < kK; ++k)
            a_rm[r * kK + k] = f32_to_bf16(0.005f * static_cast<float>((r + 1) + (k % 7)));
    for (uint32_t k = 0; k < kK; ++k)
        for (uint32_t c = 0; c < kN; ++c)
            b_rm[k * kN + c] = f32_to_bf16(0.005f * static_cast<float>((c + 1) + (k % 5)));

    std::vector<uint16_t> a_stream, b_stream;
    tile_A_stream(a_rm, a_stream);
    tile_B_stream(b_rm, b_stream);

    std::vector<uint16_t> c_ref_rm;
    matmul_reference(a_rm, b_rm, c_ref_rm);

    // ---- Device + 2 cores --------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {0, 1}});
    std::array<tt::foil::CoreCoord, kCores> cores = {{ {0, 0}, {0, 1} }};

    // A, B, C all in DRAM; CB scratch per core stays in L1.
    auto buf_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes,   cores[0]);
    auto buf_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, b_bytes,   cores[0]);
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, out_bytes, cores[0]);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), a_bytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), b_bytes);
    std::vector<uint8_t> zero(out_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), out_bytes);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};

    std::array<std::shared_ptr<tt::foil::Buffer>, kCores> buf_cb_a, buf_cb_b, buf_cb_out;
    std::array<std::shared_ptr<tt::foil::Kernel>, kCores> kernels;

    for (uint32_t cid = 0; cid < kCores; ++cid) {
        buf_cb_a  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        buf_cb_b  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        buf_cb_out[cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);

        kernels[cid] = tt::foil::load_kernel(*dev, bins, cores[cid]);

        std::array<tt::foil::CbConfig, 3> cbs = {{
            {0,  buf_cb_a  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {1,  buf_cb_b  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {16, buf_cb_out[cid]->device_addr, kTileBytes, 1, kTileBytes},
        }};
        tt::foil::register_cbs(*dev, *kernels[cid], cbs);

        // Per-core DRAM offsets: A and C are sliced; B is shared.
        const uint64_t a_off   = buf_a  ->device_addr + cid * kMtPerCore * kKt * kTileBytes;
        const uint64_t b_off   = buf_b  ->device_addr;
        const uint64_t out_off = buf_out->device_addr + cid * kMtPerCore * kNt * kTileBytes;

        uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, a_off);
        uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, b_off);
        uint64_t out_noc = tt::foil::make_noc_dram_addr(*dev, out_off);

        std::array<uint32_t, 7> ra_brisc = {
            static_cast<uint32_t>(a_noc & 0xffffffffu),
            static_cast<uint32_t>(a_noc >> 32),
            static_cast<uint32_t>(b_noc & 0xffffffffu),
            static_cast<uint32_t>(b_noc >> 32),
            kMtPerCore, kKt, kNt,
        };
        std::array<uint32_t, 3> ra_ncrisc = {
            static_cast<uint32_t>(out_noc & 0xffffffffu),
            static_cast<uint32_t>(out_noc >> 32),
            kMtPerCore * kNt,
        };
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::NCRISC, ra_ncrisc);
    }

    // Fire both cores in one execute(); per-kernel GOs go before any
    // DONE poll so both cores actually run in parallel.
    tt::foil::execute(*dev, {kernels[0].get(), kernels[1].get()});

    std::vector<uint16_t> c_tiles(kMtGlobal * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tiles.data(), out_bytes);

    std::vector<uint16_t> c_dev_rm;
    untile_C_stream(c_tiles, c_dev_rm);

    const float kAbsTol = 0.05f;
    const float kRelTol = 0.01f;
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
            "test_matmul_2core(per-core Mt=%u Kt=%u Nt=%u, %u cores): "
            "%u/%zu mismatches; first at idx %u (row=%u col=%u): "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            kMtPerCore, kKt, kNt, kCores,
            bad, c_dev_rm.size(), first_bad,
            first_bad / kN, first_bad % kN,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_2core: FAIL");
        return 1;
    }

    std::printf("test_matmul_2core: PASS  (per-core Mt=%u Kt=%u Nt=%u, %u cores, "
                "global M=%u N=%u K=%u, worst abs=%.5f, worst rel=%.4f%%)\n",
                kMtPerCore, kKt, kNt, kCores, kM, kN, kK,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_2core: FAIL — %s\n", e.what());
    return 1;
}
