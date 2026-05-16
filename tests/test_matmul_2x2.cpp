// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v6-2: 2×2 grid block-sharded DRAM matmul.
//
// Four Tensix cores, arranged on logical coords (cy, cx) for cy in {0,1}
// and cx in {0,1}. Core (cy, cx) computes C[cy*Mt_pc:(cy+1)*Mt_pc,
// cx*Nt_pc:(cx+1)*Nt_pc]. Each core reads its row-slice of A and its
// column-slice of B, then writes its sub-block of C.
//
// DRAM layout chosen so the matmul kernel (compiled with per-core
// MM_MT/KT/NT) sees a regular contiguous Mt×Kt for A, Kt×Nt for B,
// Mt×Nt for C — no stride awareness in the kernel.
//   A : one global buffer of (2*Mt_pc) × Kt tiles, row-major. Cores
//       (0, *) share the top half, (1, *) the bottom half — same A
//       layout the kernel expects, just offset by (cy * Mt_pc * Kt)
//       tiles from the global base.
//   B : two column-buffers, each Kt × Nt_pc tiles. Cores in column cx
//       share buf_b[cx].
//   C : four per-core buffers, each Mt_pc × Nt_pc tiles. Host stitches
//       them back into the global M×N layout for verification.

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
#define MM_MT 1     // per-core slice height (tiles)
#endif
#ifndef MM_KT
#define MM_KT 4
#endif
#ifndef MM_NT
#define MM_NT 1     // per-core slice width (tiles)
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
constexpr uint32_t kNtPerCore = MM_NT;
constexpr uint32_t kGridY     = 2;
constexpr uint32_t kGridX     = 2;
constexpr uint32_t kCores     = kGridY * kGridX;
constexpr uint32_t kMtGlobal  = kMtPerCore * kGridY;
constexpr uint32_t kNtGlobal  = kNtPerCore * kGridX;
constexpr uint32_t kM         = kMtGlobal  * kTileH;
constexpr uint32_t kK         = kKt        * kTileW;
constexpr uint32_t kN         = kNtGlobal  * kTileW;

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

// Tile A (row-major M×K) into one global tile stream: Mt_global × Kt tiles,
// (mt, kt) row-major. Cores (cy, *) use the slice starting at cy*Mt_pc.
void tile_A_global(const std::vector<uint16_t>& a_rm, std::vector<uint16_t>& out) {
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

// Tile B's cx-th column slice into Kt × Nt_pc tiles, (kt, nt) row-major.
void tile_B_column(const std::vector<uint16_t>& b_rm, uint32_t cx,
                   std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kNtPerCore * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        for (uint32_t nt = 0; nt < kNtPerCore; ++nt) {
            uint32_t global_nt = cx * kNtPerCore + nt;
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] =
                        b_rm[(kt * kTileH + r) * kN + global_nt * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
    }
}

// Stitch four per-core C tile streams into the global M×N row-major matrix.
void stitch_C(const std::array<std::vector<uint16_t>, kCores>& c_per_core,
              std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        uint32_t cy = cid / kGridX;
        uint32_t cx = cid % kGridX;
        for (uint32_t mt = 0; mt < kMtPerCore; ++mt) {
            for (uint32_t nt = 0; nt < kNtPerCore; ++nt) {
                const uint16_t* tile = c_per_core[cid].data() +
                                       (mt * kNtPerCore + nt) * kTileWords;
                tt::foil::test::tile_to_row_major(tile, block.data());
                uint32_t gmt = cy * kMtPerCore + mt;
                uint32_t gnt = cx * kNtPerCore + nt;
                for (uint32_t r = 0; r < kTileH; ++r)
                    for (uint32_t c = 0; c < kTileW; ++c)
                        c_rm[(gmt * kTileH + r) * kN + gnt * kTileW + c] =
                            block[r * kTileW + c];
            }
        }
    }
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const uint32_t a_bytes      = kMtGlobal  * kKt        * kTileBytes;
    const uint32_t b_col_bytes  = kKt        * kNtPerCore * kTileBytes;
    const uint32_t c_core_bytes = kMtPerCore * kNtPerCore * kTileBytes;

    // ---- Host data ----------------------------------------------------
    std::vector<uint16_t> a_rm(kM * kK);
    std::vector<uint16_t> b_rm(kK * kN);
    for (uint32_t r = 0; r < kM; ++r)
        for (uint32_t k = 0; k < kK; ++k)
            a_rm[r * kK + k] = f32_to_bf16(0.005f * static_cast<float>((r + 1) + (k % 7)));
    for (uint32_t k = 0; k < kK; ++k)
        for (uint32_t c = 0; c < kN; ++c)
            b_rm[k * kN + c] = f32_to_bf16(0.005f * static_cast<float>((c + 1) + (k % 5)));

    std::vector<uint16_t> a_stream;
    tile_A_global(a_rm, a_stream);

    std::array<std::vector<uint16_t>, kGridX> b_col_streams;
    for (uint32_t cx = 0; cx < kGridX; ++cx) tile_B_column(b_rm, cx, b_col_streams[cx]);

    std::vector<uint16_t> c_ref_rm;
    matmul_reference(a_rm, b_rm, c_ref_rm);

    // ---- Device + 4 cores ---------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    });
    std::array<tt::foil::CoreCoord, kCores> cores;
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        cores[cid] = tt::foil::CoreCoord{cid / kGridX, cid % kGridX};
    }

    auto buf_a = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes, cores[0]);
    std::array<std::shared_ptr<tt::foil::Buffer>, kGridX> buf_b_cols;
    for (uint32_t cx = 0; cx < kGridX; ++cx) {
        buf_b_cols[cx] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, b_col_bytes, cores[0]);
    }
    std::array<std::shared_ptr<tt::foil::Buffer>, kCores> buf_c_per_core;
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        buf_c_per_core[cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, c_core_bytes, cores[0]);
    }

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), a_bytes);
    for (uint32_t cx = 0; cx < kGridX; ++cx) {
        tt::foil::write_buffer(*dev, *buf_b_cols[cx], b_col_streams[cx].data(), b_col_bytes);
    }
    std::vector<uint8_t> zero_c(c_core_bytes, 0);
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        tt::foil::write_buffer(*dev, *buf_c_per_core[cid], zero_c.data(), c_core_bytes);
    }

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
        uint32_t cy = cid / kGridX;
        uint32_t cx = cid % kGridX;

        buf_cb_a  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        buf_cb_b  [cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        buf_cb_out[cid] = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, cores[cid]);
        kernels   [cid] = tt::foil::load_kernel  (*dev, bins, cores[cid]);

        std::array<tt::foil::CbConfig, 3> cbs = {{
            {0,  buf_cb_a  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {1,  buf_cb_b  [cid]->device_addr, kTileBytes, 1, kTileBytes},
            {16, buf_cb_out[cid]->device_addr, kTileBytes, 1, kTileBytes},
        }};
        tt::foil::register_cbs(*dev, *kernels[cid], cbs);

        // A: per-row-slice base into the global buffer.
        const uint64_t a_off   = buf_a->device_addr + cy * kMtPerCore * kKt * kTileBytes;
        // B: each column shares one buffer; both cores in column cx hit the same DRAM.
        const uint64_t b_off   = buf_b_cols[cx]->device_addr;
        // C: per-core dedicated buffer.
        const uint64_t out_off = buf_c_per_core[cid]->device_addr;

        uint64_t a_noc   = tt::foil::make_noc_dram_addr(*dev, a_off);
        uint64_t b_noc   = tt::foil::make_noc_dram_addr(*dev, b_off);
        uint64_t out_noc = tt::foil::make_noc_dram_addr(*dev, out_off);

        std::array<uint32_t, 7> ra_brisc = {
            static_cast<uint32_t>(a_noc & 0xffffffffu),
            static_cast<uint32_t>(a_noc >> 32),
            static_cast<uint32_t>(b_noc & 0xffffffffu),
            static_cast<uint32_t>(b_noc >> 32),
            kMtPerCore, kKt, kNtPerCore,
        };
        std::array<uint32_t, 3> ra_ncrisc = {
            static_cast<uint32_t>(out_noc & 0xffffffffu),
            static_cast<uint32_t>(out_noc >> 32),
            kMtPerCore * kNtPerCore,
        };
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(*dev, *kernels[cid], R::RiscId::NCRISC, ra_ncrisc);
    }

    tt::foil::execute(*dev, {
        kernels[0].get(), kernels[1].get(), kernels[2].get(), kernels[3].get()
    });

    std::array<std::vector<uint16_t>, kCores> c_per_core;
    for (uint32_t cid = 0; cid < kCores; ++cid) {
        c_per_core[cid].assign(kMtPerCore * kNtPerCore * kTileWords, 0);
        tt::foil::read_buffer(*dev, *buf_c_per_core[cid],
                              c_per_core[cid].data(), c_core_bytes);
    }

    std::vector<uint16_t> c_dev_rm;
    stitch_C(c_per_core, c_dev_rm);

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
            "test_matmul_2x2(per-core Mt=%u Kt=%u Nt=%u, 2x2 cores): "
            "%u/%zu mismatches; first at idx %u (row=%u col=%u): "
            "got=%.5f expected=%.5f, worst abs=%.5f, worst rel=%.4f%%\n",
            kMtPerCore, kKt, kNtPerCore,
            bad, c_dev_rm.size(), first_bad,
            first_bad / kN, first_bad % kN,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_2x2: FAIL");
        return 1;
    }

    std::printf("test_matmul_2x2: PASS  (per-core Mt=%u Kt=%u Nt=%u, 2x2 cores, "
                "global M=%u N=%u K=%u, worst abs=%.5f, worst rel=%.4f%%)\n",
                kMtPerCore, kKt, kNtPerCore, kM, kN, kK,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_2x2: FAIL — %s\n", e.what());
    return 1;
}
