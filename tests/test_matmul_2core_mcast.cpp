// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v6-3: 2-core row-sharded DRAM matmul with B forwarded over NOC.
//
// Difference from v6-1 (matmul_2core):
//   • Producer core reads the entire B stream from DRAM into a local
//     L1 staging buffer (Kt × 2 KB), then NOC-writes that buffer plus
//     a 4-byte ready flag to the consumer core.
//   • Consumer core busy-waits on its local flag, then iterates the
//     matmul without touching DRAM for B — its B tiles come from L1
//     staging instead.
// Result: B is read from DRAM once total, vs. twice in v6-1. Same
// numeric output to within bf16 rounding.

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
#define MM_MT 1
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

constexpr uint32_t kMtPerCore = MM_MT;
constexpr uint32_t kKt        = MM_KT;
constexpr uint32_t kNt        = MM_NT;
constexpr uint32_t kCores     = 2;
constexpr uint32_t kMtGlobal  = kMtPerCore * kCores;
constexpr uint32_t kM         = kMtGlobal  * kTileH;
constexpr uint32_t kK         = kKt        * kTileW;
constexpr uint32_t kN         = kNt        * kTileW;
// Staging must hold the full B substream the cores will iterate, i.e.
// all Kt × Nt tiles (kt-major). v6-3 uses each B tile multiple times
// across the Mt rows, so caching the whole thing in L1 saves DRAM
// traffic regardless of Mt.
constexpr uint32_t kBStageBytes = kKt * kNt * kTileBytes;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

void matmul_reference(const std::vector<uint16_t>& a_rm,
                      const std::vector<uint16_t>& b_rm,
                      std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    for (uint32_t i = 0; i < kM; ++i)
        for (uint32_t j = 0; j < kN; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < kK; ++k)
                acc += bf16_to_f32(a_rm[i * kK + k]) *
                       bf16_to_f32(b_rm[k * kN + j]);
            c_rm[i * kN + j] = f32_to_bf16(acc);
        }
}

void tile_A_stream(const std::vector<uint16_t>& a_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kMtGlobal) * kKt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMtGlobal; ++mt)
        for (uint32_t kt = 0; kt < kKt; ++kt) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = a_rm[(mt * kTileH + r) * kK + kt * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}

void tile_B_stream(const std::vector<uint16_t>& b_rm, std::vector<uint16_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(kKt) * kNt * kTileWords);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t kt = 0; kt < kKt; ++kt)
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    block[r * kTileW + c] = b_rm[(kt * kTileH + r) * kN + nt * kTileW + c];
            tt::foil::test::row_major_to_tile(block.data(), out);
        }
}

void untile_C_stream(const std::vector<uint16_t>& c_tiles, std::vector<uint16_t>& c_rm) {
    c_rm.assign(kM * kN, 0);
    std::vector<uint16_t> block(kTileH * kTileW);
    for (uint32_t mt = 0; mt < kMtGlobal; ++mt)
        for (uint32_t nt = 0; nt < kNt; ++nt) {
            const uint16_t* tile = c_tiles.data() + (mt * kNt + nt) * kTileWords;
            tt::foil::test::tile_to_row_major(tile, block.data());
            for (uint32_t r = 0; r < kTileH; ++r)
                for (uint32_t c = 0; c < kTileW; ++c)
                    c_rm[(mt * kTileH + r) * kN + nt * kTileW + c] = block[r * kTileW + c];
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

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {0, 1}});
    tt::foil::CoreCoord c_prod{0, 0}, c_cons{0, 1};

    // ---- DRAM buffers (A, B, C) -------------------------------------
    auto buf_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, a_bytes,   c_prod);
    auto buf_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, b_bytes,   c_prod);
    auto buf_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, out_bytes, c_prod);

    tt::foil::write_buffer(*dev, *buf_a, a_stream.data(), a_bytes);
    tt::foil::write_buffer(*dev, *buf_b, b_stream.data(), b_bytes);
    std::vector<uint8_t> zero(out_bytes, 0);
    tt::foil::write_buffer(*dev, *buf_out, zero.data(), out_bytes);

    // ---- Per-core L1: CBs + B staging + flag ------------------------
    auto cbp_a       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_prod);
    auto cbp_b       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_prod);
    auto cbp_out     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_prod);
    auto stage_p     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kBStageBytes, c_prod);
    auto flag_src_p  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c_prod);
    // Producer-side checkpoint buffer used by the kernel for debug
    // checkpoints; harmless if the production kernel still writes to it.
    auto ckpt_p      = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 16 * sizeof(uint32_t), c_prod);

    // Pre-fill the producer's flag-source word with 1 so the producer
    // kernel can just NOC-copy 4 bytes to the consumer's flag slot.
    uint32_t flag_one = 1;
    tt::foil::write_buffer(*dev, *flag_src_p, &flag_one, sizeof(flag_one));

    auto cbc_a   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_cons);
    auto cbc_b   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_cons);
    auto cbc_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes,   c_cons);
    auto stage_c = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kBStageBytes, c_cons);
    auto flag_c  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c_cons);

    uint32_t flag_zero = 0;
    tt::foil::write_buffer(*dev, *flag_c, &flag_zero, sizeof(flag_zero));

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins_prod = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader_producer.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    std::array<R, 5> bins_cons = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader_consumer.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};

    auto kp = tt::foil::load_kernel(*dev, bins_prod, c_prod);
    auto kc = tt::foil::load_kernel(*dev, bins_cons, c_cons);

    std::array<tt::foil::CbConfig, 3> cbs_p = {{
        {0,  cbp_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  cbp_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, cbp_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    std::array<tt::foil::CbConfig, 3> cbs_c = {{
        {0,  cbc_a  ->device_addr, kTileBytes, 1, kTileBytes},
        {1,  cbc_b  ->device_addr, kTileBytes, 1, kTileBytes},
        {16, cbc_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kp, cbs_p);
    tt::foil::register_cbs(*dev, *kc, cbs_c);

    // ---- Per-core DRAM + NOC addresses ------------------------------
    const uint64_t a_prod_off = buf_a  ->device_addr + 0 * kMtPerCore * kKt * kTileBytes;
    const uint64_t a_cons_off = buf_a  ->device_addr + 1 * kMtPerCore * kKt * kTileBytes;
    const uint64_t out_prod_off = buf_out->device_addr + 0 * kMtPerCore * kNt * kTileBytes;
    const uint64_t out_cons_off = buf_out->device_addr + 1 * kMtPerCore * kNt * kTileBytes;

    const uint64_t a_prod_noc   = tt::foil::make_noc_dram_addr(*dev, a_prod_off);
    const uint64_t a_cons_noc   = tt::foil::make_noc_dram_addr(*dev, a_cons_off);
    const uint64_t b_dram_noc   = tt::foil::make_noc_dram_addr(*dev, buf_b->device_addr);
    const uint64_t out_prod_noc = tt::foil::make_noc_dram_addr(*dev, out_prod_off);
    const uint64_t out_cons_noc = tt::foil::make_noc_dram_addr(*dev, out_cons_off);

    // Consumer's staging + flag, addressed from producer's NOC view.
    const uint64_t stage_c_noc = tt::foil::make_noc_unicast_addr(*dev, c_cons, stage_c->device_addr);
    const uint64_t flag_c_noc  = tt::foil::make_noc_unicast_addr(*dev, c_cons, flag_c ->device_addr);

    auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); };
    auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };

    std::array<uint32_t, 14> ra_p = {
        lo(a_prod_noc), hi(a_prod_noc),
        lo(b_dram_noc), hi(b_dram_noc),
        lo(stage_c_noc), hi(stage_c_noc),
        lo(flag_c_noc),  hi(flag_c_noc),
        static_cast<uint32_t>(stage_p   ->device_addr),
        static_cast<uint32_t>(flag_src_p->device_addr),
        kMtPerCore, kKt, kNt,
        static_cast<uint32_t>(ckpt_p->device_addr),
    };
    std::array<uint32_t, 3> ra_p_writer = {
        lo(out_prod_noc), hi(out_prod_noc), kMtPerCore * kNt,
    };

    std::array<uint32_t, 7> ra_c = {
        lo(a_cons_noc), hi(a_cons_noc),
        static_cast<uint32_t>(stage_c->device_addr),
        static_cast<uint32_t>(flag_c ->device_addr),
        kMtPerCore, kKt, kNt,
    };
    std::array<uint32_t, 3> ra_c_writer = {
        lo(out_cons_noc), hi(out_cons_noc), kMtPerCore * kNt,
    };

    tt::foil::set_runtime_args(*dev, *kp, R::RiscId::BRISC,  ra_p);
    tt::foil::set_runtime_args(*dev, *kp, R::RiscId::NCRISC, ra_p_writer);
    tt::foil::set_runtime_args(*dev, *kc, R::RiscId::BRISC,  ra_c);
    tt::foil::set_runtime_args(*dev, *kc, R::RiscId::NCRISC, ra_c_writer);

    tt::foil::execute(*dev, {kp.get(), kc.get()});

    std::vector<uint16_t> c_tiles(kMtGlobal * kNt * kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_out, c_tiles.data(), out_bytes);

    std::vector<uint16_t> c_dev_rm;
    untile_C_stream(c_tiles, c_dev_rm);

    const float kAbsTol = 0.05f;
    const float kRelTol = 0.01f;
    uint32_t bad = 0;
    uint32_t first_bad = static_cast<uint32_t>(c_dev_rm.size());
    float worst_abs = 0.0f, worst_rel = 0.0f;
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
            "test_matmul_2core_mcast(per-core Mt=%u Kt=%u Nt=%u, 2 cores): "
            "%u/%zu mismatches; first at idx %u: got=%.5f expected=%.5f, "
            "worst abs=%.5f, worst rel=%.4f%%\n",
            kMtPerCore, kKt, kNt, bad, c_dev_rm.size(), first_bad,
            bf16_to_f32(c_dev_rm[first_bad]),
            bf16_to_f32(c_ref_rm[first_bad]),
            worst_abs, worst_rel * 100.0f);
        tt::foil::close_device(std::move(dev));
        std::puts("test_matmul_2core_mcast: FAIL");
        return 1;
    }

    std::printf("test_matmul_2core_mcast: PASS  (per-core Mt=%u Kt=%u Nt=%u, 2 cores, "
                "global M=%u N=%u K=%u, B from DRAM 1× then NOC, "
                "worst abs=%.5f, worst rel=%.4f%%)\n",
                kMtPerCore, kKt, kNt, kM, kN, kK,
                worst_abs, worst_rel * 100.0f);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_matmul_2core_mcast: FAIL — %s\n", e.what());
    return 1;
}
