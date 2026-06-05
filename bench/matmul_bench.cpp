// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Profiling harness (NOT a ctest): single-core matmul wall time at the
// per-core decode shapes of qwen3_vl_2b, comparing the real reader
// (streams the full B weight matrix from DRAM) against a "no-Bread"
// variant reader that skips the B DRAM reads but keeps the identical CB
// protocol so compute + writer run unchanged.
//
//   T_full   = read B from DRAM + compute (Nt*Kt matmul_tiles) + write C
//   T_noread = compute + write C  (B DRAM reads removed)
//
// If T_noread << T_full the matmul is bound by streaming the weights from
// DRAM (memory-bound decode). If T_noread ~= T_full it is compute/write
// bound. Each (shape,config) gets a FRESH device — make_matmul bumps the
// per-core KERNEL_CONFIG + L1 arenas and never frees, so stacking many on
// one core overflows the CB-blob region and hangs cb_reserve_back.
//
// Easiest entry point is bench/run.sh, which builds the no-Bread variant and
// this target, then runs both decode and prefill sweeps. To drive directly:
//   TT_METAL_ROOT=third_party/tt-metal bash ops/matmul/build.sh
//   TT_METAL_ROOT=third_party/tt-metal bash bench/build_noBread.sh
//   TT_VISIBLE_DEVICES=0 TT_FOIL_DEVICE=0 TT_FOIL_OPS_DIR=ops MM_MODE=decode \
//     ./build/bench/matmul_bench
// MM_MODE = decode (Nt-sweep) | prefill (Mt-sweep) | cores (core-count sweep).
// MM_NOBREAD_DIR defaults to bench/prebuilt_noBread.
//
// MM_MODE=cores answers "how many cores saturate DRAM for each decode-shape
// GEMV?" — it takes each decode matmul at its real GLOBAL Nt and shards it
// over n_cores ∈ {1,2,4,8,16,32} (+64 for lm_head) via make_matmul_grid,
// reporting worker-exec µs vs cores. The knee (where T_full stops dropping)
// is the DRAM-saturation core count used to size the decode grids in
// models/qwen3_vl_2b. read_frac (= (T_full-T_noBread)/T_full) shows whether
// each width is still memory-bound.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"

using namespace tt::foil;
namespace ol = tt::foil::op_lib;
using Clock = std::chrono::steady_clock;

namespace {
constexpr uint32_t kTileWords = 32 * 32;
constexpr std::size_t kTileBytes = kTileWords * 2;

struct Shape { const char* tag; uint32_t Mt, Kt, Nt; };  // Nt = per-core slice

// Fresh device per call so the per-core bump arenas start clean.
double bench(int pcie, const std::string& kdir, const Shape& s, int iters) {
    auto dev = open_device(pcie, "", {{0, 0}});
    auto a = ol::allocate_tensor_dram(*dev, s.Mt * s.Kt);
    auto b = ol::allocate_tensor_dram(*dev, s.Kt * s.Nt);
    ol::TensorDesc c; c.num_tiles = 0;
    std::vector<uint16_t> fill(s.Kt * s.Nt * kTileWords, 0x3c00 /*1.0 bf16*/);
    write_buffer(*dev, *a.buf, fill.data(), s.Mt * s.Kt * kTileBytes);
    write_buffer(*dev, *b.buf, fill.data(), s.Kt * s.Nt * kTileBytes);

    auto op = ol::make_matmul(*dev, a, b, c, s.Mt, s.Kt, s.Nt, {0, 0}, kdir);
    for (int i = 0; i < 5; ++i) ol::execute(*dev, op);  // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) ol::execute(*dev, op);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    close_device(std::move(dev));
    return us;
}

// Dense grid within the known-valid logical Tensix region (y in 0..7, the
// range the qwen3 model already boots). The bench opens its own fresh device
// so there is no occupancy to avoid — any valid logical core is fair game.
std::vector<CoreCoord> make_grid(uint32_t n_cores) {
    std::vector<CoreCoord> g;
    g.reserve(n_cores);
    for (uint32_t i = 0; i < n_cores; ++i) g.push_back({i / 8, i % 8});
    return g;
}

// Grid matmul wall time: GLOBAL Nt sharded over n_cores via make_matmul_grid.
// Mirrors bench() but boots the whole grid and uses the multi-core factory.
double bench_grid(int pcie, const std::string& kdir, const Shape& s,
                  uint32_t n_cores, int iters) {
    auto grid = make_grid(n_cores);
    auto dev = open_device(pcie, "", grid);
    auto a = ol::allocate_tensor_dram(*dev, s.Mt * s.Kt);
    auto b = ol::allocate_tensor_dram(*dev, s.Kt * s.Nt);
    ol::TensorDesc c; c.num_tiles = 0;
    std::vector<uint16_t> fill(s.Kt * s.Nt * kTileWords, 0x3c00 /*1.0 bf16*/);
    write_buffer(*dev, *a.buf, fill.data(), s.Mt * s.Kt * kTileBytes);
    write_buffer(*dev, *b.buf, fill.data(), s.Kt * s.Nt * kTileBytes);

    auto op = ol::make_matmul_grid(*dev, a, b, c, s.Mt, s.Kt, s.Nt, grid, kdir);
    for (int i = 0; i < 5; ++i) ol::execute(*dev, op);  // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) ol::execute(*dev, op);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    close_device(std::move(dev));
    return us;
}

// Multi-channel B: n_cores = n_channels, core c reads its B column-slab from
// DRAM channel c (set_matmul_grid_args_sharded). B is written to every used
// channel; A + out stay on channel 0. Measures whether spreading the weight
// read across channels lifts the single-channel bandwidth ceiling.
double bench_channels(int pcie, const std::string& kdir, const Shape& s,
                      uint32_t n_channels, int iters) {
    auto grid = make_grid(n_channels);  // one core per channel
    auto dev = open_device(pcie, "", grid);
    auto a = ol::allocate_tensor_dram(*dev, s.Mt * s.Kt);
    auto b = ol::allocate_tensor_dram(*dev, s.Kt * s.Nt);
    ol::TensorDesc c; c.num_tiles = 0;
    std::vector<uint16_t> fill(s.Kt * s.Nt * kTileWords, 0x3c00 /*1.0 bf16*/);
    write_buffer(*dev, *a.buf, fill.data(), s.Mt * s.Kt * kTileBytes);
    // B must be present at b.buf->device_addr on every channel a core reads.
    for (uint32_t ch = 0; ch < n_channels; ++ch)
        write_dram_channel(*dev, ch, b.buf->device_addr, fill.data(),
                           (std::size_t)s.Kt * s.Nt * kTileBytes);

    auto op = ol::make_matmul_grid(*dev, a, b, c, s.Mt, s.Kt, s.Nt, grid, kdir);
    ol::set_matmul_grid_args_sharded(*dev, op, a, b, c, s.Mt, s.Kt, s.Nt, grid, n_channels);
    for (int i = 0; i < 5; ++i) ol::execute(*dev, op);  // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) ol::execute(*dev, op);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    close_device(std::move(dev));
    return us;
}
}  // namespace

int main() {
    const char* nb = std::getenv("MM_NOBREAD_DIR");
    const std::string noBread = nb ? nb : "bench/prebuilt_noBread";
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie = dev_env ? std::atoi(dev_env) : 0;

    const char* mode = std::getenv("MM_MODE");  // "decode" | "prefill" | "cores"
    const std::string mode_s = mode ? mode : "decode";
    const bool prefill = mode_s == "prefill";
    const bool cores_mode = mode_s == "cores";
    const bool channels_mode = mode_s == "channels";

    if (channels_mode) {
        const char* nb_c = std::getenv("MM_NOBREAD_DIR");
        const std::string noBread_c = nb_c ? nb_c : "bench/prebuilt_noBread";
        const int iters_c = 50;
        // Read-heavy decode shapes; one core per channel, sweep channel count.
        struct Sweep { Shape s; std::vector<uint32_t> nch; };
        const std::vector<Sweep> sweeps = {
            {{"ffn_gu",   1, 64,  384},  {1, 2, 4, 8}},
            {{"ffn_down", 1, 192, 64},   {1, 2, 4, 8}},
            {{"lm_head",  1, 64,  4748}, {1, 2, 4, 8}},
        };
        // One-time sanity: write a per-channel sentinel + read it back, so a
        // mis-resolved channel core surfaces here, not as silent timing noise.
        {
            auto dev = open_device(pcie, "", {{0, 0}});
            uint32_t nch = num_dram_channels(*dev);
            std::printf("DRAM channels resolved: %u\n", nch);
            const uint64_t addr = 0x1000;  // within UNRESERVED DRAM
            bool ok = true;
            for (uint32_t ch = 0; ch < nch; ++ch) {
                uint32_t w = 0xC0DE0000u | ch, r = 0;
                write_dram_channel(*dev, ch, addr, &w, sizeof(w));
                read_dram_channel(*dev, ch, addr, &r, sizeof(r));
                if (r != w) { ok = false; std::printf("  ch%u RT FAIL w=%08x r=%08x\n", ch, w, r); }
            }
            std::printf("channel round-trip: %s\n", ok ? "OK" : "FAILED");
            close_device(std::move(dev));
        }
        std::printf("%-9s Kt   Nt   chan  T_full(us)  T_noBread(us)  read_frac\n", "shape");
        std::printf("----------------------------------------------------------------\n");
        for (const auto& sw : sweeps) {
            for (uint32_t nc : sw.nch) {
                double tf = bench_channels(pcie, "",        sw.s, nc, iters_c);
                double tn = bench_channels(pcie, noBread_c, sw.s, nc, iters_c);
                double frac = (tf > 0) ? (tf - tn) / tf * 100.0 : 0.0;
                std::printf("%-9s %4u %4u  %4u   %9.2f    %9.2f      %5.1f%%\n",
                            sw.s.tag, sw.s.Kt, sw.s.Nt, nc, tf, tn, frac);
                std::fflush(stdout);
            }
            std::printf("----------------------------------------------------------------\n");
        }
        return 0;
    }

    if (cores_mode) {
        // Real decode matmul shapes at their GLOBAL Nt; sweep n_cores.
        struct Sweep { Shape s; std::vector<uint32_t> ncores; };
        const std::vector<Sweep> sweeps = {
            {{"qkv",      1, 64,  128},  {1, 2, 4, 8, 16, 32}},
            {{"o",        1, 64,  64},   {1, 2, 4, 8, 16, 32}},
            {{"ffn_gu",   1, 64,  384},  {1, 2, 4, 8, 16, 32}},
            {{"ffn_down", 1, 192, 64},   {1, 2, 4, 8, 16, 32}},
            {{"lm_head",  1, 64,  4748}, {1, 2, 4, 8, 16, 32, 64}},
        };
        const char* nb_c = std::getenv("MM_NOBREAD_DIR");
        const std::string noBread_c = nb_c ? nb_c : "bench/prebuilt_noBread";
        const int iters_c = 50;
        std::printf("%-9s Kt   Nt   cores  T_full(us)  T_noBread(us)  read_frac\n", "shape");
        std::printf("----------------------------------------------------------------\n");
        for (const auto& sw : sweeps) {
            for (uint32_t nc : sw.ncores) {
                double tf = bench_grid(pcie, "",        sw.s, nc, iters_c);
                double tn = bench_grid(pcie, noBread_c, sw.s, nc, iters_c);
                double frac = (tf > 0) ? (tf - tn) / tf * 100.0 : 0.0;
                std::printf("%-9s %4u %4u  %4u   %9.2f    %9.2f      %5.1f%%\n",
                            sw.s.tag, sw.s.Kt, sw.s.Nt, nc, tf, tn, frac);
                std::fflush(stdout);
            }
            std::printf("----------------------------------------------------------------\n");
        }
        return 0;
    }
    const std::vector<Shape> decode_shapes = {
        // Nt-sweep at Kt=64 (double-buffered cb_b) — slope cancels the
        // per-execute slow-dispatch intercept.
        {"k64_n8",    1, 64,  8},
        {"k64_n32",   1, 64,  32},
        {"k64_n128",  1, 64,  128},
        {"k64_n512",  1, 64,  512},
        // Nt-sweep at Kt=192 (single-buffered cb_b, no prefetch overlap).
        {"k192_n8",   1, 192, 8},
        {"k192_n32",  1, 192, 32},
        {"k192_n128", 1, 192, 128},
    };
    const std::vector<Shape> prefill_shapes = {
        // Mt-sweep: prefill grid matmuls use Mt = seq_tiles (kSt). The
        // reader re-reads B once per mt, so B DRAM traffic grows ~Mt×.
        {"k64n32_m1",  1,  64,  32},
        {"k64n32_m4",  4,  64,  32},
        {"k64n32_m8",  8,  64,  32},
        {"k64n32_m16", 16, 64,  32},
        {"k192n16_m1", 1,  192, 16},
        {"k192n16_m4", 4,  192, 16},
        {"k192n16_m8", 8,  192, 16},
    };
    const std::vector<Shape>& shapes = prefill ? prefill_shapes : decode_shapes;
    const int iters = 50;

    std::printf("%-12s Mt   Kt   Nt   T_full(us)  T_noBread(us)  read_frac\n", "shape");
    std::printf("----------------------------------------------------------------\n");
    for (const auto& s : shapes) {
        double tf  = bench(pcie, "",      s, iters);
        double tn  = bench(pcie, noBread, s, iters);
        double frac = (tf > 0) ? (tf - tn) / tf * 100.0 : 0.0;
        std::printf("%-12s %3u %4u %4u  %9.2f    %9.2f      %5.1f%%\n",
                    s.tag, s.Mt, s.Kt, s.Nt, tf, tn, frac);
        std::fflush(stdout);
    }
    return 0;
}
