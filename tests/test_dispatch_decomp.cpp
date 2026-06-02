// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefetch-hypothesis decomposition. Question under test:
//
//   "tt-foil's dispatcher is single-core. If a command's DONE-poll (worker
//    exec) is long, does a prefetch stage help?"
//
// Prefetch can only remove the *command-fetch* cost from the dispatcher's
// critical path. In tt-foil's push model that cost == the host's PCIe
// push_launch (write cmd slot + bump wptr). So the decisive numbers are:
//
//   (A) host push_launch latency, measured in isolation
//          → the absolute per-op ceiling a prefetcher / ring look-ahead
//            could ever hide.
//   (R) dispatcher round-trip for a no-op worker (C≈0): GO issue +
//          DONE poll + completion readback.
//   (C) real-op worker exec — taken from perf history (matmul 0.3–1.1 ms),
//          and bounded here by the no-op floor (R, with C≈0).
//
// We also measure a *pipelined* mode: push N launches into the 32-slot
// ring before waiting once. If per-op there ≈ serial per-op − (A), the
// ring already overlaps host-push with dispatcher work — i.e. the
// "prefetch" benefit is achievable with zero extra cores.
//
// Usage:
//   TT_FOIL_FIRMWARE_DIR=<build>/firmware \
//   TT_FOIL_KERNEL_DIR=ops/noop/prebuilt TT_FOIL_OPS_DIR=ops \
//   TT_FOIL_DEVICE=0 ./test_dispatch_decomp [iters]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "fast_dispatch.hpp"

static std::string required_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) throw std::runtime_error(std::string("Missing env var: ") + name);
    return v;
}

static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(q * v.size()))];
}
static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

int main(int argc, char** argv) try {
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 2000;
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    int pcie_index = 0;
    if (const char* e = std::getenv("TT_FOIL_DEVICE")) pcie_index = std::stoi(e);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {1, 0}});
    tt::foil::CoreCoord worker_core{0, 0};
    tt::foil::CoreCoord dispatcher_core{1, 0};

    // Pre-stage no-op on worker (0,0): ELF + launch_msg + one run, leaving
    // GO_MSG = DONE and launch_msg valid. After this the dispatcher only
    // needs to fire GO (lm_size=0) and poll DONE.
    {
        std::string noop_elf = kernel_dir + "/noop.brisc.elf";
        std::array<tt::foil::RiscBinary, 1> bins = {{
            {tt::foil::RiscBinary::RiscId::BRISC, noop_elf},
        }};
        auto wk = tt::foil::load_kernel(*dev, bins, worker_core);
        tt::foil::execute(*dev, *wk);
    }

    tt::foil::FastDispatch fd(*dev, dispatcher_core);
    fd.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    const uint32_t kLaunchAddr = 0x70;   // Blackhole HAL LAUNCH mailbox
    const uint32_t kGoMsgAddr  = 0x3F0;  // Blackhole HAL GO_MSG

    using Clock = std::chrono::steady_clock;
    auto now = [] { return Clock::now(); };
    auto us  = [](auto d) { return std::chrono::duration<double, std::micro>(d).count(); };

    // Warm-up.
    fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);
    fd.push_notify();
    fd.wait_for_completion(fd.expected_completion, 5000);

    // ---- (A) isolated push_launch + (R) serial round-trip --------------
    std::vector<double> a_us, serial_us;
    a_us.reserve(iters); serial_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = now();
        fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);  // (A)
        const auto t1 = now();
        fd.push_notify();
        fd.wait_for_completion(fd.expected_completion, 5000);              // (R) + 2nd push
        const auto t2 = now();
        a_us.push_back(us(t1 - t0));
        serial_us.push_back(us(t2 - t0));
    }

    // ---- pipelined: fill the ring, then wait once ----------------------
    // N launches enqueued back-to-back (host runs ahead), 1 notify, 1 wait.
    // per-op = total / N. If this ≈ serial − (A), the 32-slot ring already
    // overlaps host-push with dispatcher work without a 2nd core.
    const int kBatch = 16;  // <= 32 ring slots, leave headroom for notify
    const int batches = std::max(1, iters / kBatch);
    std::vector<double> pipe_perop_us;
    pipe_perop_us.reserve(batches);
    for (int b = 0; b < batches; ++b) {
        const auto t0 = now();
        for (int i = 0; i < kBatch; ++i)
            fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);
        fd.push_notify();
        fd.wait_for_completion(fd.expected_completion, 10000);
        const auto t1 = now();
        pipe_perop_us.push_back(us(t1 - t0) / kBatch);
    }

    std::printf("\n=== fast-dispatch per-op decomposition (noop worker, C≈0) ===\n");
    std::printf("  (A) push_launch (PCIe write x2):  p50=%.1f  mean=%.1f  p90=%.1f us\n",
                pct(a_us, 0.5), mean(a_us), pct(a_us, 0.9));
    std::printf("  (R) serial round-trip per op:     p50=%.1f  mean=%.1f  p90=%.1f us\n",
                pct(serial_us, 0.5), mean(serial_us), pct(serial_us, 0.9));
    std::printf("  pipelined per-op (ring look-ahead, N=%d): p50=%.1f mean=%.1f us\n",
                kBatch, pct(pipe_perop_us, 0.5), mean(pipe_perop_us));
    std::printf("\n  Interpretation:\n");
    std::printf("    prefetch's per-op ceiling = (A) = %.1f us\n", pct(a_us, 0.5));
    std::printf("    real-op worker exec (C, perf history) = 300-1100 us\n");
    std::printf("    => prefetch upper-bound gain on a real op = (A)/(A+C) "
                "~= %.1f%%-%.1f%%\n",
                100.0 * pct(a_us, 0.5) / (pct(a_us, 0.5) + 1100.0),
                100.0 * pct(a_us, 0.5) / (pct(a_us, 0.5) + 300.0));

    fd.push_terminate();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_dispatch_decomp: FAIL — %s\n", e.what());
    return 1;
}
