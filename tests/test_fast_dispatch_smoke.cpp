// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// R5 Gate-1 fast-dispatch smoke test.
//
// Stage 1: NOTIFY-only — dispatcher on (1,0), host pushes N CMD_NOTIFY,
//          polls completion_counter. Confirms cmd-ring + dispatcher main
//          loop + completion signaling.
//
// Stage 2: LAUNCH — pre-stage no-op kernel on worker (0,0) via slow-
//          dispatch (writes ELF + launch_msg to slot 0). Then loop:
//             push_launch(worker=(0,0), lm_size=0)  // skip launch_msg write
//             push_notify()                          // bump completion
//             wait_for_completion
//          Each iter drives one full GO → noop-runs → DONE-poll cycle
//          through the on-chip dispatcher.
//
// Gate-1 criterion: per-launch wall < 50 µs (slow-dispatch baseline = 3.18 ms).
//
// Usage:
//   TT_FOIL_FIRMWARE_DIR=<build>/firmware \
//   TT_FOIL_KERNEL_DIR=ops/noop/prebuilt TT_FOIL_OPS_DIR=ops \
//   TT_FOIL_DEVICE=0 ./test_fast_dispatch_smoke [iters]
//   (default iters = 1000)

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
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

int main(int argc, char** argv) try {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 1000;
    if (iters < 1) throw std::runtime_error("iters must be >= 1");

    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {1, 0}});
    tt::foil::CoreCoord worker_core{0, 0};
    tt::foil::CoreCoord dispatcher_core{1, 0};

    // Pre-stage no-op kernel on worker (0,0): writes ELF + launch_msg to
    // slot 0 + runs once. After this, GO_MSG.signal = DONE and launch_msg
    // stays valid (HOST mode firmware doesn't advance launch_msg_rd_ptr).
    {
        std::string noop_elf = kernel_dir + "/noop.brisc.elf";
        std::array<tt::foil::RiscBinary, 1> bins = {{
            {tt::foil::RiscBinary::RiscId::BRISC, noop_elf},
        }};
        auto worker_kernel = tt::foil::load_kernel(*dev, bins, worker_core);
        tt::foil::execute(*dev, *worker_kernel);
    }

    tt::foil::FastDispatch fd(*dev, dispatcher_core);
    fd.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Stage 1: NOTIFY-only.
    {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        for (int i = 0; i < 8; ++i) fd.push_notify();
        const double wait_us = fd.wait_for_completion(fd.expected_completion, 2000);
        const auto t1 = Clock::now();
        std::printf("=== stage 1 NOTIFY × 8 ===\n");
        std::printf("  wait %.1f us, total %.1f us\n", wait_us,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    // Stage 2: LAUNCH gate-1 measurement.
    const uint32_t kWorkerLaunchAddr = 0x70;   // Blackhole HAL LAUNCH mailbox
    const uint32_t kWorkerGoMsgAddr  = 0x3F0;  // Blackhole HAL GO_MSG

    // One untimed warm-up.
    fd.push_launch(worker_core, kWorkerLaunchAddr, kWorkerGoMsgAddr, nullptr, 0);
    fd.push_notify();
    fd.wait_for_completion(fd.expected_completion, 5000);

    using Clock = std::chrono::steady_clock;
    std::vector<double> us;
    us.reserve(iters);
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const auto a = Clock::now();
        fd.push_launch(worker_core, kWorkerLaunchAddr, kWorkerGoMsgAddr, nullptr, 0);
        fd.push_notify();
        fd.wait_for_completion(fd.expected_completion, 5000);
        us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }
    const auto t1 = Clock::now();

    auto pct = [&](double q) {
        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        return s[std::min<size_t>(s.size() - 1, q * s.size())];
    };
    double sum = 0, mn = us[0], mx = us[0];
    for (double v : us) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    std::printf("=== stage 2 LAUNCH × %d ===\n", iters);
    std::printf("  total %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    std::printf("  per-launch us: min=%.1f p50=%.1f mean=%.1f p90=%.1f p99=%.1f max=%.1f\n",
                mn, pct(0.50), sum / us.size(), pct(0.90), pct(0.99), mx);

    if (pct(0.50) >= 50.0) {
        std::fprintf(stderr, "GATE-1 FAIL: p50 %.1f us >= 50 us\n", pct(0.50));
        return 1;
    }

    fd.push_terminate();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_fast_dispatch_smoke: FAIL — %s\n", e.what());
    return 1;
}
