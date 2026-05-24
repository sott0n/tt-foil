// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// G0 floor-measurement test. Loads a kernel with empty `kernel_main()` and
// dispatches it N times via slow-dispatch, measuring per-launch wall time.
// Purpose: confirm whether the per-op ~3.3 ms floor observed in iter21 is
// actually firmware-side dispatch overhead (host PCIe sync latency to
// worker BRISC's go_msg poll) vs user-kernel exec time.
//
// Interpretation guide (recorded in plan):
//   p50 ≈ 3 ms       → R5 hypothesis confirmed; on-chip dispatcher should
//                       collapse the floor to ~µs.
//   p50 < 1 ms       → floor is partly user-kernel; R5 expected gain
//                       smaller than predicted.
//   p50 indeterminate→ revise measurement methodology.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=ops/noop/prebuilt ./test_noop_dispatch_loop [iters]
//   default iters = 1000

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

static std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

int main(int argc, char** argv) try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 1000;
    if (iters < 1) throw std::runtime_error("iters must be >= 1");

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;
    auto dev = tt::foil::open_device(pcie_index);
    std::puts("test_noop_dispatch_loop: device opened");

    tt::foil::CoreCoord core{0, 0};

    std::string brisc_elf = kernel_dir + "/noop.brisc.elf";
    std::array<tt::foil::RiscBinary, 1> binaries = {{
        {tt::foil::RiscBinary::RiscId::BRISC, brisc_elf},
    }};
    auto kernel = tt::foil::load_kernel(*dev, binaries, core);

    // No RTAs, no CBs. Pin the kernel so its ELF stays cached in resident_kernels
    // across iterations (we want to measure dispatch overhead, not ELF transfer).
    tt::foil::pin_persistent(*dev, *kernel, core);
    std::puts("test_noop_dispatch_loop: kernel pinned");

    // Warm-up: one launch to ensure firmware-side state is hot.
    tt::foil::execute(*dev, *kernel);

    using Clock = std::chrono::steady_clock;
    std::vector<double> us;
    us.reserve(iters);

    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const auto a = Clock::now();
        tt::foil::execute(*dev, *kernel);
        const auto b = Clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    const auto t1 = Clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto pct = [&](double q) {
        std::vector<double> sorted = us;
        std::sort(sorted.begin(), sorted.end());
        const size_t idx = std::min<size_t>(
            sorted.size() - 1,
            static_cast<size_t>(q * sorted.size()));
        return sorted[idx];
    };
    const double p50 = pct(0.50);
    const double p90 = pct(0.90);
    const double p99 = pct(0.99);
    double sum = 0; for (double v : us) sum += v;
    const double mean = sum / us.size();
    double mn = us[0], mx = us[0];
    for (double v : us) { mn = std::min(mn, v); mx = std::max(mx, v); }

    std::printf("\n=== G0 noop-dispatch floor (iters=%d) ===\n", iters);
    std::printf("  total_wall_ms %.2f\n", total_ms);
    std::printf("  per-launch us:  min=%.1f  p50=%.1f  mean=%.1f  p90=%.1f  p99=%.1f  max=%.1f\n",
                mn, p50, mean, p90, p99, mx);

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_noop_dispatch_loop: FAIL — %s\n", e.what());
    return 1;
}
