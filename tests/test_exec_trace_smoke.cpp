// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// EXEC_TRACE (trace replay) smoke test. Proves the on-chip dispatcher can
// replay a command sub-stream recorded into DRAM, with ~zero host per-op cost.
//
// Setup: pre-stage a no-op kernel on worker (0,0) via slow-dispatch (leaves
// launch_msg valid + GO_MSG=DONE), start FastDispatch on (1,0).
//
// Baseline (serial): N × (push_launch + push_notify + wait_for_completion).
//   Host pays a PCIe round-trip per op.
// Trace: begin_record(); N × push_launch; end_record() → DRAM trace; then
//   exec_trace + push_notify + wait_for_completion once.
//   Host pays one round-trip total; the dispatcher fires all N GOs on-chip.
//
// Pass criteria: no completion timeout (proves the dispatcher read N cmds from
// DRAM and fired N GO/DONE cycles), AND trace per-op < serial per-op (the win).
//
// Usage:
//   TT_FOIL_KERNEL_DIR=ops/noop/prebuilt TT_FOIL_OPS_DIR=ops \
//   TT_FOIL_DEVICE=0 ./test_exec_trace_smoke [N]

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

#include "tt_foil/runtime.hpp"
#include "fast_dispatch.hpp"

static std::string required_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) throw std::runtime_error(std::string("Missing env var: ") + name);
    return v;
}

int main(int argc, char** argv) try {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 256;
    if (N < 1) throw std::runtime_error("N must be >= 1");

    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    int pcie_index = 0;
    if (const char* e = std::getenv("TT_FOIL_DEVICE")) pcie_index = std::stoi(e);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {1, 0}});
    tt::foil::CoreCoord worker_core{0, 0};
    tt::foil::CoreCoord dispatcher_core{1, 0};

    // Pre-stage no-op on worker (0,0): ELF + launch_msg + one run.
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
    auto us = [](auto d) { return std::chrono::duration<double, std::micro>(d).count(); };

    // Warm-up.
    fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);
    fd.push_notify();
    fd.wait_for_completion(fd.expected_completion, 5000);

    // ---- Baseline: serial push per op ----------------------------------
    const auto s0 = Clock::now();
    for (int i = 0; i < N; ++i) {
        fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);
        fd.push_notify();
        fd.wait_for_completion(fd.expected_completion, 5000);
    }
    const double serial_total = us(Clock::now() - s0);
    const double serial_perop = serial_total / N;

    // ---- Trace: record N launches once, replay with one EXEC_TRACE ------
    fd.begin_record();
    for (int i = 0; i < N; ++i)
        fd.push_launch(worker_core, kLaunchAddr, kGoMsgAddr, nullptr, 0);
    auto handle = fd.end_record();
    if (handle.num_cmds != static_cast<uint32_t>(N)) {
        std::fprintf(stderr, "FAIL: recorded %u cmds, expected %d\n",
                     handle.num_cmds, N);
        return 1;
    }

    const auto t0 = Clock::now();
    fd.exec_trace(handle);
    fd.push_notify();
    fd.wait_for_completion(fd.expected_completion, 10000);
    const double trace_total = us(Clock::now() - t0);
    const double trace_perop = trace_total / N;

    std::printf("\n=== EXEC_TRACE smoke (N=%d, noop worker) ===\n", N);
    std::printf("  serial: total=%.1f us  per-op=%.2f us\n", serial_total, serial_perop);
    std::printf("  trace:  total=%.1f us  per-op=%.2f us\n", trace_total, trace_perop);
    std::printf("  speedup per-op: %.1fx\n", serial_perop / trace_perop);

    if (trace_perop >= serial_perop) {
        std::fprintf(stderr,
            "FAIL: trace per-op (%.2f) not below serial per-op (%.2f)\n",
            trace_perop, serial_perop);
        return 1;
    }

    fd.push_terminate();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tt::foil::close_device(std::move(dev));
    std::puts("test_exec_trace_smoke: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_exec_trace_smoke: FAIL — %s\n", e.what());
    return 1;
}
