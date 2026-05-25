// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include "device.hpp"
#include "device_profile.hpp"
#include "kernel.hpp"
#include "fast_dispatch.hpp"
#include "profiling.hpp"
// kMaxRtaWords defined in kernel.hpp

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

// tt-metal headers
#include "llrt/hal.hpp"
#include "llrt_local/tt_memory.h"

// HAL-generated dev_msgs (new StructBuffer/View API)
#include "hal/generated/dev_msgs.hpp"

// UMD direct API (Phase B1): bypass tt::Cluster for all dispatch I/O.
#include <umd/device/cluster.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

// NUM_CIRCULAR_BUFFERS = 64 on Blackhole; must match circular_buffer_constants.h
#include "tt-metalium/circular_buffer_constants.h"

namespace tt::foil {

namespace {

// Forward decl — defined below.
tt::umd::CoreCoord kernel_translated_coord(const Kernel& kernel);

// ---------------------------------------------------------------------------
// Dispatch-stage tracing (TT_FOIL_DISPATCH_TRACE=1)
// ---------------------------------------------------------------------------
// Aggregates per-stage wall ns across all dispatch_execute_multi calls,
// separately for single-kernel (FD or slow) and multi-kernel (batch) paths.
// Prints a summary at process exit. Cheap when disabled (one TLS load per
// call + a single branch). When enabled, ~30 ns per timestamp via
// steady_clock — negligible against ~µs PCIe writes.
struct DispatchTrace {
    std::atomic<uint64_t> single_count{0};
    std::atomic<uint64_t> single_reset_ns{0};
    std::atomic<uint64_t> single_setup_ns{0};
    std::atomic<uint64_t> single_fence_ns{0};
    std::atomic<uint64_t> single_fire_wait_ns{0};   // FD push_launch+notify+wait OR slow fire+poll
    std::atomic<uint64_t> multi_count{0};
    std::atomic<uint64_t> multi_workers{0};
    std::atomic<uint64_t> multi_reset_ns{0};
    std::atomic<uint64_t> multi_setup_ns{0};
    std::atomic<uint64_t> multi_fence_ns{0};
    std::atomic<uint64_t> multi_fire_wait_ns{0};
    bool enabled{false};

    DispatchTrace() {
        const char* env = std::getenv("TT_FOIL_DISPATCH_TRACE");
        enabled = env && env[0] && env[0] != '0';
    }
    ~DispatchTrace() {
        if (!enabled) return;
        std::fprintf(stderr,
            "\n=== dispatch trace (TT_FOIL_DISPATCH_TRACE) ===\n");
        uint64_t sc = single_count.load();
        if (sc) {
            auto avg = [sc](uint64_t v) { return v / sc / 1000.0; };  // us/call
            std::fprintf(stderr,
                "  single-kernel  count=%lu   avg per call (us):\n"
                "    reset      %7.2f\n"
                "    setup      %7.2f\n"
                "    fence      %7.2f\n"
                "    fire+wait  %7.2f   <- includes worker device exec\n"
                "    total      %7.2f\n",
                sc,
                avg(single_reset_ns), avg(single_setup_ns),
                avg(single_fence_ns), avg(single_fire_wait_ns),
                avg(single_reset_ns + single_setup_ns + single_fence_ns + single_fire_wait_ns));
        }
        uint64_t mc = multi_count.load();
        if (mc) {
            auto avg = [mc](uint64_t v) { return v / mc / 1000.0; };
            std::fprintf(stderr,
                "  multi-kernel   count=%lu  avg workers=%.2f   avg per call (us):\n"
                "    reset      %7.2f\n"
                "    setup      %7.2f\n"
                "    fence      %7.2f\n"
                "    fire+wait  %7.2f\n"
                "    total      %7.2f\n",
                mc, double(multi_workers.load()) / mc,
                avg(multi_reset_ns), avg(multi_setup_ns),
                avg(multi_fence_ns), avg(multi_fire_wait_ns),
                avg(multi_reset_ns + multi_setup_ns + multi_fence_ns + multi_fire_wait_ns));
        }
    }
};
DispatchTrace& dispatch_trace() {
    static DispatchTrace t;
    return t;
}
inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Scope l1_membar to just the worker cores participating in this dispatch.
// UMD's `l1_membar(chip, {})` (empty set) issues a host-to-device barrier
// against every Tensix + ETH + DRAM core on the chip (~150 cores on
// Blackhole), which takes ~800 µs. The launch_msg/GO_MSG writes only target
// the worker's own L1, so a 1-core barrier is sufficient and ~50× cheaper.
// See docs/perf_post_pathB_analysis.md for the measurement (wall 9.59 →
// 4.56 s, -53%) and tokens-bit-identical verification.
void scoped_l1_membar(tt::umd::Cluster& driver, uint32_t chip,
                      std::span<Kernel* const> kernels) {
    std::unordered_set<tt::umd::CoreCoord> cores;
    cores.reserve(kernels.size());
    for (Kernel* k : kernels) cores.insert(kernel_translated_coord(*k));
    driver.l1_membar(chip, cores);
}
void scoped_l1_membar_one(tt::umd::Cluster& driver, uint32_t chip, Kernel& k) {
    std::unordered_set<tt::umd::CoreCoord> cores{kernel_translated_coord(k)};
    driver.l1_membar(chip, cores);
}

// One-kernel "stage" helpers shared by single- and multi-kernel paths.
// All three operate on the same translated coord computed once.

tt::umd::CoreCoord kernel_translated_coord(const Kernel& kernel) {
    return tt::umd::CoreCoord{
        kernel.virt_x, kernel.virt_y,
        tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
}

// Stage 1: write kernel ELF + RTA + launch_msg to L1.
// No memory fence here — the caller serialises that against the GO write.
void dispatch_stage_setup(
    Device& dev,
    Kernel& kernel,
    const tt::tt_metal::Hal& hal,
    tt::umd::Cluster& driver,
    uint32_t chip) {
    TF_ZONE_N("TF_dispatch/setup");
    auto cc = kernel_translated_coord(kernel);

    const auto& dev_msgs_factory =
        hal.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::TENSIX);

    // ---- Kernel ELF binaries -> kernel_text_addr in KERNEL_CONFIG region ----
    // Skip the NOC write if the same Kernel was last dispatched to this
    // core — the L1 still holds its text. Saves ~3ms per dispatch on
    // small ops where the ELF transfer dominated. Invalidated whenever
    // release_kernels() runs (which rewinds the KERNEL_CONFIG arena, so
    // the next load_kernel may reuse this Kernel's L1 region for a
    // different binary).
    const uint64_t core_key = Device::core_key(kernel.core.x, kernel.core.y);
    auto& resident = dev.resident_kernels[core_key];
    if (!resident.contains(&kernel)) {
        for (const auto& lr : kernel.riscs) {
            lr.mem->process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr,
                                      uint64_t /*addr*/, uint32_t len_words) {
                driver.write_to_device(
                    &*mem_ptr,
                    static_cast<std::size_t>(len_words) * sizeof(uint32_t),
                    chip, cc, lr.kernel_text_addr);
            });
        }
        resident.insert(&kernel);
    }

    // ---- Runtime args -> per-RISC RTA slot ----
    for (const auto& lr : kernel.riscs) {
        if (lr.runtime_args.empty()) continue;
        uint64_t rta_addr = kernel.rta_base_addr +
            lr.processor_index * static_cast<uint64_t>(kMaxRtaWords * sizeof(uint32_t));
        driver.write_to_device(
            lr.runtime_args.data(),
            lr.runtime_args.size() * sizeof(uint32_t),
            chip, cc, rta_addr);
    }

    // ---- launch_msg ----
    auto launch_msg_buf = dev_msgs_factory.create<tt_metal::dev_msgs::launch_msg_t>();
    {
        auto kernel_config = launch_msg_buf.view().kernel_config();
        kernel_config.mode() = tt_metal::dev_msgs::DISPATCH_MODE_HOST;

        for (const auto& lr : kernel.riscs) {
            kernel_config.enables() |= (1u << lr.processor_index);
        }

        constexpr uint32_t kTensixIdx = 0;
        uint32_t kcfg_base = static_cast<uint32_t>(hal.get_dev_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::KERNEL_CONFIG));
        kernel_config.kernel_config_base()[kTensixIdx] = kcfg_base;

        for (const auto& lr : kernel.riscs) {
            uint32_t abs_rta_addr = static_cast<uint32_t>(kernel.rta_base_addr) +
                lr.processor_index * kMaxRtaWords * sizeof(uint32_t);
            uint32_t rel_rta = abs_rta_addr - kcfg_base;
            auto rta_entry = kernel_config.rta_offset()[lr.processor_index];
            rta_entry.rta_offset()  = static_cast<uint16_t>(rel_rta);
            rta_entry.crta_offset() = 0;
        }

        for (const auto& lr : kernel.riscs) {
            uint32_t text_off = static_cast<uint32_t>(lr.kernel_text_addr) - kcfg_base;
            kernel_config.kernel_text_offset()[lr.processor_index] = text_off;
        }

        // Skip remote CB setup; see notes in earlier commit (firmware would
        // otherwise misread the RTA region as CB config).
        kernel_config.min_remote_cb_start_index() =
            static_cast<uint8_t>(NUM_CIRCULAR_BUFFERS);

        // Local CB blob, populated by register_cbs() (v4-3/v4-4). When no
        // CBs were registered we leave the fields at their defaults so
        // firmware's mask-driven loop is a no-op (mask == 0).
        if (kernel.cb_alloc.valid) {
            kernel_config.local_cb_offset() =
                static_cast<uint16_t>(kernel.cb_alloc.local_cb_offset);
            kernel_config.local_cb_mask() = kernel.cb_alloc.local_cb_mask;
        }
    }

    uint64_t launch_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::LAUNCH);
    driver.write_to_device_reg(
        launch_msg_buf.data(), static_cast<uint32_t>(launch_msg_buf.size()),
        chip, cc, launch_addr);
}

// Slow-dispatch reset: send RUN_MSG_RESET_READ_PTR_FROM_HOST before each
// program launch so firmware resets its launch_msg_rd_ptr to 0 and the
// go_message_index to 0. Without this, firmware enters the kernel-launch
// path but its CB-setup writes to BRISC's local cb_interface[] are
// somehow not observable to the kernel that runs immediately after —
// possibly because firmware caches some state from the previous launch
// (or the post-cold-boot uninitialized state) that the RESET path
// invalidates. Matches tt-metal's send_reset_go_signal flow in
// llrt/llrt.cpp.
void dispatch_stage_send_reset(
    Kernel& kernel,
    const tt::tt_metal::Hal& hal,
    tt::umd::Cluster& driver,
    uint32_t chip) {
    TF_ZONE_N("TF_dispatch/send_reset");
    auto cc = kernel_translated_coord(kernel);
    uint64_t go_entry_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);
    uint32_t reset_val = hal.make_go_msg_u32(
        static_cast<uint8_t>(tt_metal::dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST),
        0, 0, 0);
    driver.write_to_device_reg(&reset_val, sizeof(reset_val), chip, cc, go_entry_addr);
    scoped_l1_membar_one(driver, chip, kernel);
    uint64_t go_idx_addr = hal.get_dev_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG_INDEX);
    uint32_t zero = 0;
    driver.write_to_device_reg(&zero, sizeof(zero), chip, cc, go_idx_addr);
}

// Stage 2: fire RUN_MSG_GO on this kernel's core.
void dispatch_stage_fire_go(
    Kernel& kernel,
    const tt::tt_metal::Hal& hal,
    tt::umd::Cluster& driver,
    uint32_t chip) {
    TF_ZONE_N("TF_dispatch/fire_go");
    auto cc = kernel_translated_coord(kernel);
    uint64_t go_entry_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);
    uint32_t go_val = hal.make_go_msg_u32(
        static_cast<uint8_t>(tt_metal::dev_msgs::RUN_MSG_GO), 0, 0, 0);
    driver.write_to_device_reg(&go_val, sizeof(go_val), chip, cc, go_entry_addr);
}

// Stage 3: poll this kernel's GO_MSG until RUN_MSG_DONE or timeout.
// Returns the elapsed milliseconds spent here so callers can apply a
// shared budget across multiple kernels.
int64_t dispatch_stage_wait_done(
    Kernel& kernel,
    const tt::tt_metal::Hal& hal,
    tt::umd::Cluster& driver,
    uint32_t chip,
    int timeout_ms) {
    TF_ZONE_N("TF_dispatch/wait_done");
    auto cc = kernel_translated_coord(kernel);
    const auto& dev_msgs_factory =
        hal.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::TENSIX);
    uint64_t go_entry_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);
    auto go_msg_buf = dev_msgs_factory.create<tt_metal::dev_msgs::go_msg_t>();
    auto start = std::chrono::steady_clock::now();
    uint32_t poll_count = 0;
    while (true) {
        ++poll_count;
        driver.read_from_device(
            go_msg_buf.data(), chip, cc, go_entry_addr,
            static_cast<std::size_t>(go_msg_buf.size()));
        if (go_msg_buf.view().signal() == tt_metal::dev_msgs::RUN_MSG_DONE) {
#if defined(TRACY_ENABLE)
            // Attach poll count to the zone so the per-call CSV / aggregate
            // can tell PCIe-bound dispatches (polls==1) from chip-bound ones
            // (polls >> 1, sleep 100us per iteration).
            char buf[32];
            int n = std::snprintf(buf, sizeof(buf), "polls=%u", poll_count);
            if (n > 0) TF_ZONE_TEXT(buf, static_cast<std::size_t>(n));
#endif
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (timeout_ms > 0 && elapsed > timeout_ms) {
            throw std::runtime_error(
                "tt-foil: timeout waiting for kernel completion on core ("
                + std::to_string(kernel.core.x) + ","
                + std::to_string(kernel.core.y) + ")");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// dispatch_execute — blocking slow-dispatch, single kernel
// ---------------------------------------------------------------------------
void dispatch_execute(Device& dev, Kernel& kernel, int timeout_ms) {
    TF_ZONE_N("TF_dispatch_execute");
    TF_ZONE_TEXT(kernel.name.c_str(), kernel.name.size());
    Kernel* one = &kernel;
    dispatch_execute_multi(dev, std::span<Kernel* const>(&one, 1), timeout_ms);
}

// ---------------------------------------------------------------------------
// dispatch_launch_async — fire-and-forget, R5 G1
// ---------------------------------------------------------------------------
void dispatch_launch_async(Device& dev, Kernel& kernel) {
    TF_ZONE_N("TF_dispatch_launch_async");
    TF_ZONE_TEXT(kernel.name.c_str(), kernel.name.size());
    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::umd::Cluster& driver     = *dev.umd_driver;
    const uint32_t chip          = dev.chip_id;

    dispatch_stage_send_reset(kernel, hal, driver, chip);
    scoped_l1_membar_one(driver, chip, kernel);

    dispatch_stage_setup(dev, kernel, hal, driver, chip);

    tt_driver_atomics::sfence();
    scoped_l1_membar_one(driver, chip, kernel);

    dispatch_stage_fire_go(kernel, hal, driver, chip);

    scoped_l1_membar_one(driver, chip, kernel);
    // No wait_done — caller manages the kernel's lifecycle.
}

void dispatch_execute_multi(
    Device& dev,
    std::span<Kernel* const> kernels,
    int timeout_ms) {
    TF_ZONE_N("TF_dispatch_execute_multi");

    if (kernels.empty()) {
        throw std::runtime_error("tt-foil: dispatch_execute_multi: no kernels");
    }

    // Build a short context string from kernel names for the Tracy zone.
    // Single kernel → "name"; multi-kernel → "name1+name2[+...]" capped at
    // the first 3 to keep zone text small.
#if defined(TRACY_ENABLE)
    {
        std::string ctx;
        const size_t kCap = 3;
        for (size_t i = 0; i < kernels.size() && i < kCap; ++i) {
            if (!ctx.empty()) ctx += '+';
            ctx += kernels[i]->name;
        }
        if (kernels.size() > kCap) ctx += "+...";
        TF_ZONE_TEXT(ctx.c_str(), ctx.size());
    }
#endif

    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::umd::Cluster& driver     = *dev.umd_driver;
    const uint32_t chip          = dev.chip_id;

    auto& trace = dispatch_trace();
    const bool trace_on = trace.enabled;
    const bool is_single = (kernels.size() == 1);
    uint64_t t0 = trace_on ? now_ns() : 0;

    // Stage 0: send RUN_MSG_RESET_READ_PTR_FROM_HOST to each core's
    // GO_MSG, then zero GO_MSG_INDEX. Mirrors tt-metal's slow-dispatch
    // send_reset_go_signal (llrt.cpp:115). Without this BRISC firmware's
    // CB-setup writes don't become visible to the kernel that runs
    // right after — see the long comment in test_tile_copy.cpp for the
    // diagnostic trail.
    for (Kernel* k : kernels) dispatch_stage_send_reset(*k, hal, driver, chip);
    scoped_l1_membar(driver, chip, kernels);

    uint64_t t1 = trace_on ? now_ns() : 0;

    // Stage 1: write ELF + RTA + launch_msg for every kernel.
    for (Kernel* k : kernels) dispatch_stage_setup(dev, *k, hal, driver, chip);

    uint64_t t2 = trace_on ? now_ns() : 0;

    // Host memory fence + device L1 barrier so all setup writes (kernel
    // ELF, RTA, launch_msg, and the CB blob from register_cbs) have
    // landed on the chip before firmware sees the GO.
    tt_driver_atomics::sfence();
    scoped_l1_membar(driver, chip, kernels);

    uint64_t t3 = trace_on ? now_ns() : 0;

    // R5 G2a: if a FastDispatch is attached AND we have a single-kernel
    // launch, route the fire+wait through the on-chip dispatcher.
    if (dev.fast_dispatch != nullptr &&
        kernels.size() <= fast_dispatch_layout::kMaxBatchWorkers) {
        // R5 G2a/G2b: route fire+wait through the on-chip dispatcher.
        // Host has already done send_reset + setup via PCIe above, so we
        // just need to fire all GO_MSGs and wait for all DONE. The batch
        // cmd fires all GOs first (NOC-parallel) then polls all DONEs, so
        // multi-core matmul preserves parallelism across workers.
        const uint32_t go_msg_addr_l1 = static_cast<uint32_t>(hal.get_dev_noc_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::GO_MSG));
        if (kernels.size() == 1) {
            const uint32_t launch_addr_l1 = static_cast<uint32_t>(hal.get_dev_noc_addr(
                tt_metal::HalProgrammableCoreType::TENSIX,
                tt_metal::HalL1MemAddrType::LAUNCH));
            dev.fast_dispatch->push_launch(
                kernels[0]->core, launch_addr_l1, go_msg_addr_l1, nullptr, 0);
        } else {
            std::vector<CoreCoord> worker_cores;
            worker_cores.reserve(kernels.size());
            for (Kernel* k : kernels) worker_cores.push_back(k->core);
            dev.fast_dispatch->push_launch_batch(worker_cores, go_msg_addr_l1);
        }
        dev.fast_dispatch->push_notify();
        dev.fast_dispatch->wait_for_completion(
            dev.fast_dispatch->expected_completion, timeout_ms);
        if (trace_on) {
            uint64_t t4 = now_ns();
            if (is_single) {
                trace.single_count.fetch_add(1);
                trace.single_reset_ns.fetch_add(t1 - t0);
                trace.single_setup_ns.fetch_add(t2 - t1);
                trace.single_fence_ns.fetch_add(t3 - t2);
                trace.single_fire_wait_ns.fetch_add(t4 - t3);
            } else {
                trace.multi_count.fetch_add(1);
                trace.multi_workers.fetch_add(kernels.size());
                trace.multi_reset_ns.fetch_add(t1 - t0);
                trace.multi_setup_ns.fetch_add(t2 - t1);
                trace.multi_fence_ns.fetch_add(t3 - t2);
                trace.multi_fire_wait_ns.fetch_add(t4 - t3);
            }
        }
        capture_device_profile(dev, kernels);
        TF_FRAME_MARK();
        return;
    }

    // Slow-dispatch path: stage 2 fire all GOs, then stage 3 poll all DONE.
    for (Kernel* k : kernels) dispatch_stage_fire_go(*k, hal, driver, chip);

    scoped_l1_membar(driver, chip, kernels);

    auto overall_start = std::chrono::steady_clock::now();
    for (Kernel* k : kernels) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - overall_start).count();
        int remaining = timeout_ms > 0
            ? std::max<int>(1, timeout_ms - static_cast<int>(elapsed))
            : 0;
        dispatch_stage_wait_done(*k, hal, driver, chip, remaining);
    }

    if (trace_on) {
        uint64_t t4 = now_ns();
        if (is_single) {
            trace.single_count.fetch_add(1);
            trace.single_reset_ns.fetch_add(t1 - t0);
            trace.single_setup_ns.fetch_add(t2 - t1);
            trace.single_fence_ns.fetch_add(t3 - t2);
            trace.single_fire_wait_ns.fetch_add(t4 - t3);
        } else {
            trace.multi_count.fetch_add(1);
            trace.multi_workers.fetch_add(kernels.size());
            trace.multi_reset_ns.fetch_add(t1 - t0);
            trace.multi_setup_ns.fetch_add(t2 - t1);
            trace.multi_fence_ns.fetch_add(t3 - t2);
            trace.multi_fire_wait_ns.fetch_add(t4 - t3);
        }
    }
    // Device profiler read-back: pull the per-RISC cycle markers each
    // kernel's core wrote into its mailbox-resident profiler buffer.
    // No-op when TT_FOIL_DEVICE_PROFILER_ENABLED is unset at compile time.
    capture_device_profile(dev, kernels);
    TF_FRAME_MARK();
}


}  // namespace tt::foil
