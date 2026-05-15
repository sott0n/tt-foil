// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include "device.hpp"
#include "kernel.hpp"
// kMaxRtaWords defined in kernel.hpp

#include <chrono>
#include <stdexcept>
#include <thread>
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
    auto cc = kernel_translated_coord(kernel);

    const auto& dev_msgs_factory =
        hal.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::TENSIX);

    // ---- Kernel ELF binaries -> kernel_text_addr in KERNEL_CONFIG region ----
    for (const auto& lr : kernel.riscs) {
        lr.mem->process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr,
                                  uint64_t /*addr*/, uint32_t len_words) {
            driver.write_to_device(
                &*mem_ptr,
                static_cast<std::size_t>(len_words) * sizeof(uint32_t),
                chip, cc, lr.kernel_text_addr);
        });
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
    }

    uint64_t launch_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::LAUNCH);
    driver.write_to_device_reg(
        launch_msg_buf.data(), static_cast<uint32_t>(launch_msg_buf.size()),
        chip, cc, launch_addr);
}

// Stage 2: fire RUN_MSG_GO on this kernel's core.
void dispatch_stage_fire_go(
    Kernel& kernel,
    const tt::tt_metal::Hal& hal,
    tt::umd::Cluster& driver,
    uint32_t chip) {
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
    auto cc = kernel_translated_coord(kernel);
    const auto& dev_msgs_factory =
        hal.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::TENSIX);
    uint64_t go_entry_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);
    auto go_msg_buf = dev_msgs_factory.create<tt_metal::dev_msgs::go_msg_t>();
    auto start = std::chrono::steady_clock::now();
    while (true) {
        driver.read_from_device(
            go_msg_buf.data(), chip, cc, go_entry_addr,
            static_cast<std::size_t>(go_msg_buf.size()));
        if (go_msg_buf.view().signal() == tt_metal::dev_msgs::RUN_MSG_DONE) {
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
    Kernel* one = &kernel;
    dispatch_execute_multi(dev, std::span<Kernel* const>(&one, 1), timeout_ms);
}

void dispatch_execute_multi(
    Device& dev,
    std::span<Kernel* const> kernels,
    int timeout_ms) {

    if (kernels.empty()) {
        throw std::runtime_error("tt-foil: dispatch_execute_multi: no kernels");
    }

    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::umd::Cluster& driver     = *dev.umd_driver;
    const uint32_t chip          = dev.chip_id;

    // Stage 1: write ELF + RTA + launch_msg for every kernel.
    for (Kernel* k : kernels) dispatch_stage_setup(dev, *k, hal, driver, chip);

    // Memory fence so the launch_msgs are visible before any GO.
    tt_driver_atomics::sfence();

    // Stage 2: fire RUN_MSG_GO on every kernel. Issuing all GOs before any
    // DONE check is what makes producer/consumer kernels actually meet on
    // the device.
    for (Kernel* k : kernels) dispatch_stage_fire_go(*k, hal, driver, chip);

    driver.l1_membar(chip);

    // Stage 3: poll each kernel's GO_MSG. We poll sequentially; the
    // hardware runs them concurrently, so total wall time is
    // max(per-kernel run time), not the sum. Shrink the per-call timeout
    // budget as we go so the overall ceiling matches the caller's value.
    auto overall_start = std::chrono::steady_clock::now();
    for (Kernel* k : kernels) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - overall_start).count();
        int remaining = timeout_ms > 0
            ? std::max<int>(1, timeout_ms - static_cast<int>(elapsed))
            : 0;
        dispatch_stage_wait_done(*k, hal, driver, chip, remaining);
    }
}

// ---------------------------------------------------------------------------
// Legacy single-kernel implementation kept below as a fallback — but it's no
// longer called: dispatch_execute() now routes through dispatch_execute_multi.
// Leaving the body in would just bit-rot; remove it once we're confident.
// ---------------------------------------------------------------------------
#if 0
static void dispatch_execute_legacy(Device& dev, Kernel& kernel, int timeout_ms) {
    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::umd::Cluster& driver     = *dev.umd_driver;
    const uint32_t chip          = dev.chip_id;

    // Virtual (translated) core coordinate resolved at kernel_load() time.
    tt::umd::CoreCoord cc{
        kernel.virt_x, kernel.virt_y,
        tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};

    // Dev-msgs factory for TENSIX core type
    const auto& dev_msgs_factory =
        hal.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::TENSIX);

    // ---- Step 1: Write kernel ELF binaries to core L1 ----
    // Write the kernel binary data to kernel_text_addr (allocated in KERNEL_CONFIG region).
    // We ignore the span addresses (which are 0 after XIPify) and write to the pre-allocated slot.
    // This mirrors llrt::write_binary_to_address: target address is explicit, not from span.
    for (const auto& lr : kernel.riscs) {
        lr.mem->process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t /*addr*/, uint32_t len_words) {
            driver.write_to_device(
                &*mem_ptr,
                static_cast<std::size_t>(len_words) * sizeof(uint32_t),
                chip, cc, lr.kernel_text_addr);
        });
    }

    // ---- Step 2: Write runtime args to the RTA region in L1 ----
    for (const auto& lr : kernel.riscs) {
        if (lr.runtime_args.empty()) {
            continue;
        }
        uint64_t rta_addr = kernel.rta_base_addr +
            lr.processor_index * static_cast<uint64_t>(kMaxRtaWords * sizeof(uint32_t));
        driver.write_to_device(
            lr.runtime_args.data(),
            lr.runtime_args.size() * sizeof(uint32_t),
            chip, cc, rta_addr);
    }

    // ---- Step 3: Build launch_msg using the new StructBuffer/View API ----
    auto launch_msg_buf = dev_msgs_factory.create<tt_metal::dev_msgs::launch_msg_t>();
    {
        auto kernel_config = launch_msg_buf.view().kernel_config();

        // DISPATCH_MODE_HOST: host controls execution directly.
        kernel_config.mode() = tt_metal::dev_msgs::DISPATCH_MODE_HOST;

        // Enable processors that have loaded kernels.
        for (const auto& lr : kernel.riscs) {
            kernel_config.enables() |= (1u << lr.processor_index);
        }

        // kernel_config_base = KERNEL_CONFIG region start (= MEM_MAP_END).
        // Firmware computes:
        //   rta_l1_base  = kernel_config_base + rta_offset[proc].rta_offset
        //   kernel_lma   = kernel_config_base + kernel_text_offset[proc]
        // rta_offset is uint16_t, so RTA must be within first 64KB of this region.
        // kernel_text_offset is uint32_t; wrapping addition recovers absolute text_addr.
        constexpr uint32_t kTensixIdx = 0;  // ProgrammableCoreType::TENSIX == 0
        uint32_t kcfg_base = static_cast<uint32_t>(hal.get_dev_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::KERNEL_CONFIG));
        kernel_config.kernel_config_base()[kTensixIdx] = kcfg_base;

        // rta_offset[proc_idx]: relative to kernel_config_base.
        for (const auto& lr : kernel.riscs) {
            uint32_t abs_rta_addr = static_cast<uint32_t>(kernel.rta_base_addr) +
                lr.processor_index * kMaxRtaWords * sizeof(uint32_t);
            uint32_t rel_rta = abs_rta_addr - kcfg_base;
            auto rta_entry = kernel_config.rta_offset()[lr.processor_index];
            rta_entry.rta_offset() = static_cast<uint16_t>(rel_rta);
            rta_entry.crta_offset() = 0;
        }

        // kernel_text_offset[proc_idx]: offset from kernel_config_base to the loaded kernel binary.
        for (const auto& lr : kernel.riscs) {
            uint32_t text_off = static_cast<uint32_t>(lr.kernel_text_addr) - kcfg_base;
            kernel_config.kernel_text_offset()[lr.processor_index] = text_off;
        }

        // Skip remote CB setup: set min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS.
        // When this field equals NUM_CIRCULAR_BUFFERS, the firmware's setup_remote_cb_interfaces
        // loop and barrier_remote_cb_interface_setup are both no-ops (brisc.cc check).
        // Without this, firmware misinterprets the RTA region (at remote_cb_offset=0) as CB
        // config data, finds non-zero RTA values, and hangs on a bogus NOC transaction.
        kernel_config.min_remote_cb_start_index() = static_cast<uint8_t>(NUM_CIRCULAR_BUFFERS);
    }

    // ---- Step 4: Write launch_msg to the mailbox in L1 ----
    uint64_t launch_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::LAUNCH);

    driver.write_to_device_reg(
        launch_msg_buf.data(), static_cast<uint32_t>(launch_msg_buf.size()),
        chip, cc, launch_addr);

    // Memory fence: ensure launch_msg is visible before go_msg.
    tt_driver_atomics::sfence();

    // ---- Step 5: Fire go_msg (RUN_MSG_GO) ----
    // For unicast Tensix cores, go_message_index is always 0.
    // Write RUN_MSG_GO directly to go_messages[0] (= GO_MSG base address),
    // matching the pattern in llrt::write_launch_msg_to_core.
    uint64_t go_entry_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);

    uint32_t go_val = hal.make_go_msg_u32(
        static_cast<uint8_t>(tt_metal::dev_msgs::RUN_MSG_GO), 0, 0, 0);

    driver.write_to_device_reg(
        &go_val, sizeof(go_val),
        chip, cc, go_entry_addr);

    driver.l1_membar(chip);

    // ---- Step 6: Poll go_msg until RUN_MSG_DONE ----
    auto go_msg_buf = dev_msgs_factory.create<tt_metal::dev_msgs::go_msg_t>();
    auto start = std::chrono::steady_clock::now();
    while (true) {
        driver.read_from_device(
            go_msg_buf.data(),
            chip, cc, go_entry_addr,
            static_cast<std::size_t>(go_msg_buf.size()));

        uint8_t sig = go_msg_buf.view().signal();
        if (sig == tt_metal::dev_msgs::RUN_MSG_DONE) {
            break;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
        if (timeout_ms > 0 && elapsed > timeout_ms) {
            throw std::runtime_error(
                "tt-foil: timeout waiting for kernel completion on core ("
                + std::to_string(kernel.core.x) + ","
                + std::to_string(kernel.core.y) + ")");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
#endif  // legacy dispatch_execute

}  // namespace tt::foil
