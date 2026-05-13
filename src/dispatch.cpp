// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include "device.hpp"
#include "kernel.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// tt-metal headers
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_memory.h"

// Blackhole device messages (launch_msg_t, go_msg_t, DISPATCH_MODE_HOST, RUN_MSG_*)
#include "dev_msgs.h"
#include "dev_mem_map.h"

// UMD pair type
#include <umd/device/types/xy_pair.hpp>

namespace tt::foil {

// ---------------------------------------------------------------------------
// Write kernel ELF to core L1 (without MetalContext)
// ---------------------------------------------------------------------------
static void write_elf_to_core(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const ll_api::memory& mem,
    uint32_t chip_id,
    const tt::CoreCoord& virt_core,
    uint64_t local_init_addr)
{
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr = hal.relocate_dev_addr(addr, local_init_addr, /*shared_local=*/false);
        cluster.write_core(
            &*mem_ptr,
            len_words * sizeof(uint32_t),
            tt_cxy_pair(chip_id, virt_core),
            relo_addr);
    });
}

// ---------------------------------------------------------------------------
// dispatch_execute — blocking slow-dispatch
// ---------------------------------------------------------------------------
void dispatch_execute(Device& dev, Kernel& kernel, int timeout_ms) {
    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::Cluster& cluster         = *dev.cluster;
    const uint32_t chip          = dev.chip_id;

    // Resolve logical → virtual core coordinate.
    tt::CoreCoord virt{kernel.core.x, kernel.core.y};

    // ---- Step 1: Write kernel ELF binaries to core L1 ----
    for (const auto& lr : kernel.riscs) {
        const auto& jit_cfg = hal.get_jit_build_config(
            static_cast<uint32_t>(tt_metal::HalProgrammableCoreType::TENSIX),
            lr.proc_class,
            lr.proc_type);

        write_elf_to_core(cluster, hal, *lr.mem, chip, virt, jit_cfg.local_init_addr);
    }

    // ---- Step 2: Write runtime args to the RTA region in L1 ----
    // RTA layout: [proc_idx * kMaxRtaWords * 4] byte offset from rta_base_addr.
    for (const auto& lr : kernel.riscs) {
        if (lr.runtime_args.empty()) {
            continue;
        }
        uint64_t rta_addr = kernel.rta_base_addr +
            lr.processor_index * static_cast<uint64_t>(kMaxRtaWords * sizeof(uint32_t));
        cluster.write_core(
            lr.runtime_args.data(),
            static_cast<uint32_t>(lr.runtime_args.size() * sizeof(uint32_t)),
            tt_cxy_pair(chip, virt),
            rta_addr);
    }

    // ---- Step 3: Build launch_msg ----
    // We construct a launch_msg_t on the stack and zero it out.
    // Only the fields needed for DISPATCH_MODE_HOST are populated.
    tt_metal::dev_msgs::launch_msg_t launch_msg{};
    std::memset(&launch_msg, 0, sizeof(launch_msg));

    // Set dispatch mode to HOST (no dispatch firmware).
    launch_msg.kernel_config.mode = tt_metal::dev_msgs::DISPATCH_MODE_HOST;

    // Enable the processors that have loaded kernels.
    // Bit i set in 'enables' means processor i is active.
    uint32_t enables = 0;
    for (const auto& lr : kernel.riscs) {
        enables |= (1u << lr.processor_index);
    }
    launch_msg.kernel_config.enables = enables;

    // kernel_config_base[TENSIX] = address in L1 that the firmware uses to find
    // the kernel config (including the RTA base offsets).
    // We point it at the RTA region itself: the firmware reads rta_offset[proc_idx]
    // and adds it to kernel_config_base to locate the args.
    // With rta_offset = proc_idx * kMaxRtaWords * 4, args are at the right place.
    constexpr uint32_t kTensixIdx = 0;  // ProgrammableCoreType::TENSIX == 0 in Blackhole
    launch_msg.kernel_config.kernel_config_base[kTensixIdx] =
        static_cast<uint32_t>(kernel.rta_base_addr);

    // Set rta_offset for each active processor.
    // rta_offset is the byte offset from kernel_config_base to where that proc's args start.
    for (const auto& lr : kernel.riscs) {
        launch_msg.kernel_config.rta_offset[lr.processor_index].rta_offset =
            static_cast<uint16_t>(lr.processor_index * kMaxRtaWords * sizeof(uint32_t));
        // crta_offset = 0 (no compile-time runtime args in v1)
        launch_msg.kernel_config.rta_offset[lr.processor_index].crta_offset = 0;
    }

    // ---- Step 4: Write launch_msg to the mailbox in L1 ----
    uint64_t launch_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::LAUNCH);

    cluster.write_core_immediate(
        &launch_msg, sizeof(launch_msg),
        tt_cxy_pair(chip, virt),
        launch_addr);

    // Memory fence: ensure launch_msg is visible before go_msg.
    tt_driver_atomics::sfence();

    // ---- Step 5: Fire go_msg (RUN_MSG_GO) ----
    tt_metal::dev_msgs::go_msg_t go_msg{};
    go_msg.signal = tt_metal::dev_msgs::RUN_MSG_GO;

    uint64_t go_addr = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::GO_MSG);

    cluster.write_core_immediate(
        &go_msg, sizeof(go_msg),
        tt_cxy_pair(chip, virt),
        go_addr);

    cluster.l1_barrier(chip);

    // ---- Step 6: Poll go_msg until RUN_MSG_DONE ----
    auto start = std::chrono::steady_clock::now();
    while (true) {
        tt_metal::dev_msgs::go_msg_t result{};
        cluster.read_core(
            &result, sizeof(result),
            tt_cxy_pair(chip, virt),
            go_addr & ~0x3ULL);  // read must be 4-byte aligned

        if (result.signal == tt_metal::dev_msgs::RUN_MSG_DONE) {
            break;
        }
        if (result.signal != tt_metal::dev_msgs::RUN_MSG_GO) {
            throw std::runtime_error(
                "tt-foil: unexpected go_msg signal: " + std::to_string(result.signal));
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

}  // namespace tt::foil
