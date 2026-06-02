// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tt-foil fast dispatch host implementation (R5 G1). See fast_dispatch.hpp
// for layout and protocol.

#include "fast_dispatch.hpp"

#include "device.hpp"
#include "dispatch.hpp"
#include "firmware_paths.hpp"
#include "kernel.hpp"
#include "noc_addr.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "tt-metalium/circular_buffer_constants.h"

namespace tt::foil {

using namespace fast_dispatch_layout;

namespace {
// Mirror of the static helper in device.cpp.
tt::umd::CoreCoord tensix_translated_local(const Device& dev, const CoreCoord& core) {
    const auto& soc_desc = dev.umd_driver->get_soc_descriptor(dev.chip_id);
    auto virt = soc_desc.translate_coord_to(
        tt::umd::CoreCoord{core.x, core.y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL},
        tt::CoordSystem::TRANSLATED);
    return tt::umd::CoreCoord{virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
}
}  // namespace

FastDispatch::FastDispatch(Device& dev_in, CoreCoord core)
    : dev(dev_in), dispatcher_core(core) {
    // Allocate L1 state buffer on the dispatcher core. Layout:
    //   0x00..0x1F   mailboxes (host_wptr, rdptr, completion, phase, pads)
    //   0x20..0x101F cmd_ring (4 KB)
    //   kGoMsgScratchOffset  go_msg scratch (kernel reuses one 4 B slot for poll)
    //   kTraceScratchOffset  one 128-B slot to stage a trace cmd read from DRAM
    state_buf = allocate_buffer(dev, BufferLocation::L1, kStateTotalBytes, core);

    // Zero-fill the state region so wptr/rdptr/counter start at 0.
    std::vector<uint8_t> zeros(kStateTotalBytes, 0);
    write_buffer(dev, *state_buf, zeros.data(), zeros.size());

    // Load dispatcher kernel ELF. The kernel ELF dir is resolved the same
    // way as user kernels — caller must set TT_FOIL_OPS_DIR or pass via
    // ops env.
    const char* ops_dir = std::getenv("TT_FOIL_OPS_DIR");
    if (!ops_dir) {
        throw std::runtime_error(
            "FastDispatch: TT_FOIL_OPS_DIR must be set (looking for cq_dispatch/prebuilt/dispatch.brisc.elf)");
    }
    std::string elf_path = std::string(ops_dir) + "/cq_dispatch/prebuilt/dispatch.brisc.elf";

    std::array<RiscBinary, 1> bins = {{
        {RiscBinary::RiscId::BRISC, elf_path},
    }};
    dispatcher_kernel = load_kernel(dev, bins, core);

    // RTA[0] = state_addr (L1 offset).
    std::array<uint32_t, 1> rta = {
        static_cast<uint32_t>(state_buf->device_addr),
    };
    set_runtime_args(dev, *dispatcher_kernel, RiscBinary::RiscId::BRISC, rta);
}

void FastDispatch::start() {
    // Fire-and-forget launch the dispatcher kernel. It runs an infinite loop
    // until we push CMD_TERMINATE.
    dispatch_launch_async(dev, *dispatcher_kernel);
}

void FastDispatch::emit_or_record(const void* cmd_buf) {
    if (recording_) {
        const uint8_t* p = static_cast<const uint8_t*>(cmd_buf);
        record_buf_.insert(record_buf_.end(), p, p + kCmdSlotBytes);
        return;
    }
    const uint32_t slot = host_wptr_local % kCmdSlotCount;
    const uint64_t slot_addr =
        state_buf->device_addr + kCmdRingOffset + slot * kCmdSlotBytes;
    write_l1(dev, dispatcher_core, slot_addr, cmd_buf, kCmdSlotBytes);
    host_wptr_local += 1;
    write_l1(dev, dispatcher_core,
             state_buf->device_addr + kHostWptrOffset,
             &host_wptr_local, sizeof(uint32_t));
}

void FastDispatch::begin_record() {
    recording_ = true;
    record_buf_.clear();
}

FastDispatch::TraceHandle FastDispatch::end_record() {
    recording_ = false;
    TraceHandle h;
    h.num_cmds = static_cast<uint32_t>(record_buf_.size() / kCmdSlotBytes);
    if (record_buf_.empty()) return h;  // empty trace
    h.dram = allocate_buffer(dev, BufferLocation::DRAM, record_buf_.size());
    write_buffer(dev, *h.dram, record_buf_.data(), record_buf_.size());
    return h;
}

void FastDispatch::exec_trace(const TraceHandle& trace) {
    if (trace.num_cmds == 0 || !trace.dram) return;
    const uint64_t dram_noc = make_noc_dram_addr(dev, trace.dram->device_addr);

    alignas(16) uint8_t cmd_buf[kCmdSlotBytes];
    std::memset(cmd_buf, 0, kCmdSlotBytes);
    ExecTraceCmd* cmd = reinterpret_cast<ExecTraceCmd*>(cmd_buf);
    cmd->op          = CMD_EXEC_TRACE;
    cmd->num_cmds    = trace.num_cmds;
    cmd->dram_noc_lo = static_cast<uint32_t>(dram_noc & 0xFFFFFFFFu);
    cmd->dram_noc_hi = static_cast<uint32_t>(dram_noc >> 32);

    // EXEC_TRACE is a control cmd — always goes to the ring, never recorded.
    const uint32_t slot = host_wptr_local % kCmdSlotCount;
    const uint64_t slot_addr =
        state_buf->device_addr + kCmdRingOffset + slot * kCmdSlotBytes;
    write_l1(dev, dispatcher_core, slot_addr, cmd_buf, kCmdSlotBytes);
    host_wptr_local += 1;
    write_l1(dev, dispatcher_core,
             state_buf->device_addr + kHostWptrOffset,
             &host_wptr_local, sizeof(uint32_t));
}

void FastDispatch::push_launch(CoreCoord worker,
                                uint32_t worker_launch_l1_addr,
                                uint32_t worker_go_msg_l1_addr,
                                const void* launch_msg_bytes,
                                uint32_t launch_msg_size) {
    if (launch_msg_size > sizeof(LaunchCmd::launch_msg)) {
        throw std::runtime_error("FastDispatch: launch_msg too large for cmd slot");
    }

    // Build the 64-bit NOC addresses to the worker's LAUNCH and GO_MSG mailboxes.
    const tt::umd::CoreCoord worker_t = tensix_translated_local(dev, worker);
    const uint64_t launch_noc = make_noc_unicast_addr(worker_t, worker_launch_l1_addr);
    const uint64_t go_noc     = make_noc_unicast_addr(worker_t, worker_go_msg_l1_addr);

    // Build the GO_MSG signal value (RUN_MSG_GO in low byte).
    const uint32_t go_val = dev.hal->make_go_msg_u32(
        static_cast<uint8_t>(tt_metal::dev_msgs::RUN_MSG_GO), 0, 0, 0);

    // Compose the cmd in a local buffer, then write to the appropriate
    // ring slot in device L1.
    alignas(16) uint8_t cmd_buf[kCmdSlotBytes];
    std::memset(cmd_buf, 0, kCmdSlotBytes);
    LaunchCmd* cmd = reinterpret_cast<LaunchCmd*>(cmd_buf);
    cmd->op               = CMD_LAUNCH;
    cmd->pad0             = 0;
    cmd->launch_noc_lo    = static_cast<uint32_t>(launch_noc & 0xFFFFFFFFu);
    cmd->launch_noc_hi    = static_cast<uint32_t>(launch_noc >> 32);
    cmd->go_noc_lo        = static_cast<uint32_t>(go_noc & 0xFFFFFFFFu);
    cmd->go_noc_hi        = static_cast<uint32_t>(go_noc >> 32);
    cmd->launch_msg_size  = launch_msg_size;
    cmd->go_msg_value     = go_val;
    if (launch_msg_size != 0 && launch_msg_bytes != nullptr) {
        std::memcpy(cmd->launch_msg, launch_msg_bytes, launch_msg_size);
    }

    emit_or_record(cmd_buf);
}

void FastDispatch::push_launch_batch(std::span<const CoreCoord> workers,
                                      uint32_t worker_go_msg_l1_addr) {
    if (workers.empty()) return;
    if (workers.size() > kMaxBatchWorkers) {
        throw std::runtime_error(
            "FastDispatch::push_launch_batch: too many workers (max "
            + std::to_string(kMaxBatchWorkers) + ")");
    }
    const uint32_t go_val = dev.hal->make_go_msg_u32(
        static_cast<uint8_t>(tt_metal::dev_msgs::RUN_MSG_GO), 0, 0, 0);

    alignas(16) uint8_t cmd_buf[kCmdSlotBytes];
    std::memset(cmd_buf, 0, kCmdSlotBytes);
    LaunchBatchCmd* cmd = reinterpret_cast<LaunchBatchCmd*>(cmd_buf);
    cmd->op           = CMD_LAUNCH_BATCH;
    cmd->num_workers  = static_cast<uint32_t>(workers.size());
    cmd->go_msg_value = go_val;
    for (size_t i = 0; i < workers.size(); ++i) {
        const tt::umd::CoreCoord wt = tensix_translated_local(dev, workers[i]);
        const uint64_t noc = make_noc_unicast_addr(wt, worker_go_msg_l1_addr);
        cmd->go_noc[i].lo = static_cast<uint32_t>(noc & 0xFFFFFFFFu);
        cmd->go_noc[i].hi = static_cast<uint32_t>(noc >> 32);
    }

    emit_or_record(cmd_buf);
}

void FastDispatch::push_notify() {
    alignas(16) uint8_t cmd_buf[kCmdSlotBytes];
    std::memset(cmd_buf, 0, kCmdSlotBytes);
    NotifyCmd* cmd = reinterpret_cast<NotifyCmd*>(cmd_buf);
    cmd->op = CMD_NOTIFY_HOST;

    const uint32_t slot = host_wptr_local % kCmdSlotCount;
    const uint64_t slot_addr = state_buf->device_addr + kCmdRingOffset + slot * kCmdSlotBytes;
    write_l1(dev, dispatcher_core, slot_addr, cmd_buf, kCmdSlotBytes);

    expected_completion += 1;
    host_wptr_local += 1;
    write_l1(dev, dispatcher_core,
             state_buf->device_addr + kHostWptrOffset,
             &host_wptr_local, sizeof(uint32_t));
}

void FastDispatch::push_terminate() {
    alignas(16) uint8_t cmd_buf[kCmdSlotBytes];
    std::memset(cmd_buf, 0, kCmdSlotBytes);
    TerminateCmd* cmd = reinterpret_cast<TerminateCmd*>(cmd_buf);
    cmd->op = CMD_TERMINATE;

    const uint32_t slot = host_wptr_local % kCmdSlotCount;
    const uint64_t slot_addr = state_buf->device_addr + kCmdRingOffset + slot * kCmdSlotBytes;
    write_l1(dev, dispatcher_core, slot_addr, cmd_buf, kCmdSlotBytes);

    host_wptr_local += 1;
    write_l1(dev, dispatcher_core,
             state_buf->device_addr + kHostWptrOffset,
             &host_wptr_local, sizeof(uint32_t));
}

double FastDispatch::wait_for_completion(uint32_t target, int timeout_ms) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    uint32_t counter = 0;
    while (true) {
        read_l1(dev, dispatcher_core,
                state_buf->device_addr + kCompletionOffset,
                &counter, sizeof(uint32_t));
        if (counter >= target) {
            break;
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - t0).count();
        if (timeout_ms > 0 && elapsed_ms > timeout_ms) {
            uint32_t phase = 0;
            read_l1(dev, dispatcher_core,
                    state_buf->device_addr + kPhaseMarkerOffset,
                    &phase, sizeof(uint32_t));
            uint32_t rdptr = 0;
            read_l1(dev, dispatcher_core,
                    state_buf->device_addr + kDispatcherRdPtrOffset,
                    &rdptr, sizeof(uint32_t));
            throw std::runtime_error(
                "FastDispatch::wait_for_completion timeout: target=" + std::to_string(target)
                + " counter=" + std::to_string(counter)
                + " rdptr=" + std::to_string(rdptr)
                + " phase=0x" + [&]{ char b[16]; std::snprintf(b, 16, "%08X", phase); return std::string(b); }()
                + " host_wptr_local=" + std::to_string(host_wptr_local));
        }
    }
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

}  // namespace tt::foil
