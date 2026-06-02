// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tt-foil fast dispatch (R5). Minimal subset of tt-metal's CQ targeted at
// Qwen3-VL-2B decode. See docs/perf_fast_dispatch_feasibility.md and
// /home/kyamaguchi/.claude/plans/qwen-benchmark-tt-metal-runtime-research-frolicking-simon.md
// for the rationale and design context.
//
// A dispatcher kernel runs persistently on (1, 0) BRISC. The host writes
// commands into a 4 KB ring in that core's L1 and advances `host_wptr`.
// The dispatcher polls the wptr, parses commands, and issues NOC writes
// to worker cores (launch_msg + GO_MSG). When a worker completes, the
// dispatcher polls its GO_MSG over NOC, then bumps `completion_counter`
// which the host polls over PCIe.
//
// L1 layout on the dispatcher core (single allocation, base = `state_addr`):
//
//   off 0x0000  host_wptr        (u32, host writes, dispatcher reads)
//   off 0x0004  pad
//   off 0x0008  pad
//   off 0x000C  pad
//   off 0x0010  dispatcher_rdptr (u32, dispatcher writes)
//   off 0x0014  completion_count (u32, dispatcher writes, host reads)
//   off 0x0018  phase_marker     (u32, dispatcher debug)
//   off 0x001C  pad
//   off 0x0020  cmd_ring         (4 KB, host writes, dispatcher reads)
//
// Total: 4 KB + 32 B = 4128 B. We over-allocate to 4 KB + 64 B = 4160 B
// for alignment headroom.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "tt_foil/runtime.hpp"

namespace tt::foil {

// ---- Wire-format constants (shared with dispatcher kernel) ----------------

namespace fast_dispatch_layout {

constexpr uint32_t kHostWptrOffset       = 0x00;
constexpr uint32_t kDispatcherRdPtrOffset= 0x10;
constexpr uint32_t kCompletionOffset     = 0x14;
constexpr uint32_t kPhaseMarkerOffset    = 0x18;
constexpr uint32_t kCmdRingOffset        = 0x20;
constexpr uint32_t kCmdRingBytes         = 4096;
constexpr uint32_t kStateBytes           = kCmdRingOffset + kCmdRingBytes;

// Fixed-size cmd slots. Simplifies dispatcher parsing — no variable-length
// scan. 128 B per cmd × 32 cmds = 4 KB ring.
constexpr uint32_t kCmdSlotBytes = 128;
constexpr uint32_t kCmdSlotCount = kCmdRingBytes / kCmdSlotBytes;  // 32

enum CmdOp : uint32_t {
    CMD_NOOP         = 0,
    CMD_LAUNCH       = 1,
    CMD_NOTIFY_HOST  = 2,  // bump completion_counter
    CMD_TERMINATE    = 3,  // dispatcher exits its main loop (test teardown)
    CMD_LAUNCH_BATCH = 4,  // fire N workers' GO_MSGs then poll all DONE
};

// LAUNCH command layout (128 B total = 32 × uint32_t words):
//   word[0]  op (= CMD_LAUNCH)
//   word[1]  pad
//   word[2]  launch_noc_lo  (worker's LAUNCH mailbox 64-bit NOC addr, low 32 b)
//   word[3]  launch_noc_hi  (   "      "        "   "    "   "  "  , high 32 b)
//   word[4]  go_noc_lo      (worker's GO_MSG 64-bit NOC addr, low 32 b)
//   word[5]  go_noc_hi      (   "      "      "  "    "  "  , high 32 b)
//   word[6]  launch_msg_size
//   word[7]  go_msg_value   (hal.make_go_msg_u32(RUN_MSG_GO, 0, 0, 0))
//   word[8..31] launch_msg payload (96 B inline; tt-metal launch_msg_t is ~92 B)
struct LaunchCmd {
    uint32_t op;
    uint32_t pad0;
    uint32_t launch_noc_lo;
    uint32_t launch_noc_hi;
    uint32_t go_noc_lo;
    uint32_t go_noc_hi;
    uint32_t launch_msg_size;
    uint32_t go_msg_value;
    uint8_t  launch_msg[96];
};
static_assert(sizeof(LaunchCmd) == kCmdSlotBytes, "LaunchCmd must fill slot exactly");

struct NotifyCmd {
    uint32_t op;       // CMD_NOTIFY_HOST
    uint32_t reserved[31];
};
static_assert(sizeof(NotifyCmd) == kCmdSlotBytes, "NotifyCmd must fill slot");

struct TerminateCmd {
    uint32_t op;       // CMD_TERMINATE
    uint32_t reserved[31];
};
static_assert(sizeof(TerminateCmd) == kCmdSlotBytes, "TerminateCmd must fill slot");

// LAUNCH_BATCH: fire GO on N workers (no launch_msg write — host has
// already pre-staged via slow-dispatch setup), then poll each worker's
// GO_MSG until all read DONE. Up to 14 workers fit per cmd slot.
//   word[0]  op (= CMD_LAUNCH_BATCH)
//   word[1]  num_workers (1..14)
//   word[2]  go_msg_value (RUN_MSG_GO for all workers)
//   word[3]  reserved
//   word[4..5]   worker[0] go_noc (lo, hi)
//   word[6..7]   worker[1] go_noc (lo, hi)
//   ...
//   word[30..31] worker[13] go_noc (lo, hi)
struct LaunchBatchCmd {
    uint32_t op;
    uint32_t num_workers;
    uint32_t go_msg_value;
    uint32_t reserved;
    struct { uint32_t lo; uint32_t hi; } go_noc[14];
};
static_assert(sizeof(LaunchBatchCmd) == kCmdSlotBytes, "LaunchBatchCmd must fill slot");
constexpr uint32_t kMaxBatchWorkers = 14;

}  // namespace fast_dispatch_layout

// ---- Host API --------------------------------------------------------------

// A FastDispatch instance owns one dispatcher kernel + cmd ring on one core.
// G1 scope: smoke test only — push LAUNCH cmds, poll completion.
struct FastDispatch {
    Device& dev;
    CoreCoord dispatcher_core;        // e.g. {1, 0}
    std::shared_ptr<Buffer> state_buf;// L1 buffer holding wptr/rdptr/counter/cmd_ring
    std::shared_ptr<Kernel> dispatcher_kernel;
    uint32_t host_wptr_local = 0;     // host-side mirror of wptr
    uint32_t expected_completion = 0; // counter target

    FastDispatch(Device& dev, CoreCoord core);

    // Push a LAUNCH cmd: dispatcher will issue launch_msg + GO_MSG to worker,
    // poll the worker's GO_MSG.signal for DONE.
    // `launch_msg_bytes` is the raw payload as produced by HAL's StructBuffer
    // (same bytes the slow-dispatch path writes).
    // worker_launch_l1_addr and worker_go_msg_l1_addr are the worker-local
    // L1 offsets (e.g. HAL LAUNCH = 0x70, GO_MSG = 0x3F0 on Blackhole);
    // FastDispatch computes the full NOC address using the worker's
    // translated coord.
    void push_launch(CoreCoord worker,
                     uint32_t worker_launch_l1_addr,
                     uint32_t worker_go_msg_l1_addr,
                     const void* launch_msg_bytes,
                     uint32_t launch_msg_size);

    // Push a LAUNCH_BATCH: fire GO on N workers in parallel, then poll all
    // DONE. Use this for multi-core ops (e.g. matmul on 1×4 grid). Host
    // must have already written each worker's launch_msg via slow-dispatch
    // setup before calling.
    void push_launch_batch(std::span<const CoreCoord> workers,
                           uint32_t worker_go_msg_l1_addr);

    // Push a NOTIFY: dispatcher bumps completion_counter.
    void push_notify();

    // Push a TERMINATE: dispatcher exits its main loop, writes RUN_MSG_DONE
    // to its own GO_MSG. After this you can release the kernel normally.
    void push_terminate();

    // Block until completion_counter on device == expected_completion.
    // Returns wall microseconds spent waiting.
    double wait_for_completion(uint32_t target, int timeout_ms = 1000);

    // Start the dispatcher kernel on its core asynchronously.
    void start();
};

}  // namespace tt::foil
