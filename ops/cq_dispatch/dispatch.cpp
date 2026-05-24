// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// G1 fast-dispatch dispatcher kernel (BRISC). See src/fast_dispatch.hpp
// for the L1 layout and cmd format the host writes into the ring.
//
// Runtime args:
//   arg[0] = state_addr (uint32 L1 offset on this core; base of host_wptr /
//            rdptr / completion_counter / cmd_ring region)
//
// Main loop:
//   - poll host_wptr until it advances past local rd
//   - for each new cmd slot in [rd .. host_wptr):
//       * CMD_LAUNCH      → NOC-write launch_msg to worker (if size > 0),
//                            write GO_MSG, poll worker's GO_MSG for DONE
//       * CMD_NOTIFY_HOST → bump completion_counter
//       * CMD_TERMINATE   → return from kernel_main (firmware writes DONE)

#include <cstdint>
#include "dataflow_api.h"

// Keep in sync with src/fast_dispatch.hpp / fast_dispatch_layout.
constexpr uint32_t kHostWptrOffset        = 0x00;
constexpr uint32_t kDispatcherRdPtrOffset = 0x10;
constexpr uint32_t kCompletionOffset      = 0x14;
constexpr uint32_t kPhaseMarkerOffset     = 0x18;
constexpr uint32_t kCmdRingOffset         = 0x20;
constexpr uint32_t kCmdSlotBytes          = 128;
constexpr uint32_t kCmdSlotCount          = 32;  // 4 KB / 128
constexpr uint32_t kGoMsgScratchOffset    = kCmdRingOffset + kCmdSlotBytes * kCmdSlotCount;

constexpr uint32_t CMD_NOOP         = 0;
constexpr uint32_t CMD_LAUNCH       = 1;
constexpr uint32_t CMD_NOTIFY_HOST  = 2;
constexpr uint32_t CMD_TERMINATE    = 3;
constexpr uint32_t CMD_LAUNCH_BATCH = 4;

constexpr uint32_t RUN_MSG_DONE_VAL = 0;  // dev_msgs.hpp: RUN_MSG_DONE = 0
constexpr uint8_t  kDispatchNoc     = 1;  // BRISC peer writes use NOC 1 (CLAUDE.md)

void kernel_main() {
    // BRISC firmware only initialises `noc_index` (=0 by default from
    // launch_msg.brisc_noc_id). The dispatcher uses NOC 1 for peer-Tensix
    // writes (CLAUDE.md invariant: BRISC peer L1 writes hang on NOC 0), so
    // we must init NOC 1's local state explicitly here. Without this,
    // noc_async_write_barrier(noc=1) hangs forever.
    noc_local_state_init(kDispatchNoc);

    const uint32_t state_addr = get_arg_val<uint32_t>(0);

    volatile uint32_t* const host_wptr_p =
        reinterpret_cast<volatile uint32_t*>(state_addr + kHostWptrOffset);
    volatile uint32_t* const rdptr_p =
        reinterpret_cast<volatile uint32_t*>(state_addr + kDispatcherRdPtrOffset);
    volatile uint32_t* const completion_p =
        reinterpret_cast<volatile uint32_t*>(state_addr + kCompletionOffset);
    volatile uint32_t* const phase_p =
        reinterpret_cast<volatile uint32_t*>(state_addr + kPhaseMarkerOffset);
    volatile uint32_t* const go_msg_scratch =
        reinterpret_cast<volatile uint32_t*>(state_addr + kGoMsgScratchOffset);

    *rdptr_p = 0;
    *completion_p = 0;
    *phase_p = 0xAA000000u;

    uint32_t rd = 0;
    bool running = true;
    while (running) {
        while (*host_wptr_p == rd) { /* spin */ }
        const uint32_t wptr_snap = *host_wptr_p;

        while (rd != wptr_snap) {
            const uint32_t slot = rd % kCmdSlotCount;
            const uint32_t cmd_l1 = state_addr + kCmdRingOffset + slot * kCmdSlotBytes;
            volatile uint32_t* const cmd_p =
                reinterpret_cast<volatile uint32_t*>(cmd_l1);
            const uint32_t op = cmd_p[0];

            if (op == CMD_LAUNCH) {
                const uint64_t launch_noc =
                    (static_cast<uint64_t>(cmd_p[3]) << 32) | cmd_p[2];
                const uint64_t go_noc =
                    (static_cast<uint64_t>(cmd_p[5]) << 32) | cmd_p[4];
                const uint32_t lm_size = cmd_p[6];
                const uint32_t go_val  = cmd_p[7];
                const uint32_t lm_payload_l1 = cmd_l1 + 8u * sizeof(uint32_t);

                if (lm_size != 0) {
                    noc_async_write_one_packet(
                        lm_payload_l1, launch_noc, lm_size, kDispatchNoc);
                    noc_async_write_barrier(kDispatchNoc);
                }

                *go_msg_scratch = go_val;
                noc_async_write_one_packet(
                    reinterpret_cast<uint32_t>(go_msg_scratch),
                    go_noc, 4, kDispatchNoc);
                noc_async_write_barrier(kDispatchNoc);

                // Poll worker's GO_MSG.signal until RUN_MSG_DONE (=0).
                // Signal byte sits at byte offset 3 (high byte) of the
                // 4-byte mailbox word per dev_msgs go_msg_t layout.
                while (true) {
                    noc_async_read_one_packet(
                        go_noc,
                        reinterpret_cast<uint32_t>(go_msg_scratch),
                        4, kDispatchNoc);
                    noc_async_read_barrier(kDispatchNoc);
                    if (((*go_msg_scratch >> 24) & 0xFFu) == RUN_MSG_DONE_VAL) {
                        break;
                    }
                }
            } else if (op == CMD_LAUNCH_BATCH) {
                // LaunchBatchCmd layout:
                //   word[1]   num_workers
                //   word[2]   go_msg_value
                //   word[3]   reserved
                //   word[4..] go_noc pairs (lo, hi) per worker
                const uint32_t num_workers = cmd_p[1];
                const uint32_t go_val      = cmd_p[2];

                // Phase 1: fire GO on every worker.
                *go_msg_scratch = go_val;
                for (uint32_t i = 0; i < num_workers; ++i) {
                    const uint32_t lo = cmd_p[4 + i * 2];
                    const uint32_t hi = cmd_p[4 + i * 2 + 1];
                    const uint64_t noc = (static_cast<uint64_t>(hi) << 32) | lo;
                    noc_async_write_one_packet(
                        reinterpret_cast<uint32_t>(go_msg_scratch),
                        noc, 4, kDispatchNoc);
                }
                noc_async_write_barrier(kDispatchNoc);

                // Phase 2: poll all workers' GO_MSG until every signal == DONE.
                // We sweep the worker list each iteration; once a worker
                // reads DONE we don't need to re-check it, but the cost of
                // a NOC read is sub-µs so re-polling is fine and keeps the
                // code simple.
                while (true) {
                    bool all_done = true;
                    for (uint32_t i = 0; i < num_workers; ++i) {
                        const uint32_t lo = cmd_p[4 + i * 2];
                        const uint32_t hi = cmd_p[4 + i * 2 + 1];
                        const uint64_t noc = (static_cast<uint64_t>(hi) << 32) | lo;
                        noc_async_read_one_packet(
                            noc,
                            reinterpret_cast<uint32_t>(go_msg_scratch),
                            4, kDispatchNoc);
                        noc_async_read_barrier(kDispatchNoc);
                        if (((*go_msg_scratch >> 24) & 0xFFu) != RUN_MSG_DONE_VAL) {
                            all_done = false;
                            break;
                        }
                    }
                    if (all_done) break;
                }
            } else if (op == CMD_NOTIFY_HOST) {
                *completion_p = *completion_p + 1u;
            } else if (op == CMD_TERMINATE) {
                running = false;
                rd += 1;
                *rdptr_p = rd;
                break;
            }

            rd += 1;
            *rdptr_p = rd;
        }
    }

    *phase_p = 0xDDDDDDDDu;
}
