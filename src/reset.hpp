// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (steps 6 + 7): Tensix RISC reset control + INIT mailbox poll.
//
// On Blackhole, only BRISC is taken out of reset by the host. BRISC firmware
// internally drives NCRISC and TRISC0/1/2 via the subordinate_sync mailbox
// (brisc.cc:438-442), so a single host-side deassert of BRISC is enough to
// kick off boot of all five RISCs.
//
// Boot protocol from the host's view:
//   - All RISCs start in soft reset
//   - Host writes firmware ELFs, mailboxes (GO_MSG.signal=RUN_MSG_INIT), bank
//     tables, core_info
//   - Host deasserts BRISC reset
//   - BRISC firmware copies bank tables into local mem, starts NCRISC+TRISCs,
//     waits for them, then writes RUN_MSG_DONE back into go_messages[0]
//   - Host polls go_msg until it sees the transition from RUN_MSG_INIT to
//     RUN_MSG_DONE

#pragma once

#include <cstdint>

namespace tt {
namespace umd {
class Cluster;
struct CoreCoord;
}  // namespace umd
namespace tt_metal {
class Hal;
}
}  // namespace tt

namespace tt::foil {

// Assert soft reset on all RISCs of a Tensix core.
void assert_tensix_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core);

// Deassert BRISC soft reset; BRISC firmware boots and brings up the
// subordinate RISCs internally.
void deassert_brisc_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core);

// Poll GO_MSG.signal until it transitions from RUN_MSG_INIT to RUN_MSG_DONE,
// indicating BRISC firmware finished its init sequence. Throws on timeout.
void wait_tensix_init_done(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    int timeout_ms = 10000);

}  // namespace tt::foil
