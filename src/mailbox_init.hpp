// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 4): Tensix mailbox boot-time init via UMD direct.
//
// Mirrors the "simple" half of risc_firmware_initializer's
// write_initial_go_launch_msg (single-core, unicast variant):
//   - LAUNCH                    <- N copies of a zero launch_msg_t
//   - GO_MSG                    <- go_msg_t with signal=RUN_MSG_INIT
//   - LAUNCH_MSG_BUFFER_RD_PTR  <- 0
//   - GO_MSG_INDEX              <- 0
//
// The CORE_INFO mailbox is more involved (populates SoC topology +
// harvesting + magic-number fields by walking the SoC descriptor) and is
// handled separately in step 4b.

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

// Initialize the four simple Tensix mailboxes at boot time. After this call
// the launch_msg buffer is fully zeroed, GO_MSG.signal == RUN_MSG_INIT, and
// both pointers are 0 — matching the state BRISC firmware expects on cold
// boot before reset is deasserted.
//
// NOTE: this should only be called while the target core's RISCs are held in
// reset. Writing RUN_MSG_INIT to GO_MSG on a running core could trigger
// unintended re-init behavior.
void init_tensix_mailboxes(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core);

}  // namespace tt::foil
