// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::foil {

struct Device;
struct Kernel;

// Blocking slow-dispatch execution of a kernel:
//   1. Write kernel ELF(s) to core L1
//   2. Write runtime args to the RTA region
//   3. Build and write launch_msg (DISPATCH_MODE_HOST)
//   4. Fire go_msg (RUN_MSG_GO)
//   5. Poll go_msg until RUN_MSG_DONE
// Throws std::runtime_error on timeout (default 5 s) or unexpected mailbox value.
void dispatch_execute(Device& dev, Kernel& kernel, int timeout_ms = 5000);

}  // namespace tt::foil
