// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

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

// Multi-kernel variant for cross-core synchronisation patterns:
//   1. For each kernel: write ELF(s) + RTA + launch_msg
//   2. For each kernel: fire go_msg (after a memory fence)
//   3. Poll every kernel's go_msg until all reach RUN_MSG_DONE
// All kernels launch effectively simultaneously from the device's POV
// (host issues every GO before any DONE check), which is what
// producer/consumer kernels need to start observing each other.
void dispatch_execute_multi(
    Device& dev,
    std::span<Kernel* const> kernels,
    int timeout_ms = 5000);

}  // namespace tt::foil
