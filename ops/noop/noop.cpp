// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// G0 floor-measurement kernel. The body is intentionally empty — its
// purpose is to isolate the firmware-side dispatch overhead from any
// user-kernel exec time. A perf test launches this 1000× in a loop and
// reports p50/p99 per-launch wall.
//
// Runtime args: none.

#include <cstdint>

void kernel_main() {
    // intentionally empty
}
