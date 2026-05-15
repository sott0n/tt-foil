// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Consumer BRISC kernel for the NOC inter-core passthrough example.
//
// Busy-loops on a flag word in this core's local L1 until the producer
// writes the agreed-upon sentinel via noc_async_write; once it does, reads
// `sum` from the slot the producer wrote, computes `result = sum * 2`, and
// stores it locally. The host then reads `result` to verify.
//
// Runtime args:
//   arg[0]  = local L1 addr of the flag slot (busy-loop target)
//   arg[1]  = local L1 addr of the sum slot   (producer's NOC write target)
//   arg[2]  = local L1 addr of the result slot (host reads this after exec)

#include <cstdint>

#include "dataflow_api.h"

namespace { constexpr uint32_t kFlagValue = 0xCAFEBABEu; }

void kernel_main() {
    uint32_t flag_addr   = get_arg_val<uint32_t>(0);
    uint32_t sum_addr    = get_arg_val<uint32_t>(1);
    uint32_t result_addr = get_arg_val<uint32_t>(2);

    auto* flag   = reinterpret_cast<volatile uint32_t*>(flag_addr);
    auto* sum    = reinterpret_cast<volatile uint32_t*>(sum_addr);
    auto* result = reinterpret_cast<volatile uint32_t*>(result_addr);

    // Busy-loop. Producer's noc_async_write_barrier ensures the sum slot
    // is visible by the time the flag write lands here.
    while (*flag != kFlagValue) {
        // spin
    }

    *result = (*sum) * 2u;
}
