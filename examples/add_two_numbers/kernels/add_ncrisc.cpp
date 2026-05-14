// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-side NCRISC kernel: read two uint32_t operands from L1, multiply,
// write the result back to L1.
//
// Runtime args (set by the host via set_runtime_args()):
//   arg[0] = L1 address of operand A
//   arg[1] = L1 address of operand B
//   arg[2] = L1 address of the result (product)
//
// This kernel runs concurrently with the BRISC kernel; both share the same
// core's L1 but write to disjoint result slots.

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t addr_a      = get_arg_val<uint32_t>(0);
    uint32_t addr_b      = get_arg_val<uint32_t>(1);
    uint32_t addr_result = get_arg_val<uint32_t>(2);

    volatile uint32_t* a      = reinterpret_cast<volatile uint32_t*>(addr_a);
    volatile uint32_t* b      = reinterpret_cast<volatile uint32_t*>(addr_b);
    volatile uint32_t* result = reinterpret_cast<volatile uint32_t*>(addr_result);

    *result = (*a) * (*b);
}
