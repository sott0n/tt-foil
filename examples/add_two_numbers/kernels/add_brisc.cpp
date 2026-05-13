// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-side BRISC kernel: read two uint32_t operands from L1, add them,
// write the result back to L1.
//
// Runtime args (set by the host via set_runtime_args()):
//   arg[0] = L1 address of operand A
//   arg[1] = L1 address of operand B
//   arg[2] = L1 address of the result
//
// Pre-compile with:
//   riscv32-unknown-elf-g++ -march=rv32imc -mabi=ilp32 \
//     -O2 -fno-exceptions -fno-rtti \
//     -I<tt-metal>/tt_metal/hw/inc \
//     -I<tt-metal>/tt_metal/hw/inc/hostdev \
//     add_brisc.cpp -o add_brisc.elf

#include <cstdint>

// Device-side API for reading runtime args.
// get_arg_val<T>(i) reads from the RTA region set up by the host.
#include "dataflow_api.h"

void kernel_main() {
    // Read the three L1 addresses passed as runtime args.
    uint32_t addr_a      = get_arg_val<uint32_t>(0);
    uint32_t addr_b      = get_arg_val<uint32_t>(1);
    uint32_t addr_result = get_arg_val<uint32_t>(2);

    // Dereference the L1 addresses directly (all in the same core's L1).
    volatile uint32_t* a      = reinterpret_cast<volatile uint32_t*>(addr_a);
    volatile uint32_t* b      = reinterpret_cast<volatile uint32_t*>(addr_b);
    volatile uint32_t* result = reinterpret_cast<volatile uint32_t*>(addr_result);

    *result = *a + *b;
}
