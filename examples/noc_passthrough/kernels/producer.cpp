// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Producer BRISC kernel for the NOC inter-core passthrough example.
//
// Reads two uint32_t operands from this core's L1, computes their sum,
// posts the sum to the consumer core's L1 via noc_async_write, and finally
// writes a sentinel flag to a separate slot on the consumer to release its
// busy-loop. The destination NOC addresses are pre-computed by the host
// (`tt::foil::make_noc_unicast_addr`) and passed in as runtime args, so
// this kernel doesn't need the worker_logical_to_virtual table.
//
// Runtime args (host writes via set_runtime_args, BRISC RTA slot):
//   arg[0]   = local L1 addr of operand A
//   arg[1]   = local L1 addr of operand B
//   arg[2]   = local L1 addr of a scratch slot to assemble `sum` in
//   arg[3]   = dst sum   NOC addr lo32   (assembled via make_noc_unicast_addr)
//   arg[4]   = dst sum   NOC addr hi32
//   arg[5]   = local L1 addr of a scratch slot for the flag word
//   arg[6]   = dst flag  NOC addr lo32
//   arg[7]   = dst flag  NOC addr hi32

#include <cstdint>

#include "dataflow_api.h"

namespace { constexpr uint32_t kFlagValue = 0xCAFEBABEu; }

void kernel_main() {
    uint32_t a_addr        = get_arg_val<uint32_t>(0);
    uint32_t b_addr        = get_arg_val<uint32_t>(1);
    uint32_t sum_scratch   = get_arg_val<uint32_t>(2);
    uint32_t dst_sum_lo    = get_arg_val<uint32_t>(3);
    uint32_t dst_sum_hi    = get_arg_val<uint32_t>(4);
    uint32_t flag_scratch  = get_arg_val<uint32_t>(5);
    uint32_t dst_flag_lo   = get_arg_val<uint32_t>(6);
    uint32_t dst_flag_hi   = get_arg_val<uint32_t>(7);

    auto* a   = reinterpret_cast<volatile uint32_t*>(a_addr);
    auto* b   = reinterpret_cast<volatile uint32_t*>(b_addr);
    auto* sum = reinterpret_cast<volatile uint32_t*>(sum_scratch);
    auto* flg = reinterpret_cast<volatile uint32_t*>(flag_scratch);

    *sum = *a + *b;
    *flg = kFlagValue;

    const uint64_t dst_sum_noc  = (static_cast<uint64_t>(dst_sum_hi)  << 32) | dst_sum_lo;
    const uint64_t dst_flag_noc = (static_cast<uint64_t>(dst_flag_hi) << 32) | dst_flag_lo;

    // Order matters: the sum write must be observed on the consumer side
    // before the flag write so a flag-then-stale-sum race can't happen.
    // A single barrier between the two NOC posts is sufficient because the
    // consumer's flag busy-loop is the only synchronisation we rely on.
    noc_async_write_one_packet(sum_scratch,  dst_sum_noc,  sizeof(uint32_t));
    noc_async_write_barrier();
    noc_async_write_one_packet(flag_scratch, dst_flag_noc, sizeof(uint32_t));
    noc_async_write_barrier();
}
