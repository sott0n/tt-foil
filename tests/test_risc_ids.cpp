// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v4-1: confirm RiscId values map to the expected
// (processor_class, processor_type, processor_index) tuples.
//
// No hardware required. This is a static surface-area test against
// risc_to_hal_indices; subsequent v4 steps (CB layout, launch_msg
// wiring, compute kernel build, end-to-end) build on these IDs.

#include <cstdio>
#include <cstdint>

#include "tt_foil/runtime.hpp"
#include "kernel.hpp"

namespace {

struct Expectation {
    tt::foil::RiscBinary::RiscId id;
    const char* name;
    uint32_t proc_class;
    uint32_t proc_type;
    uint32_t proc_idx;
};

constexpr Expectation kExpectations[] = {
    {tt::foil::RiscBinary::RiscId::BRISC,  "BRISC",  /*DM*/      0, 0, 0},
    {tt::foil::RiscBinary::RiscId::NCRISC, "NCRISC", /*DM*/      0, 1, 1},
    {tt::foil::RiscBinary::RiscId::TRISC0, "TRISC0", /*COMPUTE*/ 1, 0, 2},
    {tt::foil::RiscBinary::RiscId::TRISC1, "TRISC1", /*COMPUTE*/ 1, 1, 3},
    {tt::foil::RiscBinary::RiscId::TRISC2, "TRISC2", /*COMPUTE*/ 1, 2, 4},
};

}  // namespace

int main() {
    bool ok = true;
    for (const auto& e : kExpectations) {
        uint32_t pc = 0, pt = 0, pi = 0;
        tt::foil::risc_to_hal_indices(e.id, pc, pt, pi);
        std::printf("  %-6s class=%u type=%u idx=%u  (expect %u/%u/%u)\n",
                    e.name, pc, pt, pi, e.proc_class, e.proc_type, e.proc_idx);
        if (pc != e.proc_class || pt != e.proc_type || pi != e.proc_idx) {
            std::fprintf(stderr, "  %s: mapping mismatch\n", e.name);
            ok = false;
        }
    }
    if (!ok) { std::puts("test_risc_ids: FAIL"); return 1; }
    std::puts("test_risc_ids: PASS");
    return 0;
}
