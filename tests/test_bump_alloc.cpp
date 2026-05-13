// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for BumpAllocator — no hardware required.

#include <cassert>
#include <cstdio>
#include <stdexcept>

// Include device.hpp for L1Allocator definition (internal header)
#include "../src/device.hpp"

int main() {
    // ---- Basic allocation ----
    tt::foil::L1Allocator alloc{};
    alloc.base    = 0x1A200;
    alloc.current = 0x1A200;
    alloc.end     = 0x40000;

    uint64_t a1 = alloc.alloc(64);
    assert(a1 == 0x1A200);
    assert(alloc.current == 0x1A200 + 64);

    // ---- Alignment ----
    uint64_t a2 = alloc.alloc(3, /*alignment=*/16);
    // Should be aligned to 16 from current
    assert(a2 % 16 == 0);
    assert(a2 >= alloc.base);

    // ---- Back-to-back ----
    uint64_t a3 = alloc.alloc(256, /*alignment=*/16);
    assert(a3 > a2);

    // ---- Reset ----
    alloc.reset();
    assert(alloc.current == alloc.base);
    uint64_t a4 = alloc.alloc(64);
    assert(a4 == alloc.base);

    // ---- OOM ----
    bool threw = false;
    try {
        alloc.alloc(alloc.end - alloc.base + 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::puts("test_bump_alloc: PASS");
    return 0;
}
