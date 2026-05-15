// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v5-4 smoke test: host writes a buffer into DRAM, reads it back, asserts
// equality. Validates that BufferLocation::DRAM + write_buffer/read_buffer
// work end-to-end before we put a kernel between them.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});

    // Use a 32-KB block — bigger than any L1 tile we'd test, but tiny in
    // DRAM terms. Deterministic pattern: index * 0x10001.
    constexpr std::size_t kBytes = 32 * 1024;
    constexpr std::size_t kWords = kBytes / sizeof(uint32_t);
    std::vector<uint32_t> src(kWords), got(kWords, 0);
    for (std::size_t i = 0; i < kWords; ++i) src[i] = static_cast<uint32_t>(i * 0x10001u);

    tt::foil::CoreCoord any_core{0, 0};   // unused for DRAM allocs
    auto buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kBytes, any_core);
    tt::foil::write_buffer(*dev, *buf, src.data(), kBytes);
    tt::foil::read_buffer (*dev, *buf, got.data(), kBytes);

    uint32_t mismatches = 0, first_bad = kWords;
    for (std::size_t i = 0; i < kWords; ++i) {
        if (got[i] != src[i]) {
            if (first_bad == kWords) first_bad = i;
            ++mismatches;
        }
    }
    if (mismatches) {
        std::fprintf(stderr,
            "test_dram_roundtrip: %u/%zu mismatches; first at word %u: got=0x%08x expected=0x%08x\n",
            mismatches, kWords, first_bad, got[first_bad], src[first_bad]);
        tt::foil::close_device(std::move(dev));
        std::puts("test_dram_roundtrip: FAIL");
        return 1;
    }

    tt::foil::close_device(std::move(dev));
    std::printf("test_dram_roundtrip: PASS  (%zu B written and read back at DRAM offset 0x%llx)\n",
                kBytes, static_cast<unsigned long long>(buf->device_addr));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_dram_roundtrip: FAIL — %s\n", e.what());
    return 1;
}
