// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for ELF loading — no hardware required.
// Verifies that ll_api::memory can parse and XIP-transform a pre-built ELF.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "llrt/tt_memory.h"   // ll_api::memory (from tt-metal submodule)

int main(int argc, char* argv[]) {
    // Accept ELF path as argument; otherwise look for a default test binary.
    std::string elf_path;
    if (argc >= 2) {
        elf_path = argv[1];
    } else {
        // Default: look for a firmware ELF in the firmware directory.
        const char* fw_dir = std::getenv("TT_FOIL_FW_DIR");
        if (!fw_dir) {
            std::puts("test_elf_load: set TT_FOIL_FW_DIR or pass ELF path as argument");
            std::puts("test_elf_load: SKIP (no ELF provided)");
            return 0;
        }
        elf_path = std::string(fw_dir) + "/brisc.elf";
    }

    // Load ELF with CONTIGUOUS_XIP mode (same as used by tt-foil at runtime).
    ll_api::memory mem(elf_path, ll_api::memory::Loading::CONTIGUOUS_XIP);

    // Basic sanity checks on the loaded binary.
    assert(mem.size() > 0 && "ELF loaded but has no data");
    assert(mem.get_text_size() > 0 && "ELF has no text section");

    // Verify process_spans iterates at least one span.
    int span_count = 0;
    mem.process_spans([&](std::vector<uint32_t>::const_iterator /*mem_ptr*/,
                          uint64_t addr, uint32_t len_words) {
        assert(len_words > 0 && "span has zero length");
        ++span_count;
    });
    assert(span_count > 0 && "ELF has no loadable spans");

    std::printf("test_elf_load: loaded %s — %u words, %d span(s): PASS\n",
                elf_path.c_str(),
                static_cast<unsigned>(mem.size()),
                span_count);
    return 0;
}
