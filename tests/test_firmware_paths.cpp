// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for firmware ELF discovery (no hardware required).
// Verifies that resolve_firmware_paths() can locate a usable pre-compiled
// firmware dir under the configured tt-metal root.
//
// Usage:
//   TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./test_firmware_paths
// or
//   TT_FOIL_FIRMWARE_DIR=/path/to/pre-compiled/<hash> ./test_firmware_paths

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "firmware_paths.hpp"

int main() try {
    const char* root_env = std::getenv("TT_METAL_RUNTIME_ROOT");
    std::string tt_metal_root = root_env ? root_env : "/home/kyamaguchi/tt-metal";

    auto paths = tt::foil::resolve_firmware_paths(tt_metal_root);

    std::printf("test_firmware_paths: root   = %s\n", paths.root.c_str());
    std::printf("test_firmware_paths: brisc  = %s\n", paths.brisc.c_str());
    std::printf("test_firmware_paths: ncrisc = %s\n", paths.ncrisc.c_str());
    std::printf("test_firmware_paths: trisc0 = %s\n", paths.trisc0.c_str());
    std::printf("test_firmware_paths: trisc1 = %s\n", paths.trisc1.c_str());
    std::printf("test_firmware_paths: trisc2 = %s\n", paths.trisc2.c_str());

    namespace fs = std::filesystem;
    assert(fs::exists(paths.brisc)  && "brisc ELF missing");
    assert(fs::exists(paths.ncrisc) && "ncrisc ELF missing");
    assert(fs::exists(paths.trisc0) && "trisc0 ELF missing");
    assert(fs::exists(paths.trisc1) && "trisc1 ELF missing");
    assert(fs::exists(paths.trisc2) && "trisc2 ELF missing");

    // Check non-empty.
    for (auto risc : {tt::foil::FwRisc::BRISC,  tt::foil::FwRisc::NCRISC,
                      tt::foil::FwRisc::TRISC0, tt::foil::FwRisc::TRISC1,
                      tt::foil::FwRisc::TRISC2}) {
        auto sz = fs::file_size(tt::foil::firmware_elf(paths, risc));
        assert(sz > 0 && "firmware ELF is empty");
    }

    std::puts("test_firmware_paths: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_firmware_paths: FAIL — %s\n", e.what());
    return 1;
}
