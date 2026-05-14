// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "firmware_paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt::foil {

namespace fs = std::filesystem;

namespace {

// Required ELFs relative to the pre-compiled/<hash>/ root.
constexpr const char* kRelBrisc  = "brisc/brisc_weakened.elf";
constexpr const char* kRelNcrisc = "ncrisc/ncrisc_weakened.elf";
constexpr const char* kRelTrisc0 = "trisc0/trisc0.elf";
constexpr const char* kRelTrisc1 = "trisc1/trisc1.elf";
constexpr const char* kRelTrisc2 = "trisc2/trisc2.elf";

bool has_full_set(const fs::path& dir) {
    return fs::exists(dir / kRelBrisc) && fs::exists(dir / kRelNcrisc) &&
           fs::exists(dir / kRelTrisc0) && fs::exists(dir / kRelTrisc1) &&
           fs::exists(dir / kRelTrisc2);
}

FirmwarePaths build_paths(const fs::path& root) {
    FirmwarePaths p;
    p.root   = root.string();
    p.brisc  = (root / kRelBrisc).string();
    p.ncrisc = (root / kRelNcrisc).string();
    p.trisc0 = (root / kRelTrisc0).string();
    p.trisc1 = (root / kRelTrisc1).string();
    p.trisc2 = (root / kRelTrisc2).string();
    return p;
}

}  // namespace

FirmwarePaths resolve_firmware_paths(const std::string& tt_metal_root) {
    // 1. Explicit env var wins.
    if (const char* env = std::getenv("TT_FOIL_FIRMWARE_DIR"); env && *env) {
        fs::path root{env};
        if (!has_full_set(root)) {
            throw std::runtime_error(
                "tt-foil: TT_FOIL_FIRMWARE_DIR=" + std::string{env} +
                " does not contain the expected RISC firmware ELFs "
                "(brisc/brisc_weakened.elf, ncrisc/ncrisc_weakened.elf, "
                "trisc{0,1,2}/trisc{0,1,2}.elf)");
        }
        return build_paths(root);
    }

    // 2. Auto-discover under tt_metal/pre-compiled/. Pick the most recently
    //    modified candidate with the full ELF set — that's the directory the
    //    current tt-metal build is using (it was touched last).
    fs::path scan = fs::path{tt_metal_root} / "tt_metal" / "pre-compiled";
    if (!fs::is_directory(scan)) {
        throw std::runtime_error(
            "tt-foil: no firmware: TT_FOIL_FIRMWARE_DIR unset and "
            + scan.string() + " is not a directory");
    }

    fs::path best;
    fs::file_time_type best_mtime{};
    bool found = false;
    for (const auto& entry : fs::directory_iterator{scan}) {
        if (!entry.is_directory()) continue;
        if (!has_full_set(entry.path())) continue;
        auto mtime = fs::last_write_time(entry.path());
        if (!found || mtime > best_mtime) {
            best = entry.path();
            best_mtime = mtime;
            found = true;
        }
    }
    if (!found) {
        throw std::runtime_error(
            "tt-foil: no pre-compiled firmware dir with full RISC ELF set "
            "found under " + scan.string() +
            " (set TT_FOIL_FIRMWARE_DIR to override)");
    }
    return build_paths(best);
}

const std::string& firmware_elf(const FirmwarePaths& paths, FwRisc risc) {
    switch (risc) {
        case FwRisc::BRISC:  return paths.brisc;
        case FwRisc::NCRISC: return paths.ncrisc;
        case FwRisc::TRISC0: return paths.trisc0;
        case FwRisc::TRISC1: return paths.trisc1;
        case FwRisc::TRISC2: return paths.trisc2;
    }
    throw std::runtime_error("tt-foil: firmware_elf: unknown FwRisc");
}

}  // namespace tt::foil
