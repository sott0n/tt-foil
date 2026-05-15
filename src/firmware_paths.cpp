// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "firmware_paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt::foil {

namespace fs = std::filesystem;

namespace {

// Required ELFs relative to the pre-compiled/<hash>/ root.
// tt-metal's BuildEnvManager picks "<risc>/<risc>.elf" (see build.cpp:429).
// `*_weakened.elf` exists too but is the input to user-kernel linking, not the
// firmware. They differ in a few words (relocations resolved differently),
// so picking the wrong one looks "almost right" but mismatches at relocation
// sites.
constexpr const char* kRelBrisc  = "brisc/brisc.elf";
constexpr const char* kRelNcrisc = "ncrisc/ncrisc.elf";
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

// Pick the most-recently-modified subdir of `scan` that contains a full
// RISC firmware ELF set, or std::nullopt if none.
static std::optional<fs::path> newest_full_set(const fs::path& scan) {
    if (!fs::is_directory(scan)) return std::nullopt;
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
    return found ? std::optional<fs::path>{best} : std::nullopt;
}

FirmwarePaths resolve_firmware_paths(const std::string& tt_metal_root) {
    // 1. Explicit env var wins.
    if (const char* env = std::getenv("TT_FOIL_FIRMWARE_DIR"); env && *env) {
        fs::path root{env};
        if (!has_full_set(root)) {
            throw std::runtime_error(
                "tt-foil: TT_FOIL_FIRMWARE_DIR=" + std::string{env} +
                " does not contain the expected RISC firmware ELFs "
                "(brisc/brisc.elf, ncrisc/ncrisc.elf, "
                "trisc{0,1,2}/trisc{0,1,2}.elf)");
        }
        return build_paths(root);
    }

    // 2. tt-foil's own firmware build artifact (Plan L).  When the
    //    TT_FOIL_BUILD_FIRMWARE CMake option is on (default), the build
    //    produces a self-contained <build>/firmware/ tree from the tt-metal
    //    source — no tt-metal runtime invocation required.  This is the
    //    preferred source going forward; the JIT-cache and pre-compiled
    //    fallbacks below exist for environments that opt out of the
    //    self-build or want to pin firmware externally.
    if (const char* env = std::getenv("TT_FOIL_BUILD_FIRMWARE_DIR"); env && *env) {
        fs::path root{env};
        if (has_full_set(root)) return build_paths(root);
    }
#ifdef TT_FOIL_FIRMWARE_BUILD_DIR
    {
        fs::path root{TT_FOIL_FIRMWARE_BUILD_DIR};
        if (has_full_set(root)) return build_paths(root);
    }
#endif

    // 3. Prefer tt-metal's JIT firmware cache. The pre-compiled
    //    tt_metal/pre-compiled/ tree can diverge from what tt-metal actually
    //    loads at runtime (different build hash → different brisc.elf bytes).
    //    Loading stale pre-compiled firmware leaves BRISC's setup_local_cb
    //    writes silently dropped, which hangs cb_reserve_back in the
    //    next kernel. Prefer ~/.cache/tt-metal-cache/<hash>/firmware/ which
    //    matches the firmware tt-metal builds and uses.
    if (const char* home = std::getenv("HOME"); home && *home) {
        fs::path cache_root = fs::path{home} / ".cache" / "tt-metal-cache";
        if (fs::is_directory(cache_root)) {
            // Each cache key is a subdir; firmware lives under <key>/firmware/.
            fs::path best;
            fs::file_time_type best_mtime{};
            bool found = false;
            for (const auto& entry : fs::directory_iterator{cache_root}) {
                if (!entry.is_directory()) continue;
                fs::path candidate = entry.path() / "firmware";
                if (!has_full_set(candidate)) continue;
                auto mtime = fs::last_write_time(candidate);
                if (!found || mtime > best_mtime) {
                    best = candidate;
                    best_mtime = mtime;
                    found = true;
                }
            }
            if (found) return build_paths(best);
        }
    }

    // 3. Fallback: auto-discover under tt_metal/pre-compiled/. Pick the
    //    most recently modified candidate with the full ELF set. This may
    //    be stale relative to what tt-metal builds; if it doesn't work,
    //    set TT_FOIL_FIRMWARE_DIR explicitly.
    fs::path scan = fs::path{tt_metal_root} / "tt_metal" / "pre-compiled";
    if (auto best = newest_full_set(scan); best.has_value()) {
        return build_paths(*best);
    }
    throw std::runtime_error(
        "tt-foil: no firmware ELFs found. Searched: "
        "$HOME/.cache/tt-metal-cache/<hash>/firmware/ and "
        + scan.string() +
        "/<hash>/. Set TT_FOIL_FIRMWARE_DIR to override.");
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
