// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 1): firmware ELF discovery.
//
// CreateDevice() resolves the per-device firmware dir via
// BuildEnvManager::compute_build_key() (a hash of dispatch_core_type +
// num_hw_cqs + harvesting_mask + compile_hash_string). Re-implementing that
// hash from outside tt-metal would couple us to the internal hasher; instead
// we resolve the firmware dir by one of two mechanisms:
//
//   1. Explicit env var TT_FOIL_FIRMWARE_DIR=/path/to/pre-compiled/<hash>
//   2. Auto-discover under <tt-metal>/tt_metal/pre-compiled/ — pick the most
//      recently modified subdirectory that has the full set of RISC ELFs
//      (brisc/brisc_weakened.elf, ncrisc/ncrisc_weakened.elf, trisc{0,1,2}/
//      trisc{0,1,2}.elf).
//
// For Blackhole single-Tensix slow-dispatch we only need these 5 ELFs.
// erisc / idle_erisc / active_erisc / drisc are skipped intentionally.

#pragma once

#include <string>

namespace tt::foil {

// Five RISCs per Tensix core on Blackhole.
enum class FwRisc {
    BRISC,
    NCRISC,
    TRISC0,
    TRISC1,
    TRISC2,
};

struct FirmwarePaths {
    // Absolute path to the pre-compiled/<hash>/ root.
    std::string root;

    // Absolute paths to each RISC's firmware ELF.
    std::string brisc;    // brisc/brisc_weakened.elf
    std::string ncrisc;   // ncrisc/ncrisc_weakened.elf
    std::string trisc0;   // trisc0/trisc0.elf
    std::string trisc1;   // trisc1/trisc1.elf
    std::string trisc2;   // trisc2/trisc2.elf
};

// Resolve the firmware dir + per-RISC ELF paths.
//
// `tt_metal_root` is the tt-metal source root (e.g. /home/.../tt-metal); the
// auto-discovery scans <tt_metal_root>/tt_metal/pre-compiled/.
//
// Throws std::runtime_error if no suitable firmware dir is found.
FirmwarePaths resolve_firmware_paths(const std::string& tt_metal_root);

// Get the ELF path for one RISC from an already-resolved FirmwarePaths.
const std::string& firmware_elf(const FirmwarePaths& paths, FwRisc risc);

}  // namespace tt::foil
