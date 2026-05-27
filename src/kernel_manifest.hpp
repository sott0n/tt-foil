// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Manifest-based staleness check for pre-compiled kernel ELFs.
//
// build_kernels.sh / ops/*/build.sh write a `manifest.txt` next to each
// prebuilt/ ELF (see scripts/kernel_build_helpers.sh::write_kernel_manifest).
// The manifest records sha256 of the firmware *_weakened.elf and kernel
// sources the ELFs were linked against, plus sha256 of the ELFs themselves.
//
// This check verifies that the ELFs about to be loaded still match those
// recorded hashes, catching the "stale ELF linked against an older firmware
// → cb_reserve_back hang at runtime" class of bug.
//
// Default: no-op. Set TT_FOIL_VERIFY_MANIFEST=1 to enable. Set
// TT_FOIL_SKIP_MANIFEST_CHECK=1 to force bypass even when verify is on.

#pragma once

#include <span>

#include "firmware_paths.hpp"
#include "tt_foil/runtime.hpp"

namespace tt::foil {

void check_kernel_manifest(
    std::span<const RiscBinary> binaries,
    const FirmwarePaths& fw_paths);

}  // namespace tt::foil
