#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for examples/*/build_kernels.sh and models/*/build_kernels.sh.
# Source this from a builder script:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/../../scripts/kernel_build_helpers.sh"
#     if kernels_up_to_date "$PREBUILT" "$TT_METAL_PRECOMPILED" "$HERE/kernels"; then
#         echo "$0: kernels up-to-date, skipping rebuild"; exit 0
#     fi
#     ... do the build ...
#     touch "$PREBUILT/.build_stamp"
#
# Rationale: switching between profiler-on/off firmware silently leaves
# kernel ELFs linked against the old *_weakened.elf, which causes
# cb_reserve_back hangs at runtime (CLAUDE.md invariant). Detecting
# firmware-newer-than-kernels lets us auto-rebuild instead of requiring
# a manual `rm -rf prebuilt/` between profiler toggles.

# kernels_up_to_date PREBUILT FW_PRECOMPILED KERNEL_SRC_DIR
#   PREBUILT          where the ELFs live (e.g., examples/foo/prebuilt)
#   FW_PRECOMPILED    directory containing brisc/brisc_weakened.elf etc.
#   KERNEL_SRC_DIR    directory containing the kernel .cpp sources
#
# Returns 0 (truthy) if ALL of:
#   - PREBUILT exists and contains at least one .elf
#   - The firmware reference (brisc_weakened.elf) is NOT newer than every ELF
#   - No kernel .cpp is newer than every ELF
#   - The env fingerprint (TT_FOIL_PROFILE_KERNEL value) matches the stamp
# Otherwise returns 1 → caller should rebuild.
kernels_up_to_date() {
    local prebuilt="$1"
    local fw_dir="$2"
    local src_dir="$3"

    # No prebuilt at all → must build.
    [[ -d "$prebuilt" ]] || return 1
    local first_elf
    first_elf=$(find "$prebuilt" -maxdepth 1 -name "*.elf" -print -quit 2>/dev/null)
    [[ -n "$first_elf" ]] || return 1

    # Firmware reference newer than any ELF → must rebuild.
    local fw_ref="$fw_dir/brisc/brisc_weakened.elf"
    if [[ -e "$fw_ref" ]]; then
        for elf in "$prebuilt"/*.elf; do
            [[ "$fw_ref" -nt "$elf" ]] && return 1
        done
    fi

    # Any kernel source newer than any ELF → must rebuild.
    if [[ -d "$src_dir" ]]; then
        for src in "$src_dir"/*.cpp "$src_dir"/*.h "$src_dir"/*.hpp; do
            [[ -e "$src" ]] || continue
            for elf in "$prebuilt"/*.elf; do
                [[ "$src" -nt "$elf" ]] && return 1
            done
        done
    fi

    # Env fingerprint check. TT_FOIL_PROFILE_KERNEL toggles whether the
    # kernels were built against profiler-enabled firmware. Don't reuse
    # ELFs built under a different setting.
    local stamp="$prebuilt/.build_stamp"
    local current="profile=${TT_FOIL_PROFILE_KERNEL:-}"
    if [[ -f "$stamp" ]]; then
        local prev
        prev=$(cat "$stamp")
        [[ "$prev" == "$current" ]] || return 1
    else
        # No stamp yet → assume stale.
        return 1
    fi

    return 0
}

# Write the env fingerprint after a successful build. Caller should
# invoke this at the END of build_kernels.sh.
stamp_kernel_build() {
    local prebuilt="$1"
    echo "profile=${TT_FOIL_PROFILE_KERNEL:-}" > "$prebuilt/.build_stamp"
}

# write_kernel_manifest PREBUILT FW_DIR SRC_DIR [extra_files...]
#   PREBUILT       directory the ELFs (and the manifest) live in
#   FW_DIR         firmware directory containing <risc>/<risc>_weakened.elf
#   SRC_DIR        directory containing kernel .cpp/.h/.hpp sources
#   extra_files... additional source files to hash (e.g. the build script)
#
# Writes PREBUILT/manifest.txt in a flat key=value format:
#     version=1
#     built_at=<iso8601>
#     git_sha=<short>
#     git_dirty=0|1
#     firmware_dir=<abs path>
#     fw:<basename>=<sha256_hex>
#     src:<relpath>=<sha256_hex>
#     elf:<basename>=<sha256_hex>
#     env:TT_FOIL_PROFILE_KERNEL=<value>
#
# Read by src/kernel_manifest.cpp at kernel_load time to detect ELFs
# whose firmware reference or source no longer matches the chip's current
# firmware / repo state — preventing the "ELF linked against old
# firmware -> cb_reserve_back hang" class of bug.
write_kernel_manifest() {
    local prebuilt="$1"
    local fw_dir="$2"
    local src_dir="$3"
    shift 3
    local extra=("$@")

    local manifest="$prebuilt/manifest.txt"
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    {
        echo "version=1"
        echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if git -C "$repo_root" rev-parse --short HEAD >/dev/null 2>&1; then
            echo "git_sha=$(git -C "$repo_root" rev-parse --short HEAD)"
            if [[ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]]; then
                echo "git_dirty=1"
            else
                echo "git_dirty=0"
            fi
        fi
        echo "firmware_dir=$fw_dir"

        # Firmware weakened ELFs (only the ones that exist — TRISCs may
        # not be present for BRISC/NCRISC-only kernels).
        for risc in brisc ncrisc trisc0 trisc1 trisc2; do
            local fwelf="$fw_dir/$risc/${risc}_weakened.elf"
            if [[ -e "$fwelf" ]]; then
                echo "fw:${risc}_weakened.elf=$(sha256sum "$fwelf" | awk '{print $1}')"
            fi
        done

        # Kernel sources.
        if [[ -d "$src_dir" ]]; then
            for src in "$src_dir"/*.cpp "$src_dir"/*.cc "$src_dir"/*.h "$src_dir"/*.hpp; do
                [[ -e "$src" ]] || continue
                local rel
                rel="$(realpath --relative-to="$prebuilt" "$src")"
                echo "src:${rel}=$(sha256sum "$src" | awk '{print $1}')"
            done
        fi

        # Extra files (typically the build script itself).
        for f in "${extra[@]}"; do
            [[ -e "$f" ]] || continue
            local rel
            rel="$(realpath --relative-to="$prebuilt" "$f")"
            echo "src:${rel}=$(sha256sum "$f" | awk '{print $1}')"
        done

        # Output ELFs.
        for elf in "$prebuilt"/*.elf; do
            [[ -e "$elf" ]] || continue
            echo "elf:$(basename "$elf")=$(sha256sum "$elf" | awk '{print $1}')"
        done

        echo "env:TT_FOIL_PROFILE_KERNEL=${TT_FOIL_PROFILE_KERNEL:-}"
    } > "$manifest"
}
