#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build Embedding-lookup op kernel ELFs.
#
# Produces:
#   prebuilt/reader.brisc.elf
#   prebuilt/writer.ncrisc.elf

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT="${TT_METAL_ROOT:-/home/kyamaguchi/tt-metal}"
GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"

if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    REPO_ROOT="$(cd "$HERE/../.." && pwd)"
    if [[ -d "$REPO_ROOT/build/firmware/brisc" ]]; then
        TT_METAL_PRECOMPILED="$REPO_ROOT/build/firmware"
    fi
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    GIT_COMMON=$(git -C "$HERE" rev-parse --git-common-dir 2>/dev/null || true)
    if [[ -n "$GIT_COMMON" ]]; then
        MAIN_REPO="$(cd "$(dirname "$GIT_COMMON")" && pwd)"
        if [[ -d "$MAIN_REPO/build/firmware/brisc" ]]; then
            TT_METAL_PRECOMPILED="$MAIN_REPO/build/firmware"
        fi
    fi
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(set +o pipefail; ls -1dt "$HOME"/.cache/tt-metal-cache/*/firmware 2>/dev/null | head -n1)
fi
[[ -d "$TT_METAL_PRECOMPILED" ]] || { echo "TT_METAL_PRECOMPILED not found"; exit 1; }
echo "build: using firmware from $TT_METAL_PRECOMPILED"

BUILD="${BUILD:-/tmp/tt_foil_build_embedding}"
PREBUILT="${PREBUILT_DIR:-$HERE/prebuilt}"
mkdir -p "$BUILD" "$PREBUILT"

COMMON_CFLAGS=(
    -std=c++17 -fno-exceptions -fno-use-cxa-atexit
    -Os -mcpu=tt-bh -fno-tree-loop-distribute-patterns
    -DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0
    -DKERNEL_BUILD -DDISPATCH_MESSAGE_ADDR=0
    -DNOC_INDEX=0 -DNOC_MODE=0
    -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3 -DLOG_BASE_2_OF_NUM_L1_BANKS=7
    -DPCIE_NOC_X=0 -DPCIE_NOC_Y=3
    -I"$BUILD"
    -I"$TT" -I"$TT/tt_metal" -I"$TT/tt_metal/hw/inc"
    -I"$TT/tt_metal/hw/inc/internal" -I"$TT/tt_metal/hw/inc/internal/tt-1xx"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
    -I"$TT/tt_metal/hw/inc/api" -I"$TT/tt_metal/hw/inc/api/dataflow"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/common"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_io"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    -I"$TT/tt_metal/hw/firmware/src/tt-1xx"
    -I"$TT/tt_metal/hostdevcommon/api"
    -I"$TT/tt_metal/api"
)

build_one() {
    local risc="$1" proc_idx="$2" src="$3" out_name="$4"
    echo "#include \"$src\"" > "$BUILD/kernel_includes.hpp"
    local obj="$BUILD/${risc}k_${out_name}.o"
    local elf="$PREBUILT/${out_name}.elf"
    "$GXX" "${COMMON_CFLAGS[@]}" \
        -DCOMPILE_FOR_${risc^^} -DPROCESSOR_INDEX=$proc_idx \
        -c "$TT/tt_metal/hw/firmware/src/tt-1xx/${risc}k.cc" -o "$obj"
    "$GXX" -Os -mcpu=tt-bh -fno-tree-loop-distribute-patterns \
        -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
        -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
        -Wl,--emit-relocs \
        -Wl,--just-symbols="$TT_METAL_PRECOMPILED/${risc}/${risc}_weakened.elf" \
        -T"$LDDIR/kernel_${risc}.ld" "$obj" "$LIB/noc.o" "$LIB/substitutes.o" -o "$elf"
    echo "built: $elf"
}

build_one brisc  0 "$HERE/reader.cpp" reader.brisc
build_one ncrisc 1 "$HERE/writer.cpp" writer.ncrisc

# Stale-ELF guard: record firmware + source sha256s for the runtime check.
# See scripts/kernel_build_helpers.sh::write_kernel_manifest and
# src/kernel_manifest.cpp (TT_FOIL_VERIFY_MANIFEST=1).
bash "$HERE/../../scripts/write_manifest.sh" "$PREBUILT" \
    "$TT_METAL_PRECOMPILED" "$HERE" "${BASH_SOURCE[0]}"
