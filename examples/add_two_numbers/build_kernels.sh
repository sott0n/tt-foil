#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build pre-compiled BRISC and NCRISC kernel ELFs for the add_two_numbers example.
#
# Required env vars:
#   TT_METAL_ROOT        Path to a built tt-metal source tree (default: /home/kyamaguchi/tt-metal)
#   TT_METAL_PRECOMPILED Path to firmware *_weakened.elf objects
#                        (default: $TT_METAL_ROOT/tt_metal/pre-compiled/<hash>)
#
# Produces:
#   prebuilt/add_brisc.elf
#   prebuilt/add_ncrisc.elf

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT="${TT_METAL_ROOT:-/home/kyamaguchi/tt-metal}"
GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"

# Resolve the firmware directory containing *_weakened.elf objects. Prefer
# tt-metal's JIT cache (matches what tt-metal runtime loads) over the
# possibly-stale tt_metal/pre-compiled/ tree.
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(ls -1dt "$HOME"/.cache/tt-metal-cache/*/firmware 2>/dev/null | head -n1)
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(find "$TT/tt_metal/pre-compiled" -maxdepth 1 -mindepth 1 -type d | head -n1)
fi
[[ -d "$TT_METAL_PRECOMPILED" ]] || { echo "TT_METAL_PRECOMPILED not found"; exit 1; }

BUILD="${BUILD:-/tmp/tt_foil_build}"
PREBUILT="$HERE/prebuilt"
mkdir -p "$BUILD" "$PREBUILT"

# Common preprocessor + include flags shared by both RISC compiles.
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
    -I"$TT/tt_metal/hw/inc/api" -I"$TT/tt_metal/hw/inc/api/dataflow"
    -I"$TT/tt_metal/hw/inc/internal" -I"$TT/tt_metal/hw/inc/internal/tt-1xx"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/common"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_io"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    -I"$TT/tt_metal/hw/firmware/src/tt-1xx"
    -I"$TT/tt_metal/hostdevcommon/api"
    -I"$TT/tt_metal/api"
)

build_one() {
    local risc="$1"   # brisc or ncrisc
    local proc_idx="$2"
    local src="$3"
    local out_name="$4"

    # brisck.cc and ncrisck.cc both #include "kernel_includes.hpp".
    # Place that file in $BUILD (already on the -I path) and have it pull in
    # the user kernel source.
    local kernel_includes="$BUILD/kernel_includes.hpp"
    echo "#include \"$src\"" > "$kernel_includes"

    local obj="$BUILD/${risc}k_${out_name}.o"
    local elf="$PREBUILT/${out_name}.elf"

    "$GXX" "${COMMON_CFLAGS[@]}" \
        -DCOMPILE_FOR_${risc^^} -DPROCESSOR_INDEX=$proc_idx \
        -c "$TT/tt_metal/hw/firmware/src/tt-1xx/${risc}k.cc" \
        -o "$obj"

    "$GXX" \
        -Os -mcpu=tt-bh -fno-tree-loop-distribute-patterns \
        -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
        -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
        -Wl,--emit-relocs \
        -Wl,--just-symbols="$TT_METAL_PRECOMPILED/${risc}/${risc}_weakened.elf" \
        -T"$LDDIR/kernel_${risc}.ld" \
        "$obj" \
        "$LIB/noc.o" "$LIB/substitutes.o" \
        -o "$elf"

    echo "built: $elf"
}

build_one brisc  0 "$HERE/kernels/add_brisc.cpp"  add_brisc
build_one ncrisc 1 "$HERE/kernels/add_ncrisc.cpp" add_ncrisc
