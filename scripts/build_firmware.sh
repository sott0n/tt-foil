#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build the 5 RISC firmware ELFs (brisc, ncrisc, trisc0/1/2) plus their
# *_weakened.elf companions that kernels link against. This removes the
# "run tt-metal at least once" prerequisite — tt-foil compiles the firmware
# itself directly from the tt-metal source tree using SFPI g++.
#
# Required env vars (or arguments):
#   TT_METAL_ROOT    Path to a built tt-metal source tree (default:
#                    /home/kyamaguchi/tt-metal).  Build tree must contain
#                    libexec/tt-metalium/runtime/sfpi/compiler/bin/.
#   TT_FOIL_BUILD    tt-foil build directory holding tt_foil_weaken
#                    (default: <repo>/build).
#   OUT_DIR          Destination for the firmware tree (default:
#                    $TT_FOIL_BUILD/firmware).  Layout matches tt-metal's
#                    JIT cache: <risc>/<risc>.elf, <risc>/<risc>_weakened.elf.
#
# Mirrors the FW_BUILD path of tt-metal's JitBuildState::build() (see
# tt_metal/jit_build/build.cpp).  Per-RISC settings come from
# bh_hal.cpp::HalJitBuildQueryBlackHole::{srcs,link_objs,linker_script,common_flags}.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TT="${TT_METAL_ROOT:-/home/kyamaguchi/tt-metal}"
TT_FOIL_BUILD="${TT_FOIL_BUILD:-$REPO/build}"
OUT_DIR="${OUT_DIR:-$TT_FOIL_BUILD/firmware}"

GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"
WEAKEN="$TT_FOIL_BUILD/tools/tt_foil_weaken"

[[ -x "$GXX"    ]] || { echo "SFPI g++ not found at $GXX"; exit 1; }
[[ -x "$WEAKEN" ]] || { echo "tt_foil_weaken not found at $WEAKEN — build it first"; exit 1; }

mkdir -p "$OUT_DIR"

# Common preprocessor + include flags.  Same set as
# examples/*/build_kernels.sh except KERNEL_BUILD → FW_BUILD and the LLK
# include paths get added unconditionally so TRISC firmware compiles.  See
# build.cpp:130 (common cxx flags) + bh_hal.cpp:115 (includes).
COMMON_CFLAGS=(
    -std=c++17 -fno-exceptions -fno-use-cxa-atexit
    -Os -fno-tree-loop-distribute-patterns
    -DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0
    -DFW_BUILD -DDISPATCH_MESSAGE_ADDR=0
    -DNOC_INDEX=0 -DNOC_MODE=0
    -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3 -DLOG_BASE_2_OF_NUM_L1_BANKS=7
    -DPCIE_NOC_X=0 -DPCIE_NOC_Y=3
    -I"$TT" -I"$TT/tt_metal" -I"$TT/tt_metal/hw/inc"
    -I"$TT/tt_metal/hw/inc/api" -I"$TT/tt_metal/hw/inc/api/dataflow"
    -I"$TT/tt_metal/hw/inc/internal" -I"$TT/tt_metal/hw/inc/internal/tt-1xx"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/common"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_io"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_api"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    -I"$TT/tt_metal/tt-llk/common"
    -I"$TT/runtime/sfpi/include"
    -I"$TT/tt_metal/hw/firmware/src/tt-1xx"
    -I"$TT/tt_metal/hostdevcommon/api"
    -I"$TT/tt_metal/api"
)

# Build one DM firmware (BRISC, processor_index=0, or NCRISC, processor_index=1).
# BRISC firmware includes noc.o; NCRISC firmware does not (per bh_hal:link_objs).
build_dm() {
    local risc="$1"        # brisc or ncrisc
    local proc_idx="$2"    # 0 (brisc) or 1 (ncrisc)

    local odir="$OUT_DIR/${risc}"
    mkdir -p "$odir"
    local obj="$odir/${risc}.o"
    local elf="$odir/${risc}.elf"
    local welf="$odir/${risc}_weakened.elf"

    local link_noc=()
    if [[ "$risc" == "brisc" ]]; then
        link_noc=("$LIB/noc.o")
    fi

    "$GXX" "${COMMON_CFLAGS[@]}" \
        -mcpu=tt-bh \
        -DCOMPILE_FOR_${risc^^} -DPROCESSOR_INDEX=$proc_idx \
        -c "$TT/tt_metal/hw/firmware/src/tt-1xx/${risc}.cc" \
        -o "$obj"

    "$GXX" \
        -Os -mcpu=tt-bh -fno-tree-loop-distribute-patterns \
        -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
        -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
        -T"$LDDIR/firmware_${risc}.ld" \
        "$obj" \
        "$LIB/tmu-crt0.o" "${link_noc[@]}" "$LIB/substitutes.o" \
        -o "$elf"

    "$WEAKEN" "$elf" "$welf"
    echo "fw built: $elf  ($(stat -c%s "$elf") B)"
}

# Build all three TRISC firmware variants from trisc.cc.
build_trisc_fw() {
    local variants=(unpack math pack)
    for i in 0 1 2; do
        local var=${variants[$i]}
        local VAR=${var^^}
        local proc_idx=$((i + 2))      # TRISC0/1/2 → 2/3/4 per bh_hal
        local odir="$OUT_DIR/trisc${i}"
        mkdir -p "$odir"
        local obj="$odir/trisc.o"
        local elf="$odir/trisc${i}.elf"
        local welf="$odir/trisc${i}_weakened.elf"

        "$GXX" "${COMMON_CFLAGS[@]}" \
            -mcpu=tt-bh-tensix -O3 \
            -ffast-math \
            -ftt-nttp -ftt-constinit -ftt-consteval \
            -DCOMPILE_FOR_TRISC=$i \
            -DUCK_CHLKC_${VAR} \
            -DTRISC_${VAR} \
            -DNAMESPACE=chlkc_${var} \
            -DPROCESSOR_INDEX=$proc_idx \
            -c "$TT/tt_metal/hw/firmware/src/tt-1xx/trisc.cc" \
            -o "$obj"

        "$GXX" \
            -O3 -mcpu=tt-bh-tensix -ffast-math \
            -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
            -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
            -T"$LDDIR/firmware_trisc${i}.ld" \
            "$obj" \
            "$LIB/tmu-crt0.o" "$LIB/substitutes.o" \
            -o "$elf"

        "$WEAKEN" "$elf" "$welf"
        echo "fw built: $elf  ($(stat -c%s "$elf") B)"
    done
}

build_dm brisc  0
build_dm ncrisc 1
build_trisc_fw

echo "tt-foil firmware ready at: $OUT_DIR"
