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

# Optional: device profiler. When TT_FOIL_PROFILE_KERNEL is non-empty (and
# numeric), pass it as -DPROFILE_KERNEL=<value> so tt-metal's
# kernel_profiler.hpp activates. Default off → no profiler in firmware.
#
# PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC is required by kernel_profiler.hpp
# but only used on the DRAM-offload "quick_push" code path which tt-foil
# does not exercise (we read the L1 buffer directly post-dispatch). A
# small fixed value keeps the constexpr expressions consumable; the actual
# DRAM region this implies is never allocated.
PROFILE_DEFINES=()
if [[ -n "${TT_FOIL_PROFILE_KERNEL:-}" ]]; then
    PROFILE_DEFINES=(
        -DPROFILE_KERNEL="${TT_FOIL_PROFILE_KERNEL}"
        -DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=4096
    )
fi

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
    # Firmware build: internal/ low-level headers MUST precede api/dataflow so
    # that `#include "noc.h"` (e.g. from brisc.cc) resolves to the firmware
    # low-level internal/.../blackhole/noc/noc.h and NOT api/dataflow/noc.h.
    # The latter is kernel-only: it pulls dataflow_api.h, which #errors under
    # `#if !defined(KERNEL_BUILD)`. api/dataflow/noc.h was added in tt-metal
    # ~v0.71 (the umd v0.9.6 bump); before that api/dataflow had no noc.h, so
    # the -I order was harmless. Keep internal/ first.
    -I"$TT/tt_metal/hw/inc/internal" -I"$TT/tt_metal/hw/inc/internal/tt-1xx"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
    -I"$TT/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
    -I"$TT/tt_metal/hw/inc/api" -I"$TT/tt_metal/hw/inc/api/dataflow"
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

    # When profiler is on, BRISC firmware grows by ~1.2 KB and overflows
    # its 0x2200 (8.5 KB) region. tt-metal's JIT build uses -flto=auto
    # which inlines + drops dead code aggressively, shrinking BRISC FW
    # roughly in half. We mirror that here only when PROFILE_KERNEL is on
    # to avoid regressing the non-profile build.
    local lto_flags=()
    if [[ -n "${TT_FOIL_PROFILE_KERNEL:-}" ]]; then
        lto_flags=(-flto=auto -ffat-lto-objects)
    fi

    "$GXX" "${COMMON_CFLAGS[@]}" \
        "${PROFILE_DEFINES[@]}" \
        "${lto_flags[@]}" \
        -mcpu=tt-bh \
        -DCOMPILE_FOR_${risc^^} -DPROCESSOR_INDEX=$proc_idx \
        -c "$TT/tt_metal/hw/firmware/src/tt-1xx/${risc}.cc" \
        -o "$obj"

    "$GXX" \
        "${lto_flags[@]}" \
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

        # Same LTO trick as BRISC/NCRISC to keep the profiler-on text
        # section inside TRISC's tight region (limit 0xa00 for TRISC0).
        local trisc_lto=()
        if [[ -n "${TT_FOIL_PROFILE_KERNEL:-}" ]]; then
            trisc_lto=(-flto=auto -ffat-lto-objects)
        fi

        "$GXX" "${COMMON_CFLAGS[@]}" \
            "${PROFILE_DEFINES[@]}" \
            "${trisc_lto[@]}" \
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
            "${trisc_lto[@]}" \
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
