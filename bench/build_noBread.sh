#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build the "no-Bread" profiling variant of the matmul reader: identical CB
# protocol to ops/matmul/reader.cpp but SKIPS the B-tile DRAM reads. Used by
# bench/matmul_bench to isolate the DRAM cost of streaming the weight matrix
# (see bench/reader_noBread.cpp). The other 4 ELFs (writer + 3 TRISC) are
# copied unchanged from ops/matmul/prebuilt — so build that first:
#   TT_METAL_ROOT=third_party/tt-metal bash ops/matmul/build.sh
#
# Output: bench/prebuilt_noBread/{reader.brisc,writer.ncrisc,
#         matmul.trisc0,matmul.trisc1,matmul.trisc2}.elf
# Normally invoked by bench/run.sh; run directly with:
#   TT_METAL_ROOT=third_party/tt-metal bash bench/build_noBread.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TT="${TT_METAL_ROOT:-$REPO/third_party/tt-metal}"
[[ "$TT" = /* ]] || TT="$REPO/$TT"
GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"
FW="${TT_METAL_PRECOMPILED:-$REPO/build/firmware}"
ORIG="$REPO/ops/matmul/prebuilt"
PREBUILT="$HERE/prebuilt_noBread"
BUILD="${BUILD:-/tmp/tt_foil_build_mmbench}"
SRCREADER="$HERE/reader_noBread.cpp"
mkdir -p "$BUILD" "$PREBUILT"
[[ -d "$FW/brisc" ]] || { echo "firmware not found at $FW (build tt-foil first)"; exit 1; }
[[ -f "$ORIG/writer.ncrisc.elf" ]] || { echo "build ops/matmul first: $ORIG missing"; exit 1; }

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
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_api"
    -I"$TT/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
    -I"$TT/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    -I"$TT/tt_metal/tt-llk/common"
    -I"$TT/runtime/sfpi/include"
    -I"$TT/tt_metal/hw/firmware/src/tt-1xx"
    -I"$TT/tt_metal/hostdevcommon/api"
    -I"$TT/tt_metal/api"
    -I"$TT/tt_metal/api/tt-metalium"
)
echo "#include \"$SRCREADER\"" > "$BUILD/kernel_includes.hpp"
obj="$BUILD/brisck_reader.o"; elf="$PREBUILT/reader.brisc.elf"
"$GXX" "${COMMON_CFLAGS[@]}" -DCOMPILE_FOR_BRISC -DPROCESSOR_INDEX=0 \
    -c "$TT/tt_metal/hw/firmware/src/tt-1xx/brisck.cc" -o "$obj"
"$GXX" -Os -mcpu=tt-bh -fno-tree-loop-distribute-patterns \
    -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
    -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
    -Wl,--emit-relocs -Wl,--just-symbols="$FW/brisc/brisc_weakened.elf" \
    -T"$LDDIR/kernel_brisc.ld" "$obj" "$LIB/noc.o" "$LIB/substitutes.o" -o "$elf"
echo "built variant reader: $elf"
cp "$ORIG/writer.ncrisc.elf" "$ORIG/matmul.trisc0.elf" \
   "$ORIG/matmul.trisc1.elf" "$ORIG/matmul.trisc2.elf" "$PREBUILT/"
echo "copied writer + 3 TRISC ELFs from $ORIG -> $PREBUILT"
