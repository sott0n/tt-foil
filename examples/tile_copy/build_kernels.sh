#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build pre-compiled BRISC + NCRISC + TRISC kernel ELFs for the tile_copy
# example. Adds a build_compute() helper on top of the BRISC/NCRISC build
# pattern that compiles a single compute kernel source 3 times — once each
# for UNPACK (TRISC0), MATH (TRISC1), and PACK (TRISC2).
#
# Required env vars:
#   TT_METAL_ROOT        Path to a built tt-metal source tree (default: /home/kyamaguchi/tt-metal)
#   TT_METAL_PRECOMPILED Path to firmware *_weakened.elf objects
#                        (default: $TT_METAL_ROOT/tt_metal/pre-compiled/<hash>)
#
# Produces:
#   prebuilt/compute.trisc0.elf  (UNPACK variant)
#   prebuilt/compute.trisc1.elf  (MATH variant)
#   prebuilt/compute.trisc2.elf  (PACK variant)
# (reader/writer BRISC+NCRISC ELFs land in v4-5)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT="${TT_METAL_ROOT:-/home/kyamaguchi/tt-metal}"
GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"

# Resolve the precompiled firmware directory automatically if not specified.
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
    -I"$TT/tt_metal/tt-llk/common"
    -I"$TT/runtime/sfpi/include"
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

# --- TRISC compute build -------------------------------------------------
#
# trisck.cc is the firmware wrapper for all 3 TRISCs. It #includes
# "chlkc_list.h", which tt-metal's JIT pipeline normally generates per
# kernel. For tt-foil's pre-compiled flow we provide a hand-written stub
# version on the -I path that wires kernel_main() into trisc's run_kernel()
# without pulling in any chlkc_{unpack,math,pack}.cpp generated files.
#
# The TRISC variant (UNPACK/MATH/PACK) is selected at compile time via two
# parallel define families:
#   TRISC_UNPACK / TRISC_MATH / TRISC_PACK    used by tt-metal's
#                                              compute_kernel_api headers
#   UCK_CHLKC_UNPACK / UCK_CHLKC_MATH / UCK_CHLKC_PACK  used by the
#                                              firmware wrappers + LLK
build_compute() {
    local src="$1"        # path to compute kernel .cpp
    local out_name="$2"   # base name; produces <out_name>.trisc{0,1,2}.elf

    # kernel_includes.hpp pulls the user kernel into trisck.cc's TU.
    local kernel_includes="$BUILD/kernel_includes.hpp"
    echo "#include \"$src\"" > "$kernel_includes"

    # Stub chlkc_list.h on -I path: minimal replacement for tt-metal's JIT-
    # generated chlkc_list.h. Pulls in the user kernel via kernel_includes
    # (so kernel_main() resolves) and provides run_kernel() — that's the
    # entire contract trisck.cc relies on.
    local chlkc_stub="$BUILD/chlkc_list.h"
    cat > "$chlkc_stub" <<'EOF'
#pragma once
#include <cstdint>
#include "kernel_includes.hpp"
inline std::uint32_t run_kernel() {
    kernel_main();
    return 0;
}
EOF

    local variants=(unpack math pack)
    for i in 0 1 2; do
        local var=${variants[$i]}
        local VAR=${var^^}                    # UPPER
        local proc_idx=$((i + 2))             # TRISC0/1/2 → 2/3/4
        local obj="$BUILD/trisck_${out_name}_${var}.o"
        local elf="$PREBUILT/${out_name}.trisc${i}.elf"
        local risc="trisc${i}"

        "$GXX" "${COMMON_CFLAGS[@]}" \
            -DCOMPILE_FOR_TRISC=$i \
            -DUCK_CHLKC_${VAR} \
            -DTRISC_${VAR} \
            -DNAMESPACE=chlkc_${var} \
            -DPROCESSOR_INDEX=$proc_idx \
            -c "$TT/tt_metal/hw/firmware/src/tt-1xx/trisck.cc" \
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
    done
}

build_compute "$HERE/kernels/compute.cpp" compute
