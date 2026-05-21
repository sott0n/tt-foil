#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build MHA op kernel ELFs.
#
# Produces (in $PREBUILT_DIR):
#   reader.brisc.elf
#   writer.ncrisc.elf
#   mha.trisc0.elf  mha.trisc1.elf  mha.trisc2.elf

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
    TT_METAL_PRECOMPILED=$(ls -1dt "$HOME"/.cache/tt-metal-cache/*/firmware 2>/dev/null | head -n1)
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(find "$TT/tt_metal/pre-compiled" -maxdepth 1 -mindepth 1 -type d | head -n1)
fi
[[ -d "$TT_METAL_PRECOMPILED" ]] || { echo "TT_METAL_PRECOMPILED not found"; exit 1; }
echo "build: using firmware from $TT_METAL_PRECOMPILED"

BUILD="${BUILD:-/tmp/tt_foil_build_gqa_decode}"
PREBUILT="${PREBUILT_DIR:-$HERE/prebuilt}"
mkdir -p "$BUILD" "$PREBUILT"

COMMON_CFLAGS=(
    ${DIAG:+-DDIAG=$DIAG}
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

build_dataflow() {
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

build_compute() {
    local src="$1" out_name="$2"
    echo "#include \"$src\"" > "$BUILD/kernel_includes.hpp"

    # chlkc_list.h stub: CBs 0-7 (inputs/intermediates) and CB 16 (output) all BF16 (=5)
    # Binary + reduce + bcast ops require ckernel::MathFidelity enum.
    cat > "$BUILD/chlkc_list.h" <<'EOF'
#pragma once
#include <cstdint>
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf
constexpr bool APPROX = true;
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "llk_defs.h"
constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(4);  // HiFi4
#endif
// Float16_b (=5) for CB 0-7, 16; 255 for others
constexpr unsigned char pack_src_format[32] = {
    5,5,5,5,5,5,5,5,5,5,5,255,255,255,255,255,
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
constexpr unsigned char pack_dst_format[32] = {
    5,5,5,5,5,5,5,5,5,5,5,255,255,255,255,255,
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
constexpr std::int32_t unpack_src_format[32] = {
    5,5,5,5,5,5,5,5,5,5,5,255,255,255,255,255,
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
constexpr std::int32_t unpack_dst_format[32] = {
    5,5,5,5,5,5,5,5,5,5,5,255,255,255,255,255,
    5,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
constexpr std::uint8_t  pack_tile_num_faces[32]    = { 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 };
constexpr std::uint8_t  pack_partial_face[32]      = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  pack_tile_face_r_dim[32]   = { 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16 };
constexpr std::uint8_t  pack_narrow_tile[32]       = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  pack_tile_r_dim[32]        = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  pack_tile_c_dim[32]        = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint16_t pack_tile_size[32]         = { 1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088 };
constexpr std::uint8_t  pack_num_faces_r_dim[32]   = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  pack_num_faces_c_dim[32]   = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  unpack_tile_num_faces[32]   = { 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 };
constexpr std::uint8_t  unpack_partial_face[32]     = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  unpack_tile_face_r_dim[32]  = { 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16 };
constexpr std::uint8_t  unpack_narrow_tile[32]      = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  unpack_tile_r_dim[32]       = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  unpack_tile_c_dim[32]       = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint16_t unpack_tile_size[32]        = { 1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088,1088 };
constexpr std::uint8_t  unpack_num_faces_r_dim[32]  = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  unpack_num_faces_c_dim[32]  = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
#include "kernel_includes.hpp"
inline std::uint32_t run_kernel() { kernel_main(); return 0; }
EOF

    local variants=(unpack math pack)
    for i in 0 1 2; do
        local var=${variants[$i]} VAR=${variants[$i]^^}
        local proc_idx=$((i + 2))
        local obj="$BUILD/trisck_${out_name}_${var}.o"
        local elf="$PREBUILT/${out_name}.trisc${i}.elf"
        local risc="trisc${i}"
        "$GXX" "${COMMON_CFLAGS[@]}" \
            -mcpu=tt-bh-tensix -O3 -ffast-math \
            -ftt-nttp -ftt-constinit -ftt-consteval \
            -DCOMPILE_FOR_TRISC=$i \
            -DUCK_CHLKC_${VAR} -DTRISC_${VAR} \
            -DNAMESPACE=chlkc_${var} -DPROCESSOR_INDEX=$proc_idx \
            -c "$TT/tt_metal/hw/firmware/src/tt-1xx/trisck.cc" -o "$obj"
        "$GXX" -O3 -mcpu=tt-bh-tensix -ffast-math \
            -fno-exceptions -fno-use-cxa-atexit -std=c++17 \
            -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
            -Wl,--emit-relocs \
            -Wl,--just-symbols="$TT_METAL_PRECOMPILED/${risc}/${risc}_weakened.elf" \
            -T"$LDDIR/kernel_${risc}.ld" "$obj" "$LIB/noc.o" "$LIB/substitutes.o" -o "$elf"
        echo "built: $elf"
    done
}

build_dataflow brisc  0 "$HERE/reader.cpp" reader.brisc
build_dataflow ncrisc 1 "$HERE/writer.cpp" writer.ncrisc
build_compute "$HERE/compute.cpp" gqa_decode
