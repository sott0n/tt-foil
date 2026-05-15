#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build pre-compiled BRISC + NCRISC + TRISC kernel ELFs for the matmul_dram
# example (v5-4): same Mt×Kt × Kt×Nt → Mt×Nt schedule as matmul_mnk, but
# A/B/C streams live in DRAM (reader uses noc_async_read, writer uses
# noc_async_write).
#
# Tile counts are baked into the TRISC compute ELF at build time via
# -DMM_MT, -DMM_KT, -DMM_NT.  Override via env, e.g.:
#   MM_MT=2 MM_KT=4 MM_NT=2 ./build_kernels.sh
# test_matmul_mnk expects the same triple.
#
# Required env vars:
#   TT_METAL_ROOT        Path to a built tt-metal source tree (default: /home/kyamaguchi/tt-metal)
#   TT_METAL_PRECOMPILED Path to firmware *_weakened.elf objects
#                        (default: $TT_METAL_ROOT/tt_metal/pre-compiled/<hash>)
#
# Produces:
#   prebuilt/reader.brisc.elf    (BRISC: L1 → CB c_0)
#   prebuilt/writer.ncrisc.elf   (NCRISC: CB c_16 → L1)
#   prebuilt/compute.trisc0.elf  (UNPACK variant)
#   prebuilt/compute.trisc1.elf  (MATH variant)
#   prebuilt/compute.trisc2.elf  (PACK variant)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT="${TT_METAL_ROOT:-/home/kyamaguchi/tt-metal}"
GXX="$TT/build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
LIB="$TT/runtime/hw/lib/blackhole"
LDDIR="$TT/runtime/hw/toolchain/blackhole"

# Resolve the firmware directory containing the *_weakened.elf objects we
# need to link kernels against. Default order:
#   1. $TT_METAL_PRECOMPILED if set.
#   2. tt-foil's self-built firmware at <repo>/build/firmware/  (Plan L —
#      the preferred source; produced by scripts/build_firmware.sh from the
#      tt-metal source tree, no tt-metal runtime invocation required).
#   3. tt-metal's JIT firmware cache at $HOME/.cache/tt-metal-cache/<hash>/firmware/.
#      Fall-back when tt-foil's firmware build hasn't run yet.
#   4. tt_metal/pre-compiled/<hash>/. May be stale; last resort.
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    REPO_ROOT="$(cd "$HERE/../.." && pwd)"
    if [[ -d "$REPO_ROOT/build/firmware/brisc" ]]; then
        TT_METAL_PRECOMPILED="$REPO_ROOT/build/firmware"
    fi
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(ls -1dt "$HOME"/.cache/tt-metal-cache/*/firmware 2>/dev/null \
                            | head -n1)
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(find "$TT/tt_metal/pre-compiled" -maxdepth 1 -mindepth 1 -type d | head -n1)
fi
[[ -d "$TT_METAL_PRECOMPILED" ]] || { echo "TT_METAL_PRECOMPILED not found"; exit 1; }
echo "build_kernels: using firmware weakened.elfs from $TT_METAL_PRECOMPILED"

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

    # Stub chlkc_list.h on -I path: hand-written replacement for tt-metal's
    # JIT-generated chlkc_list.h + chlkc_descriptors.h. Provides the per-CB
    # data-format/tile-shape tables that the LLK headers index by CB id
    # (UNPACK reads unpack_*[cbid], PACK reads pack_*[cbid]). Filled in for
    # tile_copy's bf16 layout: CB 0 (input) and CB 16 (output) are
    # Float16_b (=5); all other slots stay 255 (invalid). Then pulls in the
    # user kernel via kernel_includes so kernel_main() resolves, and
    # provides run_kernel() — the contract trisck.cc relies on.
    local chlkc_stub="$BUILD/chlkc_list.h"
    cat > "$chlkc_stub" <<'EOF'
#pragma once
#include <cstdint>

// ---- chlkc_descriptors.h (scalars) ----
// DST_ACCUM_MODE=true puts DST regs into fp32 mode so the K-loop in the
// compute kernel can accumulate Kt partial products exactly without
// intermediate bf16 rounding. mm_init/llk_*_hw_configure read this
// constant at compile time, so flipping it here actually re-routes the
// pipeline configuration.
constexpr bool DST_ACCUM_MODE = true;
#define DST_SYNC_MODE DstSync::SyncHalf
constexpr bool APPROX = true;
// MATH_FIDELITY indexes the LLK fidelity tables in matmul_tiles. It is
// declared with the strongly-typed ckernel::MathFidelity enum (defined in
// llk_defs.h) only on MATH and PACK TRISCs — same shape as tt-metal's
// generated chlkc_descriptors.h. UNPACK doesn't reference it.
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "llk_defs.h"
constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(4);  // HiFi4
#endif

// ---- chlkc_descriptors.h (pack data formats; 5 = Float16_b) ----
// matmul uses CB 0 (A), CB 1 (B), CB 16 (out), all bf16.
constexpr unsigned char pack_src_format[32] = {
    5,   5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr unsigned char pack_dst_format[32] = {
    5,   5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// ---- chlkc_descriptors.h (pack tile dims; bf16 32×32 = 4 faces of 16×16) ----
constexpr std::uint8_t  pack_tile_num_faces[32]    = { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
constexpr std::uint8_t  pack_partial_face[32]      = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
constexpr std::uint8_t  pack_tile_face_r_dim[32]   = { 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 };
constexpr std::uint8_t  pack_narrow_tile[32]       = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
constexpr std::uint8_t  pack_tile_r_dim[32]        = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
constexpr std::uint8_t  pack_tile_c_dim[32]        = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
constexpr std::uint16_t pack_tile_size[32]         = { 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088 };
constexpr std::uint8_t  pack_num_faces_r_dim[32]   = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
constexpr std::uint8_t  pack_num_faces_c_dim[32]   = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

// ---- chlkc_descriptors.h (unpack data formats) ----
constexpr std::int32_t unpack_src_format[32] = {
    5,   5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr std::int32_t unpack_dst_format[32] = {
    5,   5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// ---- chlkc_descriptors.h (unpack tile dims) ----
constexpr std::uint8_t  unpack_tile_num_faces[32]   = { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
constexpr std::uint8_t  unpack_partial_face[32]     = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
constexpr std::uint8_t  unpack_tile_face_r_dim[32]  = { 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 };
constexpr std::uint8_t  unpack_narrow_tile[32]      = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
constexpr std::uint8_t  unpack_tile_r_dim[32]       = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
constexpr std::uint8_t  unpack_tile_c_dim[32]       = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
constexpr std::uint16_t unpack_tile_size[32]        = { 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088 };
constexpr std::uint8_t  unpack_num_faces_r_dim[32]  = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
constexpr std::uint8_t  unpack_num_faces_c_dim[32]  = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

// ---- run_kernel(): the contract trisck.cc relies on ----
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

        # TRISC compute kernels need extra flags beyond the BRISC/NCRISC set:
        #   -mcpu=tt-bh-tensix    enable the Tensix instruction extension so
        #                         __builtin_rvtt_* intrinsics lower inline
        #   -O3                   compute defaults to O3 in tt-metal (not Os
        #                         like data movement). Required for the
        #                         constexpr-propagation INSTRUCTION_WORD
        #                         relies on — Os leaves "n"((x))" constraints
        #                         unsolved and the build dies with
        #                         "impossible constraint in 'asm'".
        #   -ffast-math -ftt-nttp -ftt-constinit -ftt-consteval
        #                         standard tt-metal compute flags
        # Later -mcpu / -O3 win over the COMMON_CFLAGS values.
        "$GXX" "${COMMON_CFLAGS[@]}" \
            -mcpu=tt-bh-tensix -O3 \
            -ffast-math \
            -ftt-nttp -ftt-constinit -ftt-consteval \
            -DCOMPILE_FOR_TRISC=$i \
            -DUCK_CHLKC_${VAR} \
            -DTRISC_${VAR} \
            -DNAMESPACE=chlkc_${var} \
            -DPROCESSOR_INDEX=$proc_idx \
            -DMM_MT=${MM_MT:-2} -DMM_KT=${MM_KT:-4} -DMM_NT=${MM_NT:-2} \
            -c "$TT/tt_metal/hw/firmware/src/tt-1xx/trisck.cc" \
            -o "$obj"

        "$GXX" \
            -O3 -mcpu=tt-bh-tensix -ffast-math \
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

build_one brisc  0 "$HERE/kernels/reader.cpp" reader.brisc
build_one ncrisc 1 "$HERE/kernels/writer.cpp" writer.ncrisc
build_compute "$HERE/kernels/compute.cpp" compute
