#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build pre-compiled BRISC + NCRISC + TRISC kernel ELFs for the
# eltwise_sfpu example. Unlike the other examples in this directory the
# kernel sources are **byte-identical copies** of tt-metal's
# `programming_examples/eltwise_sfpu/kernels/` — proof that a tt-metal
# kernel can run on the tt-foil runtime with no changes.
#
# The reader/writer use tt-metal's modern `TensorAccessor` /
# `experimental::Noc` API, which resolves DRAM-interleaved pages via
# `dram_bank_to_noc_xy[noc][page_id % NUM_DRAM_BANKS]` — populated by
# tt-foil's BANK_TO_NOC_SCRATCH writer at boot (see
# src/bank_tables_init.cpp).
#
# Compile-time args required by TensorAccessorArgs<0>:
#   CTA[0] = ArgsConfig flag bits (tensor_accessor::ArgConfig::IsDram = 0x2)
#   CTA[1] = aligned page size in bytes (bf16 32x32 tile = 2048)
#
# These are passed as -DKERNEL_COMPILE_TIME_ARGS=2,2048 to the reader and
# writer SFPI compiles (see api/compile_time_args.h:23 which expands
# KERNEL_COMPILE_TIME_ARGS into a make_array<uint32_t>(...) initializer).
#
# Required env vars:
#   TT_METAL_ROOT        Path to a built tt-metal source tree
#   TT_METAL_PRECOMPILED Path to firmware *_weakened.elf objects
#                        (auto-resolved like tile_copy/build_kernels.sh)

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
    TT_METAL_PRECOMPILED=$(set +o pipefail; ls -1dt "$HOME"/.cache/tt-metal-cache/*/firmware 2>/dev/null \
                            | head -n1)
fi
if [[ -z "${TT_METAL_PRECOMPILED:-}" ]]; then
    TT_METAL_PRECOMPILED=$(find "$TT/tt_metal/pre-compiled" -maxdepth 1 -mindepth 1 -type d | head -n1)
fi
[[ -d "$TT_METAL_PRECOMPILED" ]] || { echo "TT_METAL_PRECOMPILED not found"; exit 1; }
echo "build_kernels: using firmware weakened.elfs from $TT_METAL_PRECOMPILED"

BUILD="${BUILD:-/tmp/tt_foil_build_eltwise_sfpu}"
PREBUILT="$HERE/prebuilt"
mkdir -p "$BUILD" "$PREBUILT"

# DRAM-interleaved TensorAccessor compile-time args (see header comment).
DRAM_TA_CT_ARGS="2,2048"

# Per-CB metadata. The reader uses CB 0 (c_0) for input, the writer
# uses CB 16 (c_16) for output, both bf16 (Float16_b = 5). Every other
# CB slot stays at format=255 (invalid) so any accidental access
# trips. Tile shape is 32x32 with 4 faces of 16x16; byte size = 2048.
#
# This file is consumed by:
#  - BRISC/NCRISC: auto-included by api/dataflow/dataflow_api.h:8
#    via __has_include, which then sets DATA_FORMATS_DEFINED so
#    `get_tile_size(cb_id)` resolves (the kernel calls this).
#  - TRISC: explicitly #included from our chlkc_list.h stub below,
#    sharing the same numbers across all 3 RISC classes.
cat > "$BUILD/chlkc_descriptors.h" <<'EOF'
#pragma once
#include <cstdint>

// Bf16 (Float16_b = 5) for CB 0 (input) and CB 16 (output); 255 elsewhere.
constexpr std::int32_t unpack_src_format[32] = {
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr std::int32_t unpack_dst_format[32] = {
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr unsigned char pack_src_format[32] = {
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
constexpr unsigned char pack_dst_format[32] = {
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    5,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// Tile size in bytes (bf16 32x32 = 2048).
constexpr std::uint16_t unpack_tile_size[32] = {
    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
};
constexpr std::uint16_t pack_tile_size[32] = {
    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
};

// Tile shape — bf16 32×32 = 4 faces of 16×16.
constexpr std::uint8_t  unpack_tile_num_faces[32]   = { 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 };
constexpr std::uint8_t  unpack_partial_face[32]     = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  unpack_tile_face_r_dim[32]  = { 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16, 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16 };
constexpr std::uint8_t  unpack_narrow_tile[32]      = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  unpack_tile_r_dim[32]       = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32, 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  unpack_tile_c_dim[32]       = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32, 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  unpack_num_faces_r_dim[32]  = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  unpack_num_faces_c_dim[32]  = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  pack_tile_num_faces[32]     = { 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 };
constexpr std::uint8_t  pack_partial_face[32]       = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  pack_tile_face_r_dim[32]    = { 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16, 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16 };
constexpr std::uint8_t  pack_narrow_tile[32]        = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
constexpr std::uint8_t  pack_tile_r_dim[32]         = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32, 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  pack_tile_c_dim[32]         = { 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32, 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 };
constexpr std::uint8_t  pack_num_faces_r_dim[32]    = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
constexpr std::uint8_t  pack_num_faces_c_dim[32]    = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };

// TRISC MATH and PACK want MathFidelity + APPROX from llk_defs.h.
// Mirrors tt-metal's JIT-generated chlkc_descriptors.h.
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "llk_defs.h"
constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(255);
constexpr bool APPROX = true;
#endif
EOF

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
)

build_one() {
    local risc="$1"
    local proc_idx="$2"
    local src="$3"
    local out_name="$4"
    local ct_args="${5:-}"

    local kernel_includes="$BUILD/kernel_includes.hpp"
    echo "#include \"$src\"" > "$kernel_includes"

    local obj="$BUILD/${risc}k_${out_name}.o"
    local elf="$PREBUILT/${out_name}.elf"

    local extra_defs=()
    if [[ -n "$ct_args" ]]; then
        extra_defs+=(-DKERNEL_COMPILE_TIME_ARGS=$ct_args)
    fi

    "$GXX" "${COMMON_CFLAGS[@]}" \
        -DCOMPILE_FOR_${risc^^} -DPROCESSOR_INDEX=$proc_idx \
        "${extra_defs[@]}" \
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

# TRISC chlkc_list.h: thin wrapper around chlkc_descriptors.h with the
# TRISC-only bits (DST_ACCUM_MODE, run_kernel) on top.
build_compute() {
    local src="$1"
    local out_name="$2"

    local kernel_includes="$BUILD/kernel_includes.hpp"
    echo "#include \"$src\"" > "$kernel_includes"

    cat > "$BUILD/chlkc_list.h" <<'EOF'
#pragma once
#include <cstdint>
#include "chlkc_descriptors.h"

// TRISC-only scalars not in chlkc_descriptors.h.
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf

// run_kernel() — the contract trisck.cc relies on. Pulls in the user
// kernel via kernel_includes so kernel_main() resolves at link time.
#include "kernel_includes.hpp"
inline std::uint32_t run_kernel() {
    kernel_main();
    return 0;
}
EOF

    local variants=(unpack math pack)
    for i in 0 1 2; do
        local var=${variants[$i]}
        local VAR=${var^^}
        local proc_idx=$((i + 2))
        local obj="$BUILD/trisck_${out_name}_${var}.o"
        local elf="$PREBUILT/${out_name}.trisc${i}.elf"
        local risc="trisc${i}"

        "$GXX" "${COMMON_CFLAGS[@]}" \
            -mcpu=tt-bh-tensix -O3 \
            -ffast-math \
            -ftt-nttp -ftt-constinit -ftt-consteval \
            -DCOMPILE_FOR_TRISC=$i \
            -DUCK_CHLKC_${VAR} \
            -DTRISC_${VAR} \
            -DNAMESPACE=chlkc_${var} \
            -DPROCESSOR_INDEX=$proc_idx \
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

build_one brisc  0 "$HERE/kernels/reader.cpp" reader.brisc  "$DRAM_TA_CT_ARGS"
build_one ncrisc 1 "$HERE/kernels/writer.cpp" writer.ncrisc "$DRAM_TA_CT_ARGS"
build_compute "$HERE/kernels/compute.cpp" compute

bash "$HERE/../../scripts/write_manifest.sh" "$PREBUILT" \
    "$TT_METAL_PRECOMPILED" "$HERE/kernels" "${BASH_SOURCE[0]}"
