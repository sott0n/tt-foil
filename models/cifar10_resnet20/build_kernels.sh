#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build the kernel set the cifar10_resnet20 test needs.
#
# ResNet-20 (akamaster, option-A skip) layer-by-layer kernel shapes:
#
#                              Cin/Cout (padded)  HW        Mt Kt Nt
#   stem      conv 3×3 s=1     16 → 16   (32→32)  32×32  →  1  9  32
#   layer1.*  conv 3×3 s=1     16 → 16   (32→32)  32×32  →  1  9  32
#   layer2.0  conv 3×3 s=2     16 → 32   (32→32)  16×16  →  1  9  8     ← downsample
#   layer2.0  conv 3×3 s=1     32 → 32             16×16  →  1  9  8
#   layer2.{1,2} conv 3×3 s=1  32 → 32             16×16  →  1  9  8
#   layer3.0  conv 3×3 s=2     32 → 64             8×8    →  2  9  2     ← downsample
#   layer3.0  conv 3×3 s=1     64 → 64             8×8    →  2  18 2
#   layer3.{1,2} conv 3×3 s=1  64 → 64             8×8    →  2  18 2
#
#   residual_add  Nt = Mt * Nt_act (i.e. tile count of the activation):
#                  layer1 → 32, layer2 → 8, layer3 → 4.
#
#   FC matmul     64 → 10 (Cin padded to 64, Cout padded to 32, N padded
#                 to 1 tile column = 32). Mt=1 Kt=2 Nt=1.
#
# Option A skip is computed on the host (zero-padded subsample) so no
# 1×1 conv kernel is needed.
#
# bias_relu_post and global_avg_pool are shape-agnostic at compile time
# (runtime args control everything), so we just symlink their stock
# prebuilt sets in.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT"

# ---- Build the underlying examples (the helpers in those scripts will
# self-skip if their prebuilt/ is already in sync with firmware + sources,
# so calling them unconditionally is cheap when nothing has changed but
# correctly picks up a firmware rebuild that would otherwise leave stale
# kernel ELFs).
bash "$HERE/../../examples/bias_relu_post/build_kernels.sh"
bash "$HERE/../../examples/global_avg_pool/build_kernels.sh"

# ---- conv_3x3 stride-1 variants ---------------------------------------
# Builder delegates to examples/conv_3x3/build_kernels.sh which already
# honours PREBUILT_DIR + MM_MT / MM_KT / MM_NT env overrides.
build_conv3x3() {
    local out_subdir="$1" mt="$2" kt="$3" nt="$4"
    echo "=== conv_3x3 → $out_subdir (Mt=$mt Kt=$kt Nt=$nt) ==="
    mkdir -p "$PREBUILT/$out_subdir"
    PREBUILT_DIR="$PREBUILT/$out_subdir" \
        MM_MT="$mt" MM_KT="$kt" MM_NT="$nt" \
        bash "$HERE/../../examples/conv_3x3/build_kernels.sh"
}

build_conv3x3_s2() {
    local out_subdir="$1" mt="$2" kt="$3" nt="$4"
    echo "=== conv_3x3_s2 → $out_subdir (Mt=$mt Kt=$kt Nt=$nt) ==="
    mkdir -p "$PREBUILT/$out_subdir"
    PREBUILT_DIR="$PREBUILT/$out_subdir" \
        MM_MT="$mt" MM_KT="$kt" MM_NT="$nt" \
        bash "$HERE/../../examples/conv_3x3_s2/build_kernels.sh"
}

build_conv3x3 conv_3x3_l1  1 9  32
build_conv3x3 conv_3x3_l2  1 9  8
build_conv3x3 conv_3x3_l3  2 18 2

build_conv3x3_s2 conv_3x3_s2_l2  1 9 8
build_conv3x3_s2 conv_3x3_s2_l3  2 9 2

# ---- residual_add at three different Nt -------------------------------
build_residual_add() {
    local out_subdir="$1" ra_nt="$2"
    echo "=== residual_add → $out_subdir (RA_NT=$ra_nt) ==="
    mkdir -p "$PREBUILT/$out_subdir"
    PREBUILT_DIR="$PREBUILT/$out_subdir" RA_NT="$ra_nt" \
        bash "$HERE/../../examples/residual_add/build_kernels.sh"
}

build_residual_add residual_add_n32 32
build_residual_add residual_add_n8  8
build_residual_add residual_add_n4  4

# ---- FC matmul (Mt=1, Kt=2, Nt=1) -------------------------------------
# Reuses examples/conv_1x1's matmul builder (same kernel source as
# matmul_mnk / matmul_dram).
echo "=== fc → fc (Mt=1 Kt=2 Nt=1) ==="
mkdir -p "$PREBUILT/fc"
PREBUILT_DIR="$PREBUILT/fc" MM_MT=1 MM_KT=2 MM_NT=1 \
    bash "$HERE/../../examples/conv_1x1/build_kernels.sh"

# ---- symlinks for shape-agnostic kernels ------------------------------
ln -sfn "$HERE/../../examples/bias_relu_post/prebuilt"  "$PREBUILT/bias_relu_post"
ln -sfn "$HERE/../../examples/global_avg_pool/prebuilt" "$PREBUILT/global_avg_pool"

echo
echo "cifar10_resnet20 prebuilt root: $PREBUILT"
ls -la "$PREBUILT"
