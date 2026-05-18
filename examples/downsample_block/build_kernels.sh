#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The downsample_block test chains: conv_3x3 stride 2 (main path), conv_3x3
# stride 1, conv_1x1 (skip projection), and residual_add. Three of the four
# kernel sets are reused from sibling examples; the 1×1 conv needs a
# custom MM_MT=1, MM_KT=1, MM_NT=2 build for the Cin=Cout=32, HW_out=64
# test shape, and we keep the residual_add RA_NT=2 build local too so
# this test is hermetic.
#
# Layout produced:
#   prebuilt/conv_s2/        ← symlink to examples/conv_3x3_s2/prebuilt
#   prebuilt/conv/           ← symlink to examples/conv_3x3/prebuilt
#   prebuilt/conv1x1/        ← fresh build (Mt=1, Kt=1, Nt=2)
#   prebuilt/residual_add/   ← fresh build (RA_NT=2)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT"

# ---- Reuse the conv_3x3 / conv_3x3_s2 prebuilt kernels as-is ----------
# Their defaults already match: Mt=1, Kt=9, Nt=2 (see tests/CMakeLists.txt
# MM_MT_C3 / MM_MT_CS2). Symlink so the test driver doesn't have to know
# multiple unrelated paths.
ln -sfn "$HERE/../conv_3x3_s2/prebuilt" "$PREBUILT/conv_s2"
ln -sfn "$HERE/../conv_3x3/prebuilt"    "$PREBUILT/conv"

# ---- Build conv_1x1 with shapes matching this test --------------------
mkdir -p "$PREBUILT/conv1x1"
PREBUILT_DIR="$PREBUILT/conv1x1" MM_MT=1 MM_KT=1 MM_NT=2 \
    bash "$HERE/../conv_1x1/build_kernels.sh"

# ---- Build residual_add with RA_NT=2 ----------------------------------
mkdir -p "$PREBUILT/residual_add"
PREBUILT_DIR="$PREBUILT/residual_add" RA_NT=2 \
    bash "$HERE/../residual_add/build_kernels.sh"
