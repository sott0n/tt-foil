#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Stem prebuilt set for tests/test_stem.cpp.
#
# The stem chain is: Conv₇ₓ₇ s=2 pad=3 → bias + ReLU → Maxpool₃ₓ₃ s=2 pad=1.
# At the test fixture shape (C=32, H_in=W_in=32, H_mid=W_mid=16, H_out=W_out=8):
#   • Conv₇ₓ₇ needs Mt=1, Kt=49, Nt=8 (HW_mid = 256 = 8 tiles), which is a
#     different Nt from conv_7x7's own prebuilt (Nt=2). We rebuild it
#     into stem/prebuilt/conv_7x7/ to keep both fixtures hermetic.
#   • Maxpool₃ₓ₃ at HW_mid → HW_out matches examples/maxpool_3x3's default
#     fixture exactly, so we symlink that one in.
#
# Output:
#   prebuilt/conv_7x7/      ← fresh build (Mt=1, Kt=49, Nt=8)
#   prebuilt/maxpool_3x3/   ← symlink to examples/maxpool_3x3/prebuilt

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT/conv_7x7"

PREBUILT_DIR="$PREBUILT/conv_7x7" MM_MT=1 MM_KT=49 MM_NT=8 \
    bash "$HERE/../../examples/conv_7x7/build_kernels.sh"

ln -sfn "$HERE/../../examples/maxpool_3x3/prebuilt" "$PREBUILT/maxpool_3x3"
