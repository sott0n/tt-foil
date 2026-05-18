#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Mini ResNet prebuilt root for tests/test_mini_resnet.cpp.
#
# The mini chain is: stem (Conv₇ₓ₇ + Maxpool₃ₓ₃) → 2× basic_block.
# Every kernel program we need already exists with the right tile shape
# under another example's prebuilt/:
#
#   stem  →  models/stem/prebuilt/conv_7x7          (Mt=1 Kt=49 Nt=8)
#         →  examples/maxpool_3x3/prebuilt             (Nt_pool=2)
#   basic_block →  examples/conv_3x3/prebuilt          (Mt=1 Kt=9 Nt=2)
#               →  models/basic_block/prebuilt        (residual_add RA_NT=2)
#
# So this script just exposes them all under one root via symlinks.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT"

# Stem must already be built (build it explicitly if needed).
if [[ ! -d "$HERE/../stem/prebuilt/conv_7x7" ]]; then
    bash "$HERE/../stem/build_kernels.sh"
fi

ln -sfn "$HERE/../stem/prebuilt/conv_7x7"  "$PREBUILT/conv_7x7"
ln -sfn "$HERE/../../examples/maxpool_3x3/prebuilt"    "$PREBUILT/maxpool_3x3"
ln -sfn "$HERE/../../examples/conv_3x3/prebuilt"       "$PREBUILT/conv"
ln -sfn "$HERE/../basic_block/prebuilt"    "$PREBUILT/residual_add"

echo "mini_resnet prebuilt root: $PREBUILT"
ls -la "$PREBUILT"
