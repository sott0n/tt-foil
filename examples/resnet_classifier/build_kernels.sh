#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Full ResNet classifier prebuilt root. Two phases, four kernel
# programs each but distinct sets:
#
#   feature path (stem + 2× basic_block):
#     Conv₇ₓ₇, Maxpool₃ₓ₃, Conv₃ₓ₃, residual_add
#   classifier tail:
#     FC matmul
#
# Both sets together would overflow Blackhole's per-core KERNEL_CONFIG
# region (~69 KB), so the test calls release_kernels() between phases
# and reloads the FC kernel into the freed slot.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT"

# Ensure dependencies exist (build them if missing).
if [[ ! -d "$HERE/../stem/prebuilt/conv_7x7" ]]; then
    bash "$HERE/../stem/build_kernels.sh"
fi
if [[ ! -d "$HERE/../classifier_tail/prebuilt/fc" ]]; then
    bash "$HERE/../classifier_tail/build_kernels.sh"
fi

ln -sfn "$HERE/../stem/prebuilt/conv_7x7"        "$PREBUILT/conv_7x7"
ln -sfn "$HERE/../maxpool_3x3/prebuilt"          "$PREBUILT/maxpool_3x3"
ln -sfn "$HERE/../conv_3x3/prebuilt"             "$PREBUILT/conv"
ln -sfn "$HERE/../basic_block/prebuilt"          "$PREBUILT/residual_add"
ln -sfn "$HERE/../classifier_tail/prebuilt/fc"   "$PREBUILT/fc"

echo "resnet_classifier prebuilt root: $PREBUILT"
ls -la "$PREBUILT"
