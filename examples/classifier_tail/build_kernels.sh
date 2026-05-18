#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Classifier tail = (host) Global Average Pool → (device) Fully-Connected
# layer + (host) bias. The FC is a thin matmul; we just rebuild conv_1x1
# at the FC's tile shape into our local prebuilt/.
#
# Default FC fixture (one tile per dim):
#   M = num_classes  → MM_MT = 1   (Cout = 32)
#   K = C_in         → MM_KT = 1   (Cin  = 32)
#   N = pad-to-tile  → MM_NT = 1   (FC has no useful N axis, but matmul
#                                    wants tile-aligned N; we pad and
#                                    only consume column 0 of the output)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT/fc"

PREBUILT_DIR="$PREBUILT/fc" MM_MT=1 MM_KT=1 MM_NT=1 \
    bash "$HERE/../conv_1x1/build_kernels.sh"

echo "classifier_tail prebuilt: $PREBUILT"
ls "$PREBUILT/fc"
