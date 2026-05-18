#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The basic_block test chains two 3×3 convolutions and a residual add
# on device. The conv stage reuses examples/conv_3x3/prebuilt directly
# (its kernels are already shaped for Mt=1, Kt=9, Nt=2). The skip-add
# stage needs the residual_add kernels built for the matching tile
# count (Mt * Nt == 2), so this script just re-invokes residual_add's
# builder with RA_NT=2 and an alternate PREBUILT_DIR so the two
# residual_add builds don't collide.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT_DIR="$HERE/prebuilt" RA_NT=2 \
    bash "$HERE/../../examples/residual_add/build_kernels.sh"
