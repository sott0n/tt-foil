#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Full ResNet classifier prebuilt root. The feature path (stem +
# 2× basic_block) needs four kernel programs: Conv₇ₓ₇, Maxpool₃ₓ₃,
# Conv₃ₓ₃, residual_add — together they fill Blackhole's per-core
# KERNEL_CONFIG region (~69 KB). The classifier-tail kernels are
# omitted from this root: a fifth pre-loaded kernel would exceed the
# region, so the test does GAP + FC on the host. The on-device FC
# version is exercised in isolation by test_classifier_tail.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="$HERE/prebuilt"
mkdir -p "$PREBUILT"

# Ensure stem dependency exists (build it if missing).
if [[ ! -d "$HERE/../stem/prebuilt/conv_7x7" ]]; then
    bash "$HERE/../stem/build_kernels.sh"
fi

ln -sfn "$HERE/../stem/prebuilt/conv_7x7"     "$PREBUILT/conv_7x7"
ln -sfn "$HERE/../maxpool_3x3/prebuilt"       "$PREBUILT/maxpool_3x3"
ln -sfn "$HERE/../conv_3x3/prebuilt"          "$PREBUILT/conv"
ln -sfn "$HERE/../basic_block/prebuilt"       "$PREBUILT/residual_add"

echo "resnet_classifier prebuilt root: $PREBUILT"
ls -la "$PREBUILT"
