#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Full ResNet classifier prebuilt root. Three phases on a single Tensix
# core, each with three kernel programs:
#
#   stem    : Conv₇ₓ₇, Maxpool₃ₓ₃, bias_relu_post (b7 + ReLU, Nt=8)
#   blocks  : Conv₃ₓ₃, residual_add, bias_relu_post (block biases + ReLU, Nt=2)
#   tail    : global_avg_pool, FC matmul, bias_relu_post (FC bias, Nt=1)
#
# bias_relu_post is reused across phases because Nt is a runtime arg —
# one binary handles every tile count. Between phases the test drops
# the Kernel handles and calls release_kernels() to recycle the
# KERNEL_CONFIG slot.

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
if [[ ! -d "$HERE/../bias_relu_post/prebuilt" ]]; then
    bash "$HERE/../bias_relu_post/build_kernels.sh"
fi
if [[ ! -d "$HERE/../global_avg_pool/prebuilt" ]]; then
    bash "$HERE/../global_avg_pool/build_kernels.sh"
fi

ln -sfn "$HERE/../stem/prebuilt/conv_7x7"        "$PREBUILT/conv_7x7"
ln -sfn "$HERE/../maxpool_3x3/prebuilt"          "$PREBUILT/maxpool_3x3"
ln -sfn "$HERE/../conv_3x3/prebuilt"             "$PREBUILT/conv"
ln -sfn "$HERE/../basic_block/prebuilt"          "$PREBUILT/residual_add"
ln -sfn "$HERE/../classifier_tail/prebuilt/fc"   "$PREBUILT/fc"
ln -sfn "$HERE/../bias_relu_post/prebuilt"       "$PREBUILT/bias_relu_post"
ln -sfn "$HERE/../global_avg_pool/prebuilt"      "$PREBUILT/global_avg_pool"

echo "resnet_classifier prebuilt root: $PREBUILT"
ls -la "$PREBUILT"
