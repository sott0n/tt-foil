#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build every op kernel under ops/<name>/prebuilt/. Each op directory has
# its own build.sh; this is just a thin loop with a per-op status line so
# a fresh checkout can populate every kernel in one shot.
#
# Usage:
#   scripts/build_ops.sh            # builds all ops in ops/
#   scripts/build_ops.sh rope mha   # builds only the named subset
#
# Required env (inherited by each ops/*/build.sh):
#   TT_METAL_ROOT (default /home/kyamaguchi/tt-metal)
#   TT_METAL_PRECOMPILED (auto-discovered from build/firmware/)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

if [[ $# -gt 0 ]]; then
    ops=("$@")
else
    ops=()
    for d in "$REPO"/ops/*/; do
        ops+=("$(basename "$d")")
    done
fi

failed=()
for op in "${ops[@]}"; do
    bs="$REPO/ops/$op/build.sh"
    if [[ ! -x "$bs" ]]; then
        echo "[skip] $op (no build.sh)"
        continue
    fi
    echo "[build] $op"
    if ! bash "$bs" >/tmp/build_${op}.log 2>&1; then
        echo "  FAILED — see /tmp/build_${op}.log" >&2
        failed+=("$op")
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo
    echo "Failures: ${failed[*]}" >&2
    exit 1
fi
echo "ops build OK"
