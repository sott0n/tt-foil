#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot driver for the matmul DRAM-read-vs-compute benchmark.
# Builds everything it needs (the matmul_bench target, the ops/matmul kernel
# ELFs, and the no-Bread variant reader), then runs the sweep(s) on HW.
#
# Usage:
#   bench/run.sh [decode|prefill|both|ws]   # default: both
#     decode/prefill/both : read-vs-compute sweep (matmul_bench)
#     ws                  : weight-stationary vs stock matmul (matmul_ws_check)
#
# Env:
#   DEV                PCIe board to use (default 0). Exposed via
#                      TT_VISIBLE_DEVICES so the bench opens it as index 0.
#   RESET=1            tt-smi -r the board before running (clean chip state).
#   TT_METAL_ROOT      tt-metal source (default third_party/tt-metal submodule).
#   BUILD_DIR          cmake build dir (default <repo>/build).
#   SKIP_BUILD=1       skip the cmake/kernel build steps (just run).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

MODE="${1:-both}"
DEV="${DEV:-0}"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
export TT_METAL_ROOT="${TT_METAL_ROOT:-$REPO/third_party/tt-metal}"

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    echo ">> configuring + building bench targets"
    cmake -B "$BUILD_DIR" >/dev/null
    cmake --build "$BUILD_DIR" --target matmul_bench matmul_ws_check -j"$(nproc)"

    if [[ ! -f "$REPO/ops/matmul/prebuilt/writer.ncrisc.elf" ]]; then
        echo ">> building ops/matmul kernel ELFs"
        bash "$REPO/ops/matmul/build.sh"
    fi
    echo ">> building no-Bread variant reader"
    bash "$HERE/build_noBread.sh"
fi

if [[ "${RESET:-0}" == 1 ]]; then
    echo ">> tt-smi -r $DEV"
    tt-smi -r "$DEV" >/dev/null 2>&1 || true
fi

run_bench() {
    local m="$1"
    echo
    echo "================  matmul_bench  MM_MODE=$m  board=$DEV  ================"
    # TT_VISIBLE_DEVICES filters+remaps boards, so the bench opens index 0.
    TT_VISIBLE_DEVICES="$DEV" TT_FOIL_DEVICE=0 TT_FOIL_OPS_DIR="$REPO/ops" \
        MM_NOBREAD_DIR="$HERE/prebuilt_noBread" MM_MODE="$m" \
        "$BUILD_DIR/bench/matmul_bench" 2>/dev/null
}
run_ws() {
    echo
    echo "================  matmul_ws_check (mb_max>1 vs mb_max=1)  board=$DEV  ================"
    TT_VISIBLE_DEVICES="$DEV" TT_FOIL_DEVICE=0 TT_FOIL_OPS_DIR="$REPO/ops" \
        "$BUILD_DIR/bench/matmul_ws_check" 2>/dev/null
}

case "$MODE" in
    decode|prefill) run_bench "$MODE" ;;
    both)           run_bench decode; run_bench prefill ;;
    ws)             run_ws ;;
    *) echo "usage: bench/run.sh [decode|prefill|both|ws]"; exit 2 ;;
esac
