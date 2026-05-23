#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Qwen3-VL-2B weight exporter. Pulls every layer + the model-
# level tensors out of the HuggingFace cache, writes bf16 .bin under
# data/qwen3_vl_2b/{layer*,model}/, and pre-tiles lm_head into
# data/qwen3_vl_2b/model/lm_head_tiled.bin so models/qwen3_vl_2b/qwen3_run can read
# it directly.
#
# Usage:
#   scripts/qwen3_export_weights.sh             # default data/qwen3_vl_2b
#   scripts/qwen3_export_weights.sh /custom/dir
#
# Requires: torch + transformers + safetensors in the active venv, and
# Qwen/Qwen3-VL-2B-Instruct already downloaded into the HF cache (the
# HF hub snapshot path is auto-detected by models/qwen3_vl_2b/export_qwen3_layer.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-$HOME/tt-venv/bin/python}"
DATA_DIR="${1:-$REPO/data/qwen3_vl_2b}"

mkdir -p "$DATA_DIR"

echo "[1/2] exporting per-layer + model tensors → $DATA_DIR"
"$PY" "$REPO/models/qwen3_vl_2b/export_qwen3_layer.py" \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --layer all \
    --model-tensors \
    --out-dir "$DATA_DIR"

echo "[2/2] tiling lm_head → $DATA_DIR/model/lm_head_tiled.bin"
"$PY" "$REPO/models/qwen3_vl_2b/golden/qwen3_lm_head_golden.py" \
    --data-dir "$DATA_DIR" \
    --hidden 2048 --vocab 151936

echo "weights exported: $DATA_DIR"
