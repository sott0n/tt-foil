#!/usr/bin/env bash
# Standing benchmark for qwen3_run. One run per iteration of the
# measure-improve-repeat loop; stderr (the profile table) is teed
# into bench/runs/<timestamp>-<tag>.txt.
#
# Usage:
#   bench/bench.sh <tag>             # uses default prompt + num_decode=4
#   bench/bench.sh <tag> <num_decode>
#
# Env (with sensible defaults):
#   TT_FOIL_DEVICE        (default 0)
#   TT_FOIL_QWEN3_DATA    (default <repo>/data/qwen3_vl_2b)
#   TT_FOIL_OPS_DIR       (default <repo>/ops)
#   BENCH_NO_RESET=1      skip tt-smi reset (faster but less reproducible)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TAG="${1:?usage: bench.sh <tag> [num_decode]}"
NDEC="${2:-4}"
DEV="${TT_FOIL_DEVICE:-0}"
DATA="${TT_FOIL_QWEN3_DATA:-$REPO/data/qwen3_vl_2b}"
OPS="${TT_FOIL_OPS_DIR:-$REPO/ops}"
PROMPT="/tmp/prompt_ids.bin"

if [[ ! -f "$PROMPT" ]]; then
    "${PYTHON:-$HOME/tt-venv/bin/python}" - <<'PY'
import numpy as np
ids = np.array([3838, 374, 279, 6722, 315, 6435] + [151643]*26, dtype=np.uint32)
ids.tofile('/tmp/prompt_ids.bin')
PY
fi

if [[ -z "${BENCH_NO_RESET:-}" ]]; then
    "${TT_SMI:-$HOME/tt-venv/bin/tt-smi}" -r "$DEV" >/dev/null 2>&1 || true
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$REPO/bench/runs/${STAMP}-${TAG}.txt"
mkdir -p "$(dirname "$OUT")"

echo "# tag=$TAG num_decode=$NDEC device=$DEV" >"$OUT"
echo "# commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo none)" >>"$OUT"
echo "# date=$STAMP" >>"$OUT"

TT_FOIL_QWEN3_DATA="$DATA" TT_FOIL_OPS_DIR="$OPS" TT_FOIL_DEVICE="$DEV" \
    "$REPO/build/models/qwen3_vl_2b/qwen3_run" "$PROMPT" "$NDEC" \
    > "$OUT.tokens" 2>>"$OUT" || { echo "RUN FAILED — see $OUT" >&2; exit 1; }

echo
echo "tokens generated:"
cat "$OUT.tokens"
echo
echo "profile saved to: $OUT"
echo
grep -E '^(\s+\w|=== profile|  TOTAL|=== real_wall|  wall_ms)' "$OUT" | head -40
echo
echo "=== real wall (from binary, single source of truth) ==="
grep '^  wall_ms' "$OUT" || echo "  (not found — old binary?)"
