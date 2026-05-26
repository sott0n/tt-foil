#!/usr/bin/env bash
# Standing benchmark for qwen3vl_run (image+text → text).
# Fixed input: N_VIS=64 image-pad tokens + 6 text tokens, padded to 96.
# Visual embeds are seeded-random (seed=42, reproducible).
#
# Usage:
#   models/qwen3_vl_2b/bench/bench_vl.sh <tag>             # uses default num_decode=4
#   models/qwen3_vl_2b/bench/bench_vl.sh <tag> <num_decode>
#
# Env (with sensible defaults):
#   TT_FOIL_DEVICE        (default 0)
#   TT_FOIL_QWEN3_DATA    (default <repo>/data/qwen3_vl_2b)
#   TT_FOIL_OPS_DIR       (default <repo>/ops)
#   BENCH_NO_RESET=1      skip tt-smi reset (faster but less reproducible)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
TAG="${1:?usage: bench_vl.sh <tag> [num_decode]}"
NDEC="${2:-4}"
DEV="${TT_FOIL_DEVICE:-0}"
DATA="${TT_FOIL_QWEN3_DATA:-$REPO/data/qwen3_vl_2b}"
OPS="${TT_FOIL_OPS_DIR:-$REPO/ops}"
PROMPT="/tmp/vl_prompt_ids.bin"
VIS_EMBEDS="/tmp/vl_visual_embeds.bin"
N_VIS=64

if [[ ! -f "$PROMPT" ]] || [[ ! -f "$VIS_EMBEDS" ]]; then
    "${PYTHON:-$HOME/tt-venv/bin/python}" - <<PY
import numpy as np

N_VIS = $N_VIS
IMAGE_PAD = 151655
kH = 2048

# N_VIS image-pad tokens + "What is the capital of Tokyo" + endoftext padding
text_ids = [3838, 374, 279, 6722, 315, 6435]
raw = [IMAGE_PAD] * N_VIS + text_ids
pad_to = ((len(raw) + 31) // 32) * 32
ids = np.array(raw + [151643] * (pad_to - len(raw)), dtype=np.uint32)
ids.tofile('$PROMPT')

# Seeded-random visual embeds [N_VIS x kH] in bf16
rng = np.random.default_rng(42)
vis_f32 = rng.normal(0, 0.02, (N_VIS, kH)).astype(np.float32)
vis_bf16 = (vis_f32.view(np.uint32) >> 16).astype(np.uint16)
vis_bf16.tofile('$VIS_EMBEDS')
PY
fi

if [[ -z "${BENCH_NO_RESET:-}" ]]; then
    "${TT_SMI:-$HOME/tt-venv/bin/tt-smi}" -r "$DEV" >/dev/null 2>&1 || true
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/runs/${STAMP}-vl-${TAG}.txt"
mkdir -p "$(dirname "$OUT")"

echo "# tag=$TAG num_decode=$NDEC device=$DEV n_vis=$N_VIS" >"$OUT"
echo "# commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo none)" >>"$OUT"
echo "# date=$STAMP" >>"$OUT"

TT_FOIL_QWEN3_DATA="$DATA" TT_FOIL_OPS_DIR="$OPS" TT_FOIL_DEVICE="$DEV" \
    "$REPO/build/models/qwen3_vl_2b/qwen3vl_run" "$PROMPT" "$NDEC" "$VIS_EMBEDS" \
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
