#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Build a pre-tiled lm_head weight buffer (transpose of embed_tokens) and a
deterministic hidden-state input + golden logits for the lm_head matmul.

Qwen3-VL has tie_word_embeddings=True, so:
    lm_head_W = embed_tokens.T            shape [hidden, vocab]
    logits    = hidden @ lm_head_W        shape [seq, vocab]

The host-side transpose + tile pass takes >10 GB of intermediate memory if
done naively, so this script does it row-tile by row-tile and streams the
result straight to disk. The output `lm_head_tiled.bin` lays out tiles in
the same [Kt, Nt] row-major layout the tt-foil matmul kernel expects.

Usage:
  python3 models/qwen3_vl_2b/golden/qwen3_lm_head_golden.py \
      --data-dir data/qwen3_vl_2b \
      --hidden 2048 --vocab 151936 --seq 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TILE = 32


def bf16_to_f32(arr_u16):
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr_f32):
    if arr_f32.dtype != np.float32:
        arr_f32 = arr_f32.astype(np.float32)
    u32 = arr_f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + (0x7fff + lsb).astype(np.uint32)) >> 16).astype(np.uint16)


def row_major_block_to_tile(block: np.ndarray) -> np.ndarray:
    """32×32 BF16 row-major block → 4-face tile layout (4 × 16 × 16)."""
    out = np.empty(TILE * TILE, dtype=np.uint16)
    for face in range(4):
        roff = (face // 2) * 16
        coff = (face %  2) * 16
        sub = block[roff:roff + 16, coff:coff + 16]
        out[face * 256:(face + 1) * 256] = sub.reshape(-1)
    return out


def tile2d(rm: np.ndarray) -> bytes:
    """Row-major [Rows, Cols] BF16 (uint16) → tile-format bytes."""
    Rows, Cols = rm.shape
    Rt, Ct = Rows // TILE, Cols // TILE
    if Rows % TILE or Cols % TILE:
        raise ValueError(f"shape {rm.shape} not tile-aligned")
    out = np.empty(Rt * Ct * TILE * TILE, dtype=np.uint16)
    idx = 0
    for rt in range(Rt):
        for ct in range(Ct):
            block = rm[rt*TILE:(rt+1)*TILE, ct*TILE:(ct+1)*TILE]
            out[idx*TILE*TILE:(idx+1)*TILE*TILE] = row_major_block_to_tile(block)
            idx += 1
    return out.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--hidden",   required=True, type=int)
    ap.add_argument("--vocab",    required=True, type=int)
    ap.add_argument("--seq",      default=32, type=int)
    ap.add_argument("--seed",     default=0xDEAD, type=int)
    args = ap.parse_args()

    if args.seq % TILE or args.hidden % TILE or args.vocab % TILE:
        raise SystemExit("seq, hidden, and vocab must all be multiples of 32")

    H, V, S = args.hidden, args.vocab, args.seq
    embed_path = args.data_dir / "model" / "embed_tokens.bin"
    if not embed_path.is_file():
        raise SystemExit(f"missing {embed_path} — run export_qwen3_layer.py --model-tensors")

    print(f"loading embed_tokens [{V}, {H}] from {embed_path}")
    embed = np.fromfile(embed_path, dtype=np.uint16).reshape(V, H)

    # ---- lm_head_W = embed_tokens.T, then tile to [Kt, Nt] ----
    out_dir = args.data_dir / "model"
    tiled_path = out_dir / "lm_head_tiled.bin"
    print(f"transposing + tiling to {tiled_path}  "
          f"(Kt={H//TILE}, Nt={V//TILE}, {H*V*2 / 1e6:.0f} MB on disk)")

    # Transpose lazily — np.ascontiguousarray of (V,H).T allocates ~622 MB
    # transient, which is fine on a workstation but big on smaller hosts.
    embed_T = np.ascontiguousarray(embed.T)  # [H, V]
    with open(tiled_path, "wb") as f:
        f.write(tile2d(embed_T))
    del embed_T

    # ---- Hidden state input + golden logits ----
    rng = np.random.default_rng(args.seed)
    x_f32  = rng.standard_normal((S, H), dtype=np.float32) * 0.5
    x_bf16 = f32_to_bf16(x_f32)
    x      = bf16_to_f32(x_bf16).reshape(S, H)
    # Logits in float32 from bf16 weight bit patterns (matches device math).
    embed_f32 = bf16_to_f32(embed).reshape(V, H)
    logits = x @ embed_f32.T  # [S, V]

    np.asarray(x_bf16,            dtype=np.uint16).tofile(out_dir / "lm_head_input.bin")
    np.asarray(f32_to_bf16(logits), dtype=np.uint16).tofile(out_dir / "lm_head_golden.bin")

    # Convenience: top-1 token for each row (cheap host-side argmax).
    top1 = np.argmax(logits, axis=-1)
    print(f"top-1 tokens per row: {top1.tolist()[:8]}…")

    manifest_path = out_dir / "lm_head_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "hidden": H, "vocab": V, "seq": S,
            "Kt": H // TILE, "Nt": V // TILE, "St": S // TILE,
            "tiled_path":  "lm_head_tiled.bin",
            "input_path":  "lm_head_input.bin",
            "golden_path": "lm_head_golden.bin",
            "top1_per_row": top1.tolist(),
        }, f, indent=2)
    print(f"wrote {tiled_path.name}, lm_head_input.bin, lm_head_golden.bin, "
          f"lm_head_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
