#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate a deterministic input.bin and golden.bin for the MLP half of a
Qwen3 Transformer layer exported by export_qwen3_layer.py.

The reference computation matches what tests/test_qwen3_mlp.cpp drives
on device:

    y_norm  = rmsnorm(x, ln2_gamma, eps)
    gate    = y_norm @ W_gate
    up      = y_norm @ W_up
    mlp_out = (silu(gate) * up) @ W_down
    y_out   = x + mlp_out                    (residual)

Everything is computed in float32 from bf16 weight bit patterns (matching
the device path), then cast back to bf16 for the saved golden.bin so the
comparison tolerance only has to absorb the kernel's bf16-vs-fp32 drift,
not host-side rounding choices.

Usage:
  python3 models/qwen3_vl_2b/golden/qwen3_mlp_golden.py --layer-dir data/qwen3_06b/layer0 [--seq 32]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr_f32: np.ndarray) -> np.ndarray:
    if arr_f32.dtype != np.float32:
        arr_f32 = arr_f32.astype(np.float32)
    u32 = arr_f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + (0x7fff + lsb).astype(np.uint32)) >> 16).astype(np.uint16)


def load_bf16(path: Path, shape: list[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != int(np.prod(shape)):
        raise RuntimeError(f"{path}: size {raw.size} != prod(shape)={int(np.prod(shape))}")
    return bf16_to_f32(raw).reshape(shape)


def rmsnorm(x_f32, gamma_f32, eps):
    ss = (x_f32 * x_f32).mean(axis=-1, keepdims=True)
    return x_f32 * (1.0 / np.sqrt(ss + eps)) * gamma_f32


def silu(x):
    return x / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True, type=Path)
    ap.add_argument("--seq", default=32, type=int,
                    help="Number of token rows (must be multiple of 32)")
    ap.add_argument("--seed", default=0xC0FFEE, type=int)
    ap.add_argument("--eps", default=1e-6, type=float)
    args = ap.parse_args()

    manifest_path = args.layer_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    t = manifest["tensors"]

    ln2g  = load_bf16(args.layer_dir / t["ln2_gamma"]["path"], t["ln2_gamma"]["shape"])
    Wgate = load_bf16(args.layer_dir / t["W_gate"]["path"],    t["W_gate"]["shape"])
    Wup   = load_bf16(args.layer_dir / t["W_up"]["path"],      t["W_up"]["shape"])
    Wdown = load_bf16(args.layer_dir / t["W_down"]["path"],    t["W_down"]["shape"])

    H = ln2g.shape[0]
    FFN = Wgate.shape[1]
    print(f"layer config: H={H}, FFN={FFN}, S={args.seq}, eps={args.eps}")

    if args.seq % 32 != 0:
        raise SystemExit("--seq must be a multiple of 32")

    rng = np.random.default_rng(args.seed)
    # Small-magnitude inputs (post layer-norm activations are typically O(1)).
    x_f32 = rng.standard_normal((args.seq, H), dtype=np.float32) * 0.5

    # Quantize input to bf16 just like the device sees.
    x_bf16 = f32_to_bf16(x_f32)
    x_f32_q = bf16_to_f32(x_bf16).reshape(args.seq, H)

    # Reference forward (fp32 from bf16 bit patterns).
    y_norm  = rmsnorm(x_f32_q, ln2g, args.eps)
    gate    = y_norm @ Wgate
    up      = y_norm @ Wup
    mlp_out = (silu(gate) * up) @ Wdown
    y_out   = x_f32_q + mlp_out

    # Save bf16 inputs + golden.
    x_path     = args.layer_dir / "mlp_input.bin"
    gold_path  = args.layer_dir / "mlp_golden.bin"
    x_bf16.tofile(x_path)
    f32_to_bf16(y_out).tofile(gold_path)

    print(f"wrote {x_path.name}    ({args.seq}×{H} bf16, {x_path.stat().st_size} B)")
    print(f"wrote {gold_path.name} ({args.seq}×{H} bf16, {gold_path.stat().st_size} B)")
    print(f"|y_out| mean: {float(np.mean(np.abs(y_out))):.4f}")
    print(f"|y_out| max:  {float(np.max(np.abs(y_out))):.4f}")

    # Patch manifest with the new entries.
    manifest.setdefault("mlp", {})
    manifest["mlp"].update({
        "seq": args.seq,
        "eps": args.eps,
        "input_path":  x_path.name,
        "golden_path": gold_path.name,
        "shape":       [args.seq, H],
        "dtype":       "bf16",
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
