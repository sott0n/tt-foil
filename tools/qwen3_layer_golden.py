#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate input + RoPE tables + golden output for a full Qwen3 Transformer
layer (attention block + MLP block + 2 residuals).

Reference forward (matches what tests/test_qwen3_layer.cpp drives):

    x_norm1 = rmsnorm(x, ln1_gamma)
    Q = x_norm1 @ W_q;  K = x_norm1 @ W_k;  V = x_norm1 @ W_v
    Q = rmsnorm(Q.reshape(.., num_q,  head_dim), q_norm)
    K = rmsnorm(K.reshape(.., num_kv, head_dim), k_norm)
    Q = rope_split_half(Q, cos, sin)
    K = rope_split_half(K, cos, sin)
    attn = causal GQA over (Q, K, V)
    proj = attn @ W_o
    x_mid = x + proj                                      (residual #1)

    y_norm = rmsnorm(x_mid, ln2_gamma)
    gate = y_norm @ W_gate;   up = y_norm @ W_up
    mlp = (silu(gate) * up) @ W_down
    y_out = x_mid + mlp                                   (residual #2)

Usage:
  python3 tools/qwen3_layer_golden.py \
      --layer-dir data/qwen3_vl_2b/layer0 \
      --num-q 16 --num-kv 8 --head-dim 128 \
      --rope-theta 5000000.0 --seq 32
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


def rmsnorm(x, gamma, eps):
    ss = (x * x).mean(axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ss + eps)) * gamma


def silu(x):
    return x / (1.0 + np.exp(-x))


def build_rope_tables(seq: int, head_dim: int, theta: float):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    positions = np.arange(seq, dtype=np.float64)
    angles = positions[:, None] * freqs[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_split_half(x, cos, sin):
    seq, total = x.shape
    head_dim = 2 * cos.shape[-1]
    num_heads = total // head_dim
    half = head_dim // 2
    x = x.reshape(seq, num_heads, head_dim)
    x_first  = x[..., :half]
    x_second = x[..., half:]
    cb = cos[:, None, :]
    sb = sin[:, None, :]
    out_first  = x_first * cb - x_second * sb
    out_second = x_second * cb + x_first * sb
    return np.concatenate([out_first, out_second], axis=-1).reshape(seq, total)


def causal_attention(Q, K, V, num_q, num_kv, head_dim, gqa_groups):
    seq = Q.shape[0]
    Q = Q.reshape(seq, num_q, head_dim)
    K = K.reshape(seq, num_kv, head_dim)
    V = V.reshape(seq, num_kv, head_dim)
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros((seq, num_q, head_dim), dtype=np.float32)
    mask = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), k=1)
    for h in range(num_q):
        kv = h // gqa_groups
        Qh, Kh, Vh = Q[:, h], K[:, kv], V[:, kv]
        scores = (Qh @ Kh.T) * scale + mask
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(axis=-1, keepdims=True)
        out[:, h] = p @ Vh
    return out.reshape(seq, num_q * head_dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True, type=Path)
    ap.add_argument("--seq", default=32, type=int)
    ap.add_argument("--num-q", required=True, type=int)
    ap.add_argument("--num-kv", required=True, type=int)
    ap.add_argument("--head-dim", required=True, type=int)
    ap.add_argument("--rope-theta", required=True, type=float)
    ap.add_argument("--eps", default=1e-6, type=float)
    ap.add_argument("--seed", default=0xCAFE, type=int)
    args = ap.parse_args()

    if args.seq % 32 != 0:
        raise SystemExit("--seq must be a multiple of 32")
    if args.head_dim % 64 != 0:
        raise SystemExit("--head-dim must be a multiple of 64")
    if args.num_q % args.num_kv != 0:
        raise SystemExit("num_q must be a multiple of num_kv")
    gqa = args.num_q // args.num_kv

    with open(args.layer_dir / "manifest.json") as f:
        manifest = json.load(f)
    t = manifest["tensors"]

    L = lambda short: load_bf16(args.layer_dir / t[short]["path"], t[short]["shape"])
    ln1g  = L("ln1_gamma");   ln2g  = L("ln2_gamma")
    Wq    = L("W_q");         Wk    = L("W_k");         Wv = L("W_v");  Wo = L("W_o")
    qng   = L("q_norm");      kng   = L("k_norm")
    Wgate = L("W_gate");      Wup   = L("W_up");        Wdown = L("W_down")

    H = ln1g.shape[0]
    print(f"layer config: H={H}, S={args.seq}, num_q={args.num_q}, num_kv={args.num_kv}, "
          f"head_dim={args.head_dim}, gqa={gqa}, theta={args.rope_theta}")

    rng = np.random.default_rng(args.seed)
    x_f32 = rng.standard_normal((args.seq, H), dtype=np.float32) * 0.5
    x_bf16 = f32_to_bf16(x_f32)
    x = bf16_to_f32(x_bf16).reshape(args.seq, H)

    cos, sin = build_rope_tables(args.seq, args.head_dim, args.rope_theta)

    # Attention block.
    x_norm1 = rmsnorm(x, ln1g, args.eps)
    Q = x_norm1 @ Wq
    K = x_norm1 @ Wk
    V = x_norm1 @ Wv
    Q_h = Q.reshape(args.seq, args.num_q,  args.head_dim)
    K_h = K.reshape(args.seq, args.num_kv, args.head_dim)
    Q_h = rmsnorm(Q_h, qng, args.eps)
    K_h = rmsnorm(K_h, kng, args.eps)
    Q = Q_h.reshape(args.seq, args.num_q  * args.head_dim)
    K = K_h.reshape(args.seq, args.num_kv * args.head_dim)
    Q = apply_rope_split_half(Q, cos, sin)
    K = apply_rope_split_half(K, cos, sin)
    attn = causal_attention(Q, K, V, args.num_q, args.num_kv, args.head_dim, gqa)
    proj = attn @ Wo
    x_mid = x + proj

    # MLP block.
    y_norm = rmsnorm(x_mid, ln2g, args.eps)
    gate = y_norm @ Wgate
    up   = y_norm @ Wup
    mlp  = (silu(gate) * up) @ Wdown
    y    = x_mid + mlp

    out = {
        "layer_input.bin":  x_bf16,
        "layer_golden.bin": f32_to_bf16(y),
        "cos_table.bin":    f32_to_bf16(cos),
        "sin_table.bin":    f32_to_bf16(sin),
    }
    for name, arr in out.items():
        path = args.layer_dir / name
        arr.tofile(path)
        print(f"wrote {name}  ({arr.shape}, {path.stat().st_size} B)")

    manifest.setdefault("full_layer", {})
    manifest["full_layer"].update({
        "seq": args.seq,
        "eps": args.eps,
        "num_q": args.num_q,
        "num_kv": args.num_kv,
        "head_dim": args.head_dim,
        "gqa_groups": gqa,
        "rope_theta": args.rope_theta,
        "input_path":  "layer_input.bin",
        "golden_path": "layer_golden.bin",
        "cos_path":    "cos_table.bin",
        "sin_path":    "sin_table.bin",
    })
    with open(args.layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"|y| mean={float(np.mean(np.abs(y))):.4f} max={float(np.max(np.abs(y))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
