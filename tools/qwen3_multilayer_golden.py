#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate a deterministic input + RoPE tables + golden output for an
N-layer Qwen3 Transformer chain on real exported weights.

Applies the full Transformer layer (attention + MLP + 2 residuals) N
times sequentially. cos/sin tables are shared across all layers (and
written to the parent --data-dir, not per-layer).

Usage:
  python3 tools/qwen3_multilayer_golden.py \
      --data-dir data/qwen3_vl_2b \
      --num-layers 3 \
      --num-q 16 --num-kv 8 --head-dim 128 \
      --rope-theta 5000000.0 --seq 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16_to_f32(arr_u16):
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr_f32):
    if arr_f32.dtype != np.float32:
        arr_f32 = arr_f32.astype(np.float32)
    u32 = arr_f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + (0x7fff + lsb).astype(np.uint32)) >> 16).astype(np.uint16)


def load_bf16(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != int(np.prod(shape)):
        raise RuntimeError(f"{path}: size {raw.size} != prod={int(np.prod(shape))}")
    return bf16_to_f32(raw).reshape(shape)


def rmsnorm(x, gamma, eps):
    ss = (x * x).mean(axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ss + eps)) * gamma


def silu(x):
    return x / (1.0 + np.exp(-x))


def build_rope_tables(seq, head_dim, theta):
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
    x_first, x_second = x[..., :half], x[..., half:]
    cb = cos[:, None, :]; sb = sin[:, None, :]
    out_first  = x_first * cb - x_second * sb
    out_second = x_second * cb + x_first * sb
    return np.concatenate([out_first, out_second], axis=-1).reshape(seq, total)


def causal_attention(Q, K, V, num_q, num_kv, head_dim, gqa):
    seq = Q.shape[0]
    Q = Q.reshape(seq, num_q, head_dim)
    K = K.reshape(seq, num_kv, head_dim)
    V = V.reshape(seq, num_kv, head_dim)
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros((seq, num_q, head_dim), dtype=np.float32)
    mask = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), k=1)
    for h in range(num_q):
        kv = h // gqa
        scores = (Q[:, h] @ K[:, kv].T) * scale + mask
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(axis=-1, keepdims=True)
        out[:, h] = p @ V[:, kv]
    return out.reshape(seq, num_q * head_dim)


def load_layer(layer_dir: Path):
    with open(layer_dir / "manifest.json") as f:
        manifest = json.load(f)
    t = manifest["tensors"]
    def L(short):
        return load_bf16(layer_dir / t[short]["path"], t[short]["shape"])
    return {
        "ln1g": L("ln1_gamma"), "ln2g": L("ln2_gamma"),
        "Wq": L("W_q"), "Wk": L("W_k"), "Wv": L("W_v"), "Wo": L("W_o"),
        "qng": L("q_norm"), "kng": L("k_norm"),
        "Wgate": L("W_gate"), "Wup": L("W_up"), "Wdown": L("W_down"),
    }


def transformer_layer(x, w, cos, sin, num_q, num_kv, head_dim, gqa, eps):
    x_norm1 = rmsnorm(x, w["ln1g"], eps)
    Q = x_norm1 @ w["Wq"]
    K = x_norm1 @ w["Wk"]
    V = x_norm1 @ w["Wv"]
    Q_h = Q.reshape(x.shape[0], num_q,  head_dim)
    K_h = K.reshape(x.shape[0], num_kv, head_dim)
    Q_h = rmsnorm(Q_h, w["qng"], eps)
    K_h = rmsnorm(K_h, w["kng"], eps)
    Q = apply_rope_split_half(Q_h.reshape(x.shape[0], num_q  * head_dim), cos, sin)
    K = apply_rope_split_half(K_h.reshape(x.shape[0], num_kv * head_dim), cos, sin)
    attn = causal_attention(Q, K, V, num_q, num_kv, head_dim, gqa)
    proj = attn @ w["Wo"]
    x_mid = x + proj

    y_norm = rmsnorm(x_mid, w["ln2g"], eps)
    mlp = (silu(y_norm @ w["Wgate"]) * (y_norm @ w["Wup"])) @ w["Wdown"]
    return x_mid + mlp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  required=True, type=Path,
                    help="parent dir containing layer0/, layer1/, ...")
    ap.add_argument("--num-layers", required=True, type=int)
    ap.add_argument("--seq",       default=32, type=int)
    ap.add_argument("--num-q",     required=True, type=int)
    ap.add_argument("--num-kv",    required=True, type=int)
    ap.add_argument("--head-dim",  required=True, type=int)
    ap.add_argument("--rope-theta",required=True, type=float)
    ap.add_argument("--eps",       default=1e-6, type=float)
    ap.add_argument("--seed",      default=0xFEED, type=int)
    args = ap.parse_args()

    if args.seq % 32 != 0:
        raise SystemExit("--seq must be a multiple of 32")
    gqa = args.num_q // args.num_kv

    weights = [load_layer(args.data_dir / f"layer{i}") for i in range(args.num_layers)]
    H = weights[0]["ln1g"].shape[0]
    print(f"chain: N={args.num_layers}, H={H}, S={args.seq}, num_q={args.num_q}, "
          f"num_kv={args.num_kv}, head_dim={args.head_dim}, gqa={gqa}, theta={args.rope_theta}")

    rng = np.random.default_rng(args.seed)
    x_f32 = rng.standard_normal((args.seq, H), dtype=np.float32) * 0.5
    x_bf16 = f32_to_bf16(x_f32)
    x = bf16_to_f32(x_bf16).reshape(args.seq, H)

    cos, sin = build_rope_tables(args.seq, args.head_dim, args.rope_theta)

    for i, w in enumerate(weights):
        x = transformer_layer(x, w, cos, sin,
                              args.num_q, args.num_kv, args.head_dim, gqa, args.eps)
        print(f"  after layer {i}: |x| mean={float(np.mean(np.abs(x))):.4f} "
              f"max={float(np.max(np.abs(x))):.4f}")

    out_dir = args.data_dir / f"chain{args.num_layers}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(x_bf16,             dtype=np.uint16).tofile(out_dir / "chain_input.bin")
    np.asarray(f32_to_bf16(x),     dtype=np.uint16).tofile(out_dir / "chain_golden.bin")
    np.asarray(f32_to_bf16(cos),   dtype=np.uint16).tofile(out_dir / "cos_table.bin")
    np.asarray(f32_to_bf16(sin),   dtype=np.uint16).tofile(out_dir / "sin_table.bin")
    manifest = {
        "num_layers": args.num_layers,
        "seq": args.seq, "eps": args.eps,
        "num_q": args.num_q, "num_kv": args.num_kv, "head_dim": args.head_dim,
        "gqa_groups": gqa, "rope_theta": args.rope_theta,
        "input_path":  "chain_input.bin",
        "golden_path": "chain_golden.bin",
        "cos_path":    "cos_table.bin",
        "sin_path":    "sin_table.bin",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_dir}/chain_input.bin / chain_golden.bin / cos / sin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
