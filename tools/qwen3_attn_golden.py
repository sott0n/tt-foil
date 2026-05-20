#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate a deterministic input + RoPE tables + golden attention output for
the attention half of a Qwen3 Transformer layer (exported by
export_qwen3_layer.py).

Reference computation (matches what tests/test_qwen3_attn.cpp drives):

    x_norm = rmsnorm(x, ln1_gamma, eps)
    Q = x_norm @ W_q       # [S, num_q*head_dim]
    K = x_norm @ W_k       # [S, num_kv*head_dim]
    V = x_norm @ W_v       # [S, num_kv*head_dim]
    Q = rope_split_half(Q, cos, sin)   # per-head, shared cos/sin
    K = rope_split_half(K, cos, sin)
    for q_head in range(num_q):
        kv_head = q_head // gqa_groups
        Q_h = Q[:, q_head*head_dim:(q_head+1)*head_dim]
        K_h = K[:, kv_head*head_dim:(kv_head+1)*head_dim]
        V_h = V[:, kv_head*head_dim:(kv_head+1)*head_dim]
        attn_h = softmax_causal(Q_h @ K_h.T / sqrt(head_dim)) @ V_h
    attn = concat(attn_h)
    proj = attn @ W_o
    y    = x + proj

RoPE: split-half, Llama/Qwen3 style. cos/sin tables have shape
[seq, head_dim/2] and are shared across heads.

cos/sin are written as bf16 row-major tile-friendly layouts:
  cos_table.bin: [seq, head_dim/2] bf16 row-major
  sin_table.bin: [seq, head_dim/2] bf16 row-major

Usage:
  python3 tools/qwen3_attn_golden.py \
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


def build_rope_tables(seq: int, head_dim: int, theta: float):
    """cos/sin tables, shape [seq, head_dim/2]. Standard RoPE frequencies."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))  # [half]
    positions = np.arange(seq, dtype=np.float64)                          # [seq]
    angles = positions[:, None] * freqs[None, :]                          # [seq, half]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_split_half(x, cos, sin):
    """
    x:   [seq, num_heads * head_dim]
    cos: [seq, head_dim/2]
    sin: [seq, head_dim/2]
    """
    seq, total = x.shape
    head_dim = 2 * cos.shape[-1]
    num_heads = total // head_dim
    half = head_dim // 2

    x = x.reshape(seq, num_heads, head_dim)
    x_first  = x[..., :half]
    x_second = x[..., half:]
    cos_b = cos[:, None, :]  # [seq, 1, half], broadcast over heads
    sin_b = sin[:, None, :]

    out_first  = x_first * cos_b - x_second * sin_b
    out_second = x_second * cos_b + x_first * sin_b
    return np.concatenate([out_first, out_second], axis=-1).reshape(seq, total)


def causal_attention(Q, K, V, num_q, num_kv, head_dim, gqa_groups):
    seq = Q.shape[0]
    Q = Q.reshape(seq, num_q, head_dim)
    K = K.reshape(seq, num_kv, head_dim)
    V = V.reshape(seq, num_kv, head_dim)
    scale = 1.0 / np.sqrt(head_dim)

    out = np.zeros((seq, num_q, head_dim), dtype=np.float32)
    mask = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), k=1)  # causal

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
    ap.add_argument("--seed", default=0xBEEF, type=int)
    args = ap.parse_args()

    if args.seq % 32 != 0:
        raise SystemExit("--seq must be a multiple of 32")
    if args.head_dim % 64 != 0:
        raise SystemExit("--head-dim must be a multiple of 64 (Dt_half tiles)")
    if args.num_q % args.num_kv != 0:
        raise SystemExit("num_q must be a multiple of num_kv")

    gqa = args.num_q // args.num_kv

    with open(args.layer_dir / "manifest.json") as f:
        manifest = json.load(f)
    t = manifest["tensors"]

    ln1g = load_bf16(args.layer_dir / t["ln1_gamma"]["path"], t["ln1_gamma"]["shape"])
    Wq   = load_bf16(args.layer_dir / t["W_q"]["path"],       t["W_q"]["shape"])
    Wk   = load_bf16(args.layer_dir / t["W_k"]["path"],       t["W_k"]["shape"])
    Wv   = load_bf16(args.layer_dir / t["W_v"]["path"],       t["W_v"]["shape"])
    Wo   = load_bf16(args.layer_dir / t["W_o"]["path"],       t["W_o"]["shape"])
    q_norm_g = load_bf16(args.layer_dir / t["q_norm"]["path"], t["q_norm"]["shape"])
    k_norm_g = load_bf16(args.layer_dir / t["k_norm"]["path"], t["k_norm"]["shape"])

    H = ln1g.shape[0]
    if Wq.shape[1] != args.num_q * args.head_dim:
        raise SystemExit(
            f"W_q.shape[1] ({Wq.shape[1]}) != num_q * head_dim "
            f"({args.num_q * args.head_dim}); check export"
        )
    print(f"attn config: H={H}, S={args.seq}, num_q={args.num_q}, num_kv={args.num_kv}, "
          f"head_dim={args.head_dim}, gqa_groups={gqa}, theta={args.rope_theta}")

    # Deterministic input (post-embedding magnitude).
    rng = np.random.default_rng(args.seed)
    x_f32 = rng.standard_normal((args.seq, H), dtype=np.float32) * 0.5
    x_bf16 = f32_to_bf16(x_f32)
    x_f32_q = bf16_to_f32(x_bf16).reshape(args.seq, H)

    # RoPE tables.
    cos, sin = build_rope_tables(args.seq, args.head_dim, args.rope_theta)

    # Reference forward.
    x_norm = rmsnorm(x_f32_q, ln1g, args.eps)
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv
    # q_norm / k_norm: RMSNorm over head_dim, applied per (token, head).
    Q_h = Q.reshape(args.seq, args.num_q,  args.head_dim)
    K_h = K.reshape(args.seq, args.num_kv, args.head_dim)
    Q_h = rmsnorm(Q_h, q_norm_g, args.eps)
    K_h = rmsnorm(K_h, k_norm_g, args.eps)
    Q = Q_h.reshape(args.seq, args.num_q  * args.head_dim)
    K = K_h.reshape(args.seq, args.num_kv * args.head_dim)
    Q = apply_rope_split_half(Q, cos, sin)
    K = apply_rope_split_half(K, cos, sin)
    attn = causal_attention(Q, K, V, args.num_q, args.num_kv, args.head_dim, gqa)
    proj = attn @ Wo
    y = x_f32_q + proj

    # Save binaries (all bf16).
    out = {
        "attn_input.bin":  x_bf16,
        "attn_golden.bin": f32_to_bf16(y),
        "cos_table.bin":   f32_to_bf16(cos),
        "sin_table.bin":   f32_to_bf16(sin),
    }
    for name, arr in out.items():
        path = args.layer_dir / name
        arr.tofile(path)
        print(f"wrote {name}  ({arr.shape}, {path.stat().st_size} B)")

    manifest.setdefault("attn", {})
    manifest["attn"].update({
        "seq": args.seq,
        "eps": args.eps,
        "num_q": args.num_q,
        "num_kv": args.num_kv,
        "head_dim": args.head_dim,
        "gqa_groups": gqa,
        "rope_theta": args.rope_theta,
        "input_path":  "attn_input.bin",
        "golden_path": "attn_golden.bin",
        "cos_path":    "cos_table.bin",
        "sin_path":    "sin_table.bin",
    })
    with open(args.layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"|y| mean: {float(np.mean(np.abs(y))):.4f}, max: {float(np.max(np.abs(y))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
