#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end forward pass golden for an N-layer prefix of Qwen3-VL-2B:

    token_ids
      → embed_tokens lookup           [S, H]
      → N × Transformer layer         [S, H]
      → final RMSNorm (model.norm)    [S, H]
      → lm_head (= embed_tokens.T)    [S, V]
      → argmax                        [S]

Mirrors the math the device drives in tests/test_qwen3_inference.cpp.
Uses real exported weights for layers and model-level tensors; token IDs
are deterministic (no tokenizer dependency — the model treats any id in
[0, V) as input).

Usage:
  python3 tools/qwen3_inference_golden.py \
      --data-dir data/qwen3_vl_2b --num-layers 3 \
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
    angles = np.arange(seq, dtype=np.float64)[:, None] * freqs[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_split_half(x, cos, sin):
    seq, total = x.shape
    head_dim = 2 * cos.shape[-1]
    num_heads = total // head_dim
    half = head_dim // 2
    x = x.reshape(seq, num_heads, head_dim)
    x_first, x_second = x[..., :half], x[..., half:]
    cb, sb = cos[:, None, :], sin[:, None, :]
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


def transformer_layer(x, w, cos, sin, num_q, num_kv, head_dim, gqa, eps,
                      cache=None):
    """Prefill layer. If `cache` is a dict, populates cache['K'], cache['V']
    with the post-RoPE K and the raw V (used by the decode step)."""
    x_norm1 = rmsnorm(x, w["ln1g"], eps)
    Q = x_norm1 @ w["Wq"]
    K = x_norm1 @ w["Wk"]
    V = x_norm1 @ w["Wv"]
    S = x.shape[0]
    Q = rmsnorm(Q.reshape(S, num_q,  head_dim), w["qng"], eps).reshape(S, num_q  * head_dim)
    K = rmsnorm(K.reshape(S, num_kv, head_dim), w["kng"], eps).reshape(S, num_kv * head_dim)
    Q = apply_rope_split_half(Q, cos, sin)
    K = apply_rope_split_half(K, cos, sin)
    if cache is not None:
        cache["K"] = K.copy()
        cache["V"] = V.copy()
    attn = causal_attention(Q, K, V, num_q, num_kv, head_dim, gqa)
    proj = attn @ w["Wo"]
    x_mid = x + proj
    y_norm = rmsnorm(x_mid, w["ln2g"], eps)
    mlp = (silu(y_norm @ w["Wgate"]) * (y_norm @ w["Wup"])) @ w["Wdown"]
    return x_mid + mlp


def transformer_layer_decode(x_new, w, cos_pos, sin_pos, K_cache, V_cache,
                             num_q, num_kv, head_dim, gqa, eps):
    """Decode-step layer for a single new token.

    x_new      : [1, H]
    cos_pos/sin_pos : [1, head_dim/2]  — RoPE cos/sin at the new position
    K_cache    : [S_prev, num_kv*head_dim]  — post-RoPE K from prefill
    V_cache    : [S_prev, num_kv*head_dim]  — raw V from prefill

    Returns (x_out [1, H], K_all [S_prev+1, ...], V_all [...]).
    """
    x_norm1 = rmsnorm(x_new, w["ln1g"], eps)
    Q = x_norm1 @ w["Wq"]
    K = x_norm1 @ w["Wk"]
    V = x_norm1 @ w["Wv"]
    Q = rmsnorm(Q.reshape(1, num_q,  head_dim), w["qng"], eps).reshape(1, num_q  * head_dim)
    K = rmsnorm(K.reshape(1, num_kv, head_dim), w["kng"], eps).reshape(1, num_kv * head_dim)
    Q = apply_rope_split_half(Q, cos_pos, sin_pos)
    K = apply_rope_split_half(K, cos_pos, sin_pos)

    K_all = np.concatenate([K_cache, K], axis=0)        # [S_prev+1, ...]
    V_all = np.concatenate([V_cache, V], axis=0)

    # Single-query attention over all cached + new positions (no future to mask).
    scale = 1.0 / np.sqrt(head_dim)
    Qh = Q.reshape(1, num_q,  head_dim)
    Kh = K_all.reshape(-1, num_kv, head_dim)
    Vh = V_all.reshape(-1, num_kv, head_dim)
    out = np.zeros((1, num_q, head_dim), dtype=np.float32)
    for h in range(num_q):
        kv = h // gqa
        scores = (Qh[0, h] @ Kh[:, kv].T) * scale       # [S_prev+1]
        scores -= scores.max()
        e = np.exp(scores)
        p = e / e.sum()
        out[0, h] = p @ Vh[:, kv]
    attn = out.reshape(1, num_q * head_dim)
    proj = attn @ w["Wo"]
    x_mid = x_new + proj
    y_norm = rmsnorm(x_mid, w["ln2g"], eps)
    mlp = (silu(y_norm @ w["Wgate"]) * (y_norm @ w["Wup"])) @ w["Wdown"]
    return x_mid + mlp, K_all, V_all


def load_layer(layer_dir: Path):
    with open(layer_dir / "manifest.json") as f:
        manifest = json.load(f)
    t = manifest["tensors"]
    def L(short): return load_bf16(layer_dir / t[short]["path"], t[short]["shape"])
    return {
        "ln1g": L("ln1_gamma"), "ln2g": L("ln2_gamma"),
        "Wq": L("W_q"), "Wk": L("W_k"), "Wv": L("W_v"), "Wo": L("W_o"),
        "qng": L("q_norm"), "kng": L("k_norm"),
        "Wgate": L("W_gate"), "Wup": L("W_up"), "Wdown": L("W_down"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   required=True, type=Path)
    ap.add_argument("--num-layers", required=True, type=int)
    ap.add_argument("--seq",        default=32, type=int)
    ap.add_argument("--num-q",      required=True, type=int)
    ap.add_argument("--num-kv",     required=True, type=int)
    ap.add_argument("--head-dim",   required=True, type=int)
    ap.add_argument("--rope-theta", required=True, type=float)
    ap.add_argument("--eps",        default=1e-6, type=float)
    ap.add_argument("--seed",       default=0xBABE, type=int)
    ap.add_argument("--prompt",     default=None, type=str,
                    help="If set, tokenize this prompt (Qwen3 tokenizer) "
                         "and pad/truncate to --seq instead of using --seed.")
    ap.add_argument("--tokenizer",  default=None, type=Path,
                    help="Path to tokenizer.json (defaults to HF cache).")
    args = ap.parse_args()

    if args.seq % 32 != 0:
        raise SystemExit("--seq must be a multiple of 32")
    gqa = args.num_q // args.num_kv

    # ---- Load model-level tensors ----
    model_dir = args.data_dir / "model"
    with open(model_dir / "manifest.json") as f:
        mman = json.load(f)
    embed = load_bf16(model_dir / mman["tensors"]["embed_tokens"]["path"],
                      mman["tensors"]["embed_tokens"]["shape"])  # [V, H]
    final_g = load_bf16(model_dir / mman["tensors"]["final_norm"]["path"],
                        mman["tensors"]["final_norm"]["shape"])  # [H]
    V, H = embed.shape
    print(f"model: H={H}, V={V}")

    layers = [load_layer(args.data_dir / f"layer{i}") for i in range(args.num_layers)]

    # ---- Token IDs: real prompt (if --prompt) or deterministic synthetic. ----
    if args.prompt is not None:
        from tokenizers import Tokenizer
        tk_path = args.tokenizer or Path(
            "/home/kyamaguchi/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
            "89644892e4d85e24eaac8bacfd4f463576704203/tokenizer.json"
        )
        tk = Tokenizer.from_file(str(tk_path))
        ids = tk.encode(args.prompt).ids
        # Pad with the endoftext token (151643) or truncate to --seq.
        pad = 151643
        if len(ids) < args.seq:
            ids = ids + [pad] * (args.seq - len(ids))
        else:
            ids = ids[: args.seq]
        token_ids = np.array(ids, dtype=np.uint32)
        print(f"prompt → {args.seq} tokens; first 8 = {token_ids[:8].tolist()}")
    else:
        rng = np.random.default_rng(args.seed)
        token_ids = rng.integers(low=0, high=V, size=args.seq, dtype=np.uint32)
        token_ids[0] = 0
        token_ids[1] = 151643      # <|endoftext|>
        token_ids[-1] = V - 1
        print(f"token_ids[:8] = {token_ids[:8].tolist()}")

    # ---- Embedding lookup ----
    x = embed[token_ids]                                        # [S, H]  bf16-bit float
    # Match device path: bf16 hidden state out of the gather.
    x = bf16_to_f32(f32_to_bf16(x)).reshape(args.seq, H)

    cos, sin = build_rope_tables(args.seq, args.head_dim, args.rope_theta)

    # ---- N Transformer layers (capture KV cache per layer for decode) ----
    kv_caches = [{} for _ in layers]
    for i, w in enumerate(layers):
        x = transformer_layer(x, w, cos, sin,
                              args.num_q, args.num_kv, args.head_dim, gqa, args.eps,
                              cache=kv_caches[i])
        print(f"  after layer {i}: |x| mean={float(np.mean(np.abs(x))):.4f} "
              f"max={float(np.max(np.abs(x))):.4f}")

    # ---- Final RMSNorm ----
    x = rmsnorm(x, final_g, args.eps)
    print(f"  after final norm: |x| mean={float(np.mean(np.abs(x))):.4f} "
          f"max={float(np.max(np.abs(x))):.4f}")

    # ---- lm_head matmul (tied: embed_tokens.T) ----
    logits = x @ embed.T                                         # [S, V]
    top1 = np.argmax(logits, axis=-1).astype(np.uint32)
    print(f"top-1 tokens (first 8 of {args.seq}): {top1[:8].tolist()}")

    # ---- Save ----
    chain = args.data_dir / f"chain{args.num_layers}"
    chain.mkdir(parents=True, exist_ok=True)
    token_ids.tofile(chain / "inf_token_ids.bin")
    f32_to_bf16(logits).tofile(chain / "inf_logits_golden.bin")
    top1.tofile(chain / "inf_top1.bin")
    np.asarray(f32_to_bf16(cos), dtype=np.uint16).tofile(chain / "cos_table.bin")
    np.asarray(f32_to_bf16(sin), dtype=np.uint16).tofile(chain / "sin_table.bin")
    with open(chain / "inf_manifest.json", "w") as f:
        json.dump({
            "num_layers": args.num_layers,
            "seq": args.seq, "vocab": V, "hidden": H,
            "num_q": args.num_q, "num_kv": args.num_kv,
            "head_dim": args.head_dim, "gqa_groups": gqa,
            "rope_theta": args.rope_theta, "eps": args.eps,
            "token_ids_path":     "inf_token_ids.bin",
            "logits_golden_path": "inf_logits_golden.bin",
            "top1_path":          "inf_top1.bin",
            "top1_first8":        top1[:8].tolist(),
        }, f, indent=2)
    print(f"wrote {chain}/inf_token_ids.bin, inf_logits_golden.bin, inf_top1.bin")

    # =================================================================
    # ---- 1-step decode: feed last predicted token, attend over the
    #      S-token cache + the new K/V, save next-token argmax.
    # =================================================================
    decode_in = int(top1[args.seq - 1])
    print(f"decode input token = {decode_in}")

    x_new = embed[np.array([decode_in], dtype=np.uint32)]  # [1, H]
    x_new = bf16_to_f32(f32_to_bf16(x_new)).reshape(1, H)

    # cos/sin at the decode position (= seq, 0-indexed past the prompt).
    half = args.head_dim // 2
    freqs = 1.0 / (args.rope_theta ** (np.arange(half, dtype=np.float64) / half))
    angle = float(args.seq) * freqs
    cos_pos = np.cos(angle).astype(np.float32).reshape(1, half)
    sin_pos = np.sin(angle).astype(np.float32).reshape(1, half)

    for i, w in enumerate(layers):
        x_new, kv_caches[i]["K"], kv_caches[i]["V"] = transformer_layer_decode(
            x_new, w, cos_pos, sin_pos,
            kv_caches[i]["K"], kv_caches[i]["V"],
            args.num_q, args.num_kv, args.head_dim, gqa, args.eps,
        )

    x_new = rmsnorm(x_new, final_g, args.eps)
    logits_dec = x_new @ embed.T                         # [1, V]
    top1_dec = int(np.argmax(logits_dec[0]))
    print(f"decode top1 next token = {top1_dec}")

    np.array([decode_in], dtype=np.uint32).tofile(chain / "decode_input.bin")
    np.array([top1_dec], dtype=np.uint32).tofile(chain / "decode_top1.bin")
    np.asarray(f32_to_bf16(cos_pos), dtype=np.uint16).tofile(chain / "decode_cos.bin")
    np.asarray(f32_to_bf16(sin_pos), dtype=np.uint16).tofile(chain / "decode_sin.bin")
    print(f"wrote {chain}/decode_input.bin, decode_top1.bin, decode_cos.bin, decode_sin.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
