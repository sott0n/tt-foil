#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Reference forward pass via HuggingFace transformers for sanity-checking
tt-foil's numpy golden (tools/qwen3_inference_golden.py).

Runs the *full* Qwen3-VL-2B text model on the same token ids the golden
uses, captures hidden states after every layer, and prints per-layer
mean/max so we can compare against the numpy golden's output:

  python3 tools/qwen3_hf_reference.py \
      --data-dir data/qwen3_vl_2b --num-layers 3 --seq 32

Use --prompt "..." to match the golden's prompt mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   required=True, type=Path)
    ap.add_argument("--num-layers", default=3, type=int)
    ap.add_argument("--seq",        default=32, type=int)
    ap.add_argument("--prompt",     default=None, type=str)
    ap.add_argument("--seed",       default=0xBABE, type=int)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).eval()

    # Resolve to the text-only LLM submodule and verify.
    llm = model.language_model
    print(f"  loaded; LLM has {len(llm.layers)} layers, hidden={llm.config.hidden_size}")

    # Reuse the golden's token ids: either from a prompt or the rng draw.
    V = llm.config.vocab_size
    if args.prompt is not None:
        ids = tok.encode(args.prompt, add_special_tokens=False)
        pad = tok.eos_token_id or 151643
        if len(ids) < args.seq:
            ids = ids + [pad] * (args.seq - len(ids))
        else:
            ids = ids[: args.seq]
        token_ids = np.array(ids, dtype=np.int64)
        print(f"prompt → {args.seq} tokens; first 8 = {token_ids[:8].tolist()}")
    else:
        rng = np.random.default_rng(args.seed)
        token_ids = rng.integers(low=0, high=V, size=args.seq, dtype=np.uint32).astype(np.int64)
        token_ids[0] = 0; token_ids[1] = 151643; token_ids[-1] = V - 1
        print(f"random token_ids[:8] = {token_ids[:8].tolist()}")

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # [1, S]

    # Capture per-layer hidden states. The model accepts output_hidden_states.
    with torch.no_grad():
        outputs = llm(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = outputs.hidden_states  # tuple of (S, H), len = num_layers + 1
    print(f"\nHF reference per-layer hidden-state magnitudes:")
    for i, h in enumerate(hs[: args.num_layers + 1]):
        h_np = h[0].float().cpu().numpy()
        tag = "embed" if i == 0 else f"layer {i-1}"
        print(f"  {tag:10s}: |x| mean={float(np.mean(np.abs(h_np))):.4f} "
              f"max={float(np.max(np.abs(h_np))):.4f}")

    # If --num-layers is the full depth, also report final-norm + lm_head.
    if args.num_layers >= len(llm.layers):
        # outputs.last_hidden_state already has the final norm applied.
        h_final = outputs.last_hidden_state[0].float().cpu().numpy()
        print(f"  final norm: |x| mean={float(np.mean(np.abs(h_final))):.4f} "
              f"max={float(np.max(np.abs(h_final))):.4f}")
        # HF Qwen3-VL doesn't expose lm_head on AutoModel; build via tied embed.
        embed = llm.embed_tokens.weight.float().cpu().numpy()
        logits = h_final @ embed.T
        top1 = np.argmax(logits, axis=-1)
        print(f"  top-1 (first 8): {top1[:8].tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
