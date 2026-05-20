#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Export one (or all) Transformer layer(s) of a Qwen3-family model into the
bf16 row-major binaries that test_qwen3_layer / test_transformer_block
can load and tile at runtime.

The script downloads (and caches) the safetensors shards via
`huggingface_hub`, picks out the tensors for the requested layer, transposes
each weight matrix so its tt-foil-side memory layout is [K, N] (matching
our matmul kernel's expectation of `B[Kt × Nt]`), casts to bf16, and
writes flat row-major `.bin` files under `<out-dir>/layer<N>/`.

A `manifest.json` records every binary's shape and dtype so the C++ test
doesn't have to guess.

Usage:
  python3 tools/export_qwen3_layer.py \
      --model Qwen/Qwen3-VL-2B-Instruct \
      --layer 0 \
      --out-dir data/qwen3_vl_2b

  --layer all                export every Transformer layer (large!)
  --cache-dir DIR            where to cache safetensors downloads (default .cache/)
  --dry-run                  print what would be exported without writing

Notes:
  * No PyTorch model instantiation — we only need the raw safetensors.
  * For Qwen3-VL-2B-Instruct (28 layers, hidden=2048, ffn=6144) each
    layer's binaries total ~96 MB. Reserve disk accordingly.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Lazy-import the heavy deps so --help works without them installed.
def _require(mod_name: str):
    try:
        return __import__(mod_name)
    except ImportError as e:
        sys.stderr.write(
            f"error: this script needs `{mod_name}`. Install with "
            f"`pip install {mod_name}`.\n"
        )
        raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# bf16 conversion (numpy float32 → uint16 round-to-nearest-even, matches the
# device-side f32_to_bf16 used by tt-foil tests).
# ---------------------------------------------------------------------------
def f32_to_bf16(arr_f32: np.ndarray) -> np.ndarray:
    if arr_f32.dtype != np.float32:
        arr_f32 = arr_f32.astype(np.float32)
    u32 = arr_f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounded = u32 + (0x7fff + lsb).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# safetensors loading. We pull just the tensors we need — Qwen3 weights are
# sharded across several .safetensors files, but the lazy `safe_open` API
# only mmaps the file, so memory stays modest.
# ---------------------------------------------------------------------------
def _open_shards(model_id: str, cache_dir: Path):
    hub = _require("huggingface_hub")
    _require("safetensors")
    _require("torch")
    from safetensors import safe_open  # noqa: E402

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Pull the index, then each shard referenced by it.
    index_path = hub.hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json",
        cache_dir=str(cache_dir),
    )
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    # Use the "pt" framework so bf16 weights load as torch.bfloat16 — the
    # numpy framework chokes on bf16. We move to numpy via .view(torch.uint16)
    # after any transpose, preserving the raw bit pattern.
    shard_handles = {}
    for shard in shard_files:
        local = hub.hf_hub_download(repo_id=model_id, filename=shard, cache_dir=str(cache_dir))
        shard_handles[shard] = safe_open(local, framework="pt")
    return weight_map, shard_handles


def _open_single(model_id: str, cache_dir: Path):
    """Fallback for single-file models (no shard index)."""
    hub = _require("huggingface_hub")
    _require("safetensors")
    _require("torch")
    from safetensors import safe_open  # noqa: E402

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = hub.hf_hub_download(
        repo_id=model_id, filename="model.safetensors", cache_dir=str(cache_dir)
    )
    handle = safe_open(local, framework="pt")
    weight_map = {k: "model.safetensors" for k in handle.keys()}
    return weight_map, {"model.safetensors": handle}


def open_model(model_id: str, cache_dir: Path):
    try:
        return _open_shards(model_id, cache_dir)
    except Exception:
        return _open_single(model_id, cache_dir)


def fetch(weight_map, shard_handles, name):
    """Returns a torch tensor (dtype bf16 or f32, as stored)."""
    shard = weight_map[name]
    return shard_handles[shard].get_tensor(name)


# ---------------------------------------------------------------------------
# Layer export
# ---------------------------------------------------------------------------
# Weight names follow the standard HuggingFace Llama/Qwen layout.
LAYER_TENSORS = {
    "ln1_gamma": "model.layers.{L}.input_layernorm.weight",        # [hidden]
    "W_q":       "model.layers.{L}.self_attn.q_proj.weight",       # [out=Q*head_dim, in=hidden]
    "W_k":       "model.layers.{L}.self_attn.k_proj.weight",       # [out=KV*head_dim, in=hidden]
    "W_v":       "model.layers.{L}.self_attn.v_proj.weight",       # [out=KV*head_dim, in=hidden]
    "W_o":       "model.layers.{L}.self_attn.o_proj.weight",       # [out=hidden, in=Q*head_dim]
    "ln2_gamma": "model.layers.{L}.post_attention_layernorm.weight",
    "W_gate":    "model.layers.{L}.mlp.gate_proj.weight",          # [ffn, hidden]
    "W_up":      "model.layers.{L}.mlp.up_proj.weight",            # [ffn, hidden]
    "W_down":    "model.layers.{L}.mlp.down_proj.weight",          # [hidden, ffn]
}


def export_layer(weight_map, shard_handles, layer_idx: int, out_dir: Path, dry_run: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"layer": layer_idx, "tensors": {}}

    torch = _require("torch")  # noqa: F841 — used via fetch()
    for short, fmt in LAYER_TENSORS.items():
        name = fmt.format(L=layer_idx)
        t = fetch(weight_map, shard_handles, name)  # torch tensor
        # Transpose 2D weights from HF's [out, in] to our matmul's [in, out].
        if t.ndim == 2:
            t = t.transpose(0, 1).contiguous()
        # Coerce to bf16 (most Qwen3 weights are already bf16; gammas might be f32).
        if t.dtype != __import__("torch").bfloat16:
            t = t.to(__import__("torch").bfloat16)
        # Reinterpret the bf16 bit pattern as uint16 so we can .numpy() it.
        bytes_view = t.view(__import__("torch").uint16).numpy()
        path = out_dir / f"{short}.bin"
        shape = list(t.shape)
        manifest["tensors"][short] = {
            "shape": shape,
            "dtype": "bf16",
            "path":  path.name,
            "src":   name,
        }
        if not dry_run:
            with open(path, "wb") as f:
                f.write(bytes_view.tobytes())
        print(f"  {short:9s}  {str(shape):s}  {bytes_view.nbytes / 1e6:.1f} MB"
              + ("  (dry-run)" if dry_run else ""))

    if not dry_run:
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct",
                    help="HuggingFace repo id")
    ap.add_argument("--layer", default="0",
                    help="layer index (int) or 'all'")
    ap.add_argument("--out-dir", default="data/qwen3_vl_2b",
                    type=Path)
    ap.add_argument("--cache-dir", default=Path(".cache"), type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"opening model: {args.model}")
    weight_map, shard_handles = open_model(args.model, args.cache_dir)

    # Pick the layer indices to export.
    if args.layer == "all":
        # Scan weight_map for unique layer indices.
        layers = sorted({
            int(k.split("model.layers.")[1].split(".")[0])
            for k in weight_map
            if k.startswith("model.layers.")
        })
    else:
        layers = [int(args.layer)]

    for L in layers:
        layer_dir = args.out_dir / f"layer{L}"
        print(f"exporting layer {L} → {layer_dir}")
        export_layer(weight_map, shard_handles, L, layer_dir, args.dry_run)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
