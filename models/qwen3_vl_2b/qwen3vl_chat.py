#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end Qwen3-VL-2B chat with image support on Tenstorrent Blackhole.

Pipeline:
  1. Load Qwen3-VL processor + visual encoder via HuggingFace (CPU/GPU).
  2. Build the chat template (Qwen3-VL conversation format).
  3. Extract visual token embeddings [N_vis × 2048] from the image.
  4. Write token_ids.bin (uint32[kS]) and visual_embeds.bin (bf16[N_vis × H]).
  5. Invoke qwen3vl_run for device-side LLM inference.
  6. Decode and print the output.

Usage:
  python3 models/qwen3_vl_2b/qwen3vl_chat.py \\
      --image path/to/image.jpg \\
      --prompt "Describe this image." \\
      --num-decode 16

Required env:
  TT_FOIL_QWEN3_DATA  — root of exported LLM weights (scripts/qwen3_export_weights.sh)
  TT_FOIL_OPS_DIR     — root of prebuilt kernel ELFs (scripts/build_ops.sh)
  TT_FOIL_DEVICE      — PCIe chip index (optional, default 0)
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

PAD_TOKEN       = 151643   # <|endoftext|>
IMAGE_PAD_TOKEN = 151655   # <|image_pad|>
VISION_START    = 151652   # <|vision_start|>
VISION_END      = 151653   # <|vision_end|>

MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
TILE_H     = 32   # must match C++ kTileH

DEFAULT_TOKENIZER = (
    "/home/kyamaguchi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
    "89644892e4d85e24eaac8bacfd4f463576704203/tokenizer.json"
)


# ---------------------------------------------------------------------------
# bf16 helpers
# ---------------------------------------------------------------------------
def _f32_to_bf16_array(arr: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 → uint16 bfloat16."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    u32 = arr.view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounded = u32 + (0x7FFF + lsb).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_f32_array(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


# ---------------------------------------------------------------------------
# Visual encoder (CPU/GPU via HuggingFace)
# ---------------------------------------------------------------------------
def _load_vl_model():
    """Load the Qwen3-VL model and processor. Returns (model, processor)."""
    try:
        import torch
        from transformers import AutoProcessor

        # Prefer the native Qwen3VL class if transformers knows about it;
        # fall back to AutoModelForImageTextToText which auto-detects the arch.
        # Do NOT use Qwen2_5_VLForConditionalGeneration — Qwen3-VL has a
        # different visual encoder (deepstack_merger_list, linear_fc1/fc2 MLP,
        # per-head q/k_norm) and the 2.5 class silently drops/ignores those
        # weights, producing garbage visual embeddings.
        model = None
        for loader_name, loader_fn in [
            ("Qwen3VLForConditionalGeneration", lambda: __import__(
                "transformers", fromlist=["Qwen3VLForConditionalGeneration"]
            ).Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16)),
            ("AutoModelForImageTextToText", lambda: __import__(
                "transformers", fromlist=["AutoModelForImageTextToText"]
            ).AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16)),
        ]:
            try:
                print(f"loading {MODEL_ID} ({loader_name})...", file=sys.stderr)
                model = loader_fn()
                break
            except (ImportError, AttributeError) as e:
                print(f"  {loader_name} not available: {e}", file=sys.stderr)
        if model is None:
            raise RuntimeError("no suitable model class found — upgrade transformers")

        model.eval()
        if torch.cuda.is_available():
            print("  moving model to CUDA...", file=sys.stderr)
            model = model.cuda()

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("  model loaded", file=sys.stderr)
        return model, processor
    except Exception as e:
        print(f"error loading HuggingFace model: {e}", file=sys.stderr)
        print("  install: pip install transformers torch qwen-vl-utils", file=sys.stderr)
        raise


def extract_visual_embeddings(
        model, processor,
        image_path: str,
        prompt: str,
        max_seq: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Qwen3-VL visual encoder on `image_path` combined with `prompt`.

    Returns
    -------
    token_ids  : uint32 [kS]  — tokenised chat-formatted sequence, IMAGE_PAD at
                                image positions, padded to a multiple of 32.
    vis_embeds : float16 [N_vis, H]  — merged visual features (after merger MLP).
    """
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Build the chat template as Qwen3-VL expects.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # Process — returns pixel_values, image_grid_thw, input_ids, attention_mask.
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    pixel_values   = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    input_ids_raw  = inputs["input_ids"][0].cpu().numpy().astype(np.int64)

    print(f"  input_ids length: {len(input_ids_raw)}", file=sys.stderr)
    print(f"  pixel_values shape: {list(pixel_values.shape)}", file=sys.stderr)
    print(f"  image_grid_thw: {image_grid_thw.tolist()}", file=sys.stderr)

    # Run the visual encoder (and merger) only — no full LLM forward pass.
    # The visual tower lives at model.model.visual for Qwen3VLForConditionalGeneration
    # and at model.visual for Qwen2_5_VLForConditionalGeneration.
    visual_tower = (getattr(model, 'visual', None)
                    or getattr(getattr(model, 'model', None), 'visual', None))
    if visual_tower is None:
        raise AttributeError(
            "Cannot find visual encoder: checked model.visual and model.model.visual. "
            f"model type={type(model).__name__}, model.model attrs with 'vis': "
            + str([a for a in dir(getattr(model, 'model', model)) if 'vis' in a.lower()])
        )
    with torch.no_grad():
        # visual_tower returns BaseModelOutputWithDeepstackFeatures.
        # pooler_output = merger(last_hidden_state) = [N_vis, 2048] (spatially merged + projected).
        # last_hidden_state = [N_raw_patches, 1024] (before merger — NOT what we want).
        vis_out = visual_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        vis_tensor = vis_out.pooler_output  # [N_vis, 2048]
        if vis_tensor is None:
            raise RuntimeError(
                "visual_tower.pooler_output is None — unexpected architecture change. "
                f"last_hidden_state shape: {vis_out.last_hidden_state.shape}"
            )
        vis_embeds_f32 = vis_tensor.float().cpu().numpy()  # [N_vis, 2048]

    num_vis = vis_embeds_f32.shape[0]
    print(f"  visual embeddings: {vis_embeds_f32.shape}", file=sys.stderr)

    # Verify the number of IMAGE_PAD tokens in input_ids matches N_vis.
    n_pad_in_ids = int(np.sum(input_ids_raw == IMAGE_PAD_TOKEN))
    if n_pad_in_ids != num_vis:
        print(f"WARN: IMAGE_PAD count in input_ids ({n_pad_in_ids}) != N_vis ({num_vis}). "
              "The sequence may not produce correct output.", file=sys.stderr)

    # Pad / truncate token_ids to a multiple of 32.
    seq_len = len(input_ids_raw)
    if max_seq is not None and seq_len > max_seq:
        # Truncate text tokens at the end; keep all image pads.
        print(f"  WARN: truncating sequence from {seq_len} to {max_seq}", file=sys.stderr)
        input_ids_raw = input_ids_raw[:max_seq]
        seq_len = max_seq

    # Round up to multiple of TILE_H.
    padded_len = ((seq_len + TILE_H - 1) // TILE_H) * TILE_H
    if padded_len != seq_len:
        pad_amount = padded_len - seq_len
        input_ids_raw = np.concatenate(
            [input_ids_raw, np.full(pad_amount, PAD_TOKEN, dtype=np.int64)]
        )

    token_ids = input_ids_raw.astype(np.uint32)

    # Convert visual embeddings to bfloat16 (uint16 bit pattern).
    vis_bf16 = _f32_to_bf16_array(vis_embeds_f32)  # [N_vis, 2048] uint16

    return token_ids, vis_bf16


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def _preflight(args) -> None:
    problems = []
    if not args.binary.exists():
        problems.append(
            f"missing executable {args.binary}\n"
            "  → cmake -B build -DTT_FOIL_HW_TESTS=ON ...\n"
            "    cmake --build build -j --target qwen3vl_run"
        )
    data_root = os.environ.get("TT_FOIL_QWEN3_DATA")
    if not data_root or not (Path(data_root) / "model" / "embed_tokens.bin").exists():
        problems.append(
            "missing LLM weights (data/qwen3_vl_2b/model/embed_tokens.bin)\n"
            "  → scripts/qwen3_export_weights.sh"
        )
    ops_root = os.environ.get("TT_FOIL_OPS_DIR")
    if not ops_root or not (Path(ops_root) / "embedding" / "prebuilt" /
                            "reader.brisc.elf").exists():
        problems.append(
            "missing kernel ELFs (ops/*/prebuilt/)\n"
            "  → scripts/build_ops.sh"
        )
    if problems:
        print("qwen3vl_chat: setup incomplete:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",       required=True,  help="Image file path")
    ap.add_argument("--prompt",      required=True,  help="Text prompt")
    ap.add_argument("--num-decode",  type=int,  default=16, help="Decode steps")
    ap.add_argument("--max-seq",     type=int,  default=None,
                    help="Max sequence length (padded to multiple of 32). "
                         "Default: auto from processor output.")
    ap.add_argument("--tokenizer",   type=Path, default=Path(DEFAULT_TOKENIZER),
                    help="Path to tokenizer.json for decoding output IDs")
    ap.add_argument("--binary",      type=Path,
                    default=Path("build/models/qwen3_vl_2b/qwen3vl_run"),
                    help="Path to qwen3vl_run executable")
    ap.add_argument("--no-fast-dispatch", action="store_true",
                    help="Disable TT_FOIL_FAST_DISPATCH (slower but no FD overhead)")
    args = ap.parse_args()

    _preflight(args)

    # -- Step 1: extract visual embeddings ------------------------------------
    print(f"image: {args.image}", file=sys.stderr)
    print(f"prompt: {args.prompt!r}", file=sys.stderr)

    model, processor = _load_vl_model()
    token_ids, vis_bf16 = extract_visual_embeddings(
        model, processor, args.image, args.prompt, max_seq=args.max_seq
    )
    num_vis = vis_bf16.shape[0]
    kS      = len(token_ids)
    print(f"  kS={kS}, num_vis={num_vis}", file=sys.stderr)

    # -- Step 2: write binaries to temp files ----------------------------------
    with tempfile.NamedTemporaryFile(suffix="_ids.bin",   delete=False) as f_ids, \
         tempfile.NamedTemporaryFile(suffix="_vis.bin",   delete=False) as f_vis:
        ids_path = f_ids.name
        vis_path = f_vis.name
        token_ids.tofile(ids_path)
        vis_bf16.tofile(vis_path)

    try:
        # -- Step 3: run qwen3vl_run ------------------------------------------
        env = os.environ.copy()
        if not args.no_fast_dispatch:
            env["TT_FOIL_FAST_DISPATCH"] = "1"

        cmd = [
            str(args.binary),
            ids_path,
            str(args.num_decode),
            vis_path,
        ]
        print(f"running: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    finally:
        os.unlink(ids_path)
        os.unlink(vis_path)

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    if proc.stderr:
        sys.stderr.write(proc.stderr)

    gen_ids = [int(line) for line in proc.stdout.split() if line.strip().isdigit()]
    print(f"\ngenerated IDs = {gen_ids}", file=sys.stderr)

    # -- Step 4: decode token IDs → text -------------------------------------
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(args.tokenizer))
        completion = tk.decode(gen_ids)
        print(f"\n=== completion ===\n{args.prompt}{completion}")
    except Exception:
        # If tokenizer not available, just print raw IDs.
        print(f"\n=== generated token IDs ===\n{gen_ids}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
