#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Prompt → Qwen3-VL-2B (Blackhole) → text. Wraps the device-only
inference binary `build/models/qwen3_vl_2b/qwen3_run`:

  1. Tokenise the prompt (right-pad to 32 with endoftext = 151643)
  2. Spawn qwen3_run on it
  3. Read generated token IDs from stdout
  4. Detokenise and print the prompt + completion

Usage:
  $HOME/tt-venv/bin/python models/qwen3_vl_2b/qwen3_chat.py \\
      --prompt "The capital of Japan is" --num-decode 8

Requires:
  - $TT_FOIL_QWEN3_DATA pointing at the exported-weights tree
  - $TT_FOIL_OPS_DIR pointing at ops/
  - $TT_FOIL_DEVICE (optional, default 0)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PAD_TOKEN = 151643          # <|endoftext|> for Qwen3
SEQ       = 32
DEFAULT_TOKENIZER = (
    "/home/kyamaguchi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
    "89644892e4d85e24eaac8bacfd4f463576704203/tokenizer.json"
)


def _preflight(args) -> None:
    """Check for the artifacts qwen3_run needs and point at the setup
    scripts when something is missing — saves a confused round-trip
    through the C++ binary's first failing fopen()."""
    problems = []
    if not args.binary.exists():
        problems.append(
            f"missing executable {args.binary}\n"
            "  → cmake -B build -DTT_FOIL_HW_TESTS=ON "
            "-DTT_METAL_BUILD_DIR=<tt-metal>/build_Release\n"
            "    cmake --build build -j --target qwen3_run"
        )
    data_root = os.environ.get("TT_FOIL_QWEN3_DATA")
    if not data_root or not (Path(data_root) / "model" / "embed_tokens.bin").exists():
        problems.append(
            "missing weights (data/qwen3_vl_2b/model/embed_tokens.bin)\n"
            "  → scripts/qwen3_export_weights.sh\n"
            "    (then set TT_FOIL_QWEN3_DATA to that directory)"
        )
    ops_root = os.environ.get("TT_FOIL_OPS_DIR")
    if not ops_root or not (Path(ops_root) / "embedding" / "prebuilt" /
                            "reader.brisc.elf").exists():
        problems.append(
            "missing kernel ELFs (ops/*/prebuilt/)\n"
            "  → scripts/build_ops.sh\n"
            "    (then set TT_FOIL_OPS_DIR to the ops/ directory)"
        )
    if problems:
        print("qwen3_chat: setup incomplete:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}\n", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Text prompt (≤ 32 tokens)")
    ap.add_argument("--num-decode", type=int, default=8,
                    help="Number of tokens to generate (≤ 32)")
    ap.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    ap.add_argument("--binary", type=Path,
                    default=Path("build/models/qwen3_vl_2b/qwen3_run"),
                    help="Path to the compiled qwen3_run executable")
    args = ap.parse_args()

    _preflight(args)

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(args.tokenizer))

    ids = tk.encode(args.prompt).ids
    if len(ids) > SEQ:
        print(f"WARN: prompt has {len(ids)} tokens > {SEQ}; truncating", file=sys.stderr)
        ids = ids[:SEQ]
    prompt_len = len(ids)
    if len(ids) < SEQ:
        ids = ids + [PAD_TOKEN] * (SEQ - len(ids))
    ids_arr = np.asarray(ids, dtype=np.uint32)

    print(f"prompt ({prompt_len} tokens): {args.prompt!r}", file=sys.stderr)
    print(f"input IDs[:{prompt_len}] = {ids_arr[:prompt_len].tolist()}", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        ids_arr.tofile(f.name)
        tid_path = f.name

    try:
        proc = subprocess.run(
            [str(args.binary), tid_path, str(args.num_decode)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(tid_path)

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    # qwen3_run echoes device-side progress on stderr; forward it.
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    # qwen3_run writes one decimal ID per line on stdout; UMD shutdown
    # noise occasionally leaks here too, so skip anything non-numeric.
    gen_ids = [int(line) for line in proc.stdout.split()
               if line.strip().isdigit()]
    print(f"generated IDs   = {gen_ids}", file=sys.stderr)
    completion = tk.decode(gen_ids)
    print(f"\n=== completion ===\n{args.prompt}{completion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
