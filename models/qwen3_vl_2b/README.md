# Qwen3-VL-2B on tt-foil (Blackhole)

Device-only inference of `Qwen/Qwen3-VL-2B-Instruct` on a single Tenstorrent
Blackhole chip using the tt-foil runtime + the prebuilt op library under
`ops/`. Prefill + multi-step greedy decode, BF16 throughout, KV cache
device-resident.

| Component | Path |
|-----------|------|
| Inference binary (C++) | [`qwen3_run.cpp`](qwen3_run.cpp) — built as `build/models/qwen3_vl_2b/qwen3_run` |
| Prompt/detokenize wrapper | [`qwen3_chat.py`](qwen3_chat.py) |
| Weight exporter (HF → bf16 .bin) | [`export_qwen3_layer.py`](export_qwen3_layer.py) |
| HF reference (cross-check) | [`qwen3_hf_reference.py`](qwen3_hf_reference.py) |
| Numpy/Torch goldens | [`golden/`](golden/) |

## Model config

vocab=151936, hidden=2048, ffn=6144, num_q=16, num_kv=8, head_dim=128,
rope_theta=5e6, rms_norm_eps=1e-6, tie_word_embeddings=True, 28 layers.
`qwen3_run` is fixed `seq=32` (right-padded with `<|endoftext|>` = 151643).
`qwen3vl_run` takes a **variable** prefill length (`kS` inferred from the
token_ids.bin size, any multiple of 32) and runs up to **seq=1024 (kSt=32)**
since the flash-attention gqa rework — see [Attention](#attention) below.

## How to run

### 1. Build tt-foil + the inference binary

```bash
git submodule update --init --recursive --depth 1 third_party/tt-metal
( cd third_party/tt-metal && ./build_metal.sh --release )      # ~30 min, first time only

TT_METAL_BUILD_DIR=$PWD/third_party/tt-metal/build_Release \
    cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_DEVICE=0
cmake --build build -j --target qwen3_run
```

### 2. Build the op-library kernel ELFs

```bash
TT_METAL_ROOT=$PWD/third_party/tt-metal bash scripts/build_ops.sh
# → ops/<op>/prebuilt/{reader,writer,compute}.{brisc,ncrisc,trisc{0,1,2}}.elf
```

### 3. Export Qwen3-VL-2B weights from the HF cache

```bash
# Needs Qwen/Qwen3-VL-2B-Instruct already in $HOME/.cache/huggingface/hub.
scripts/qwen3_export_weights.sh                       # → data/qwen3_vl_2b/{model,layer0..layer27}/
```

### 4. Run

Chat wrapper (recommended — tokenises, decodes):

```bash
TT_FOIL_DEVICE=0 \
TT_FOIL_QWEN3_DATA=$PWD/data/qwen3_vl_2b \
TT_FOIL_OPS_DIR=$PWD/ops \
TT_FOIL_FAST_DISPATCH=1 \
$HOME/tt-venv/bin/python models/qwen3_vl_2b/qwen3_chat.py \
    --prompt "The capital of Japan is" --num-decode 8
# → "The capital of Japan is Tokyo.\n\n"
```

Raw binary (token IDs in, token IDs out):

```bash
TT_FOIL_QWEN3_DATA=$PWD/data/qwen3_vl_2b \
TT_FOIL_OPS_DIR=$PWD/ops \
TT_FOIL_FAST_DISPATCH=1 \
./build/models/qwen3_vl_2b/qwen3_run prompt_ids.bin 4
```

`prompt_ids.bin` is a raw `uint32[32]` file (token IDs, pad with 151643).

### 5. Benchmark

```bash
bench/bench.sh <tag>           # uses the canonical "What is the capital of Tokyo" prompt
                               # writes profile dump to bench/runs/<timestamp>-<tag>.txt
```

History of every measured iteration (text + VL) lives in [`bench/PERF_HISTORY.md`](bench/PERF_HISTORY.md).

## Performance

Single Blackhole chip (`TT_FOIL_DEVICE=0`), BF16 throughout, fast dispatch on,
fresh `tt-smi -r`. Decode-path figures are the standing `qwen3_run` benchmark
(prompt `[3838, 374, 279, 6722, 315, 6435] + 26×PAD` = "What is the capital of
Tokyo" + endoftext, `num_decode=4`, latest `decode-dram-shard`, 2026-06-05).
Prefill-throughput figures are the variable-length `qwen3vl_run` path
(2026-06-05, post flash-attention + WS-matmul work).

| Metric | Value | How it's measured |
|--------|-------|-------------------|
| **Prefill throughput** | 192 tok/s (seq=32) → **591 tok/s** (seq=512) | prefill tokens ÷ prefill wall (`qwen3vl_run`, see scaling table) |
| **Decode throughput** | **6.1 tok/s** | 1 ÷ ITL (4 tokens ÷ 0.66 s) |
| **TTFT** (time to first token) | **0.63 s** warm · **3.80 s** cold | prefill latency (warm); + weight upload (cold, 1.24 GB sharded across 8 channels) |
| **End-to-end latency** | **4.46 s** | full wall: weight upload + prefill(seq=32) + 4 decode steps |
| **ITL** (inter-token latency) | **~165 ms / token** | decode wall ÷ tokens, 28 layers per token |
| **Cores utilization** | 22 / 140 booted (**16 %**); 8 active per matmul (one per DRAM channel) | static grid pinning vs the 14×10 Tensix worker grid |

Decode matmul weights are column-sharded across all 8 DRAM channels (one grid
core per channel), lifting the single-channel weight-read ceiling that bounds
single-token decode. Same-build A/B vs un-sharded: **ITL 210 → 165 ms (−21.5 %),
4.76 → 6.07 tok/s (+27.5 %), bit-identical**. The one-time cold weight upload is
~0.45 s slower (per-shard repack); warm TTFT is unchanged.

Tokens emitted (must stay bit-identical across optimisations): `2303, 220, 220, 16, 13`.

### Prefill throughput scaling (`qwen3vl_run`)

Prefill is matmul-bound and parallelises across the sequence, so throughput
climbs with prompt length until weight-read bandwidth saturates. `prefill:total`
(28 layers, `num_decode=0`), single-session runs on `TT_FOIL_DEVICE=0`:

| seq | kSt | prefill wall | throughput |
|-----|-----|--------------|------------|
| 32   | 1  | 167 ms  | 192 tok/s |
| 128  | 4  | 259 ms  | 494 tok/s |
| 512  | 16 | 867 ms  | **591 tok/s** |
| 1024 | 32 | 1872 ms | 547 tok/s |

seq=32 is throughput-poor (`Mt=1`, no weight-stationary reuse); seq≥128 unlocks
WS `mb_max` blocking (Kt=64 → 8, Kt=192 ffn_down → 2). seq=1024 is reachable at
all only because the flash-attention gqa rework made attention L1 O(Dt),
sequence-independent. All outputs bit-identical with the seq=32 oracle.

### Cores utilization detail

The chip exposes a **14×10 = 140** Tensix worker grid. The model statically
pins **32** of them (1 transient + 4+4+8 matmul/lm_head grids + 1 dispatcher +
14 per-op cached cores). At any instant the heaviest stage (matmul) shards over
only **4–8** cores — utilization is dispatch/latency-bound, not compute-bound,
so the remaining cores are headroom, not a bottleneck. Per-op cores are pinned
(OpCache) so repeated calls skip kernel reload.

Full per-iteration history (39.3 s → 4.28 s, -89% cumulative; bit-identical
output throughout) lives in [`bench/PERF_HISTORY.md`](bench/PERF_HISTORY.md).

## Architecture (per token)

```
[prefill]
token_ids[32] → embed (V=151936)
             → 28 × { add_rmsnorm → fused QKV matmul → rmsnorm_rope (Q,K)
                     → gqa_fused (KV cache slot 0 write)
                     → matmul_o → add_rmsnorm → fused gate+up matmul
                     → silu_mul → matmul_down }
             → final_rmsnorm → lm_head → device argmax → next token

[decode] (KV cache device-resident, slot 1 appended via ops/kv_append)
new_token → embed (1 row, tile-padded)
         → 28 × { ... same chain, gqa_decode reads slot 0 + slot 1 ... }
         → final_rmsnorm → lm_head → argmax → next token
```

Key ops (all under `ops/`): `embedding`, `rmsnorm`, `rmsnorm_rope`,
`add_rmsnorm`, `matmul` (sharded 1×4 / 1×8 grid), `gqa_fused` (prefill),
`gqa_decode`, `kv_append`, `silu_mul`, `argmax_row0`, plus the
`cq_dispatch` on-chip dispatcher used when `TT_FOIL_FAST_DISPATCH=1`.

### Attention

`gqa_fused` (prefill) and `gqa_decode` (decode) are **flash-attention
streaming** kernels: they process one key/value block at a time and keep a
running output accumulator + denominator in L1, so the L1 footprint is **O(Dt),
independent of sequence length** (~110 KB). This is what lets `qwen3vl_run`
reach seq=1024 — the earlier full-materialization kernels held the whole `St²`
causal mask (2 MB at kSt=32) plus per-head Q/Kᵀ/V in L1 and OOM'd past ~kSt=16.

The softmax has no row-max subtraction (the `1/√d` score scale is pre-folded
into Q via the q_norm gamma), so the streaming accumulation is algebraically
exact — no online-softmax rescaling. The causal mask is a single 32×32
lower-triangular tile (off-diagonal key blocks are fully in-range; the diagonal
block is the same triangle for every query row). Trade-off: the reader re-reads
Kᵀ/V per query row-tile (DRAM O(St²·Dt)), but the causal inner loop also skips
the upper-triangle score matmuls the old kernel computed-then-masked, so at
kSt=4 flash is net **faster** (`pre:gqa_fused` -13%); seq=32 is unchanged and
bit-identical.

## Environment summary

| Var | Required | Notes |
|-----|----------|-------|
| `TT_FOIL_QWEN3_DATA` | yes | Root of exported weights (`scripts/qwen3_export_weights.sh` output) |
| `TT_FOIL_OPS_DIR` | yes | Root of prebuilt kernel ELFs (`scripts/build_ops.sh` output) |
| `TT_FOIL_DEVICE` | no (default 0) | PCIe chip index |
| `TT_FOIL_FAST_DISPATCH` | no | `1` to enable on-chip dispatcher (R5). Recommended for perf. |
| `TT_FOIL_FIRMWARE_DIR` | no | Override firmware path; auto-resolves to `build/firmware/` otherwise |

## Integration tests

Per-layer / multi-layer / full-inference tests live under
[`tests/`](tests/) and are opt-in via the `TT_FOIL_MODEL_TESTS` CMake
option (they aren't part of the default `ctest` runtime regression
because they need the exported fixtures above).

```bash
# After steps 2 & 3 above (ops built, weights exported):
cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_MODEL_TESTS=ON
cmake --build build -j$(nproc)

tt-smi -r 0
ctest --test-dir build -L model        # all qwen3 tests
ctest --test-dir build -R test_qwen3_layer   # one specific test
```

| Test | Fixture root | What it covers |
|------|--------------|----------------|
| `test_qwen3_mlp` | `data/qwen3_06b/layer0/` | Qwen3-0.6B MLP smoke |
| `test_qwen3_attn` | `data/qwen3_vl_2b/layer0/` | Layer-0 attention block |
| `test_qwen3_layer` | `data/qwen3_vl_2b/layer0/` | Layer-0 attn + MLP + residuals |
| `test_qwen3_multilayer` | `data/qwen3_vl_2b/{layer0..N-1, chainN}/` | N-layer chain (default N=3) |
| `test_qwen3_embed` | `data/qwen3_vl_2b/model/` | Embedding lookup |
| `test_qwen3_lm_head` | `data/qwen3_vl_2b/model/` | lm_head matmul (Nt=4748) |
| `test_qwen3_inference` | `data/qwen3_vl_2b/chain3/` | Full prefill forward |
| `test_qwen3_decode` | `data/qwen3_vl_2b/chain3/` | Prefill + 1-step decode w/ KV cache |
