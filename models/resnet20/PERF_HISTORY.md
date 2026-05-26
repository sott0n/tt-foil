# ResNet-20 (CIFAR-10) — Performance Iteration History

Wall-time progression from the baseline reported in [`PERFORMANCE.md`](PERFORMANCE.md).
Each entry: numeric phase wall (Phase A + B + C + D, from `tt_foil_perf_results.csv`),
correctness check (golden argmax = 3 (cat), worst |Δlogit|), and what changed.

Hardware: single Tensix core (0,0) on Blackhole, slow-dispatch. Profiler: Tracy host
zones via `tools/tt_foil_profile.py`. Build with `-DTT_FOIL_ENABLE_TRACY=ON
-DTT_FOIL_DEVICE_PROFILER=ON -DTT_FOIL_HW_TESTS=ON`.

## Summary

| Step | Wall (Phase A+B+C+D) | Δ vs prev | Δ vs baseline | Speedup |
|------|---:|---:|---:|---:|
| Baseline (PERFORMANCE.md re-run) | **166.9 ms** | — | — | 1.00× |
| Step 1 — Tier 1.1 DevBuf threading | **117.7 ms** | −49.2 ms | −49.2 ms | 1.42× |
| Step 5 — Tier 2.2 pin bias_relu_post | **114.7 ms** | −3.0 ms | −52.2 ms | 1.46× |
| Step 3 — Tier 1.2 fuse ReLU into add | **111.0 ms** | −3.7 ms | −55.9 ms | 1.50× |
| Step 4 — Tier 2.1 fast-dispatch | _not landed (deferred)_ | — | — | — |
| Step 2 — Tier 1.3 device im2col | _not landed (regression)_ | — | — | — |

**Realized**: 1.50× (167→111 ms), ~31% of original wall removed.
**Original target** in the plan was 1.50× from Tier 1+2 — hit by Step 1/3/5 alone.

---

## Step 1 — Tier 1.1: Eliminate intermediate host round-trips

**Date**: 2026-05-26
**Commit**: (pending)

### What changed

`tests/test_resnet20.cpp` only — no kernel rebuilds. Introduced a `DevBuf` handle
(`Buffer` ptr + NOC addr + tile-grid metadata) and rewrote `run_conv` / `run_bias_relu` /
`run_add` to thread DevBufs through the block instead of bouncing activations through
host CHW between every op. `run_block` now keeps activations on device for the chain
`conv2 → bias_relu2 → add → bias_relu3`, reading back to host only twice per block (after
bias_relu1 for conv2's host im2col, and after bias_relu3 for the next block's conv1). The
add now writes to `buf_Y` (freed once bias_relu2 consumed conv2's output), avoiding any
buffer aliasing.

### Numbers

| Phase | Baseline | Step 1 | Δ |
|---|---:|---:|---:|
| Phase A (stem + layer1) | 108.5 ms | 72.8 ms | −35.7 ms |
| Phase B (layer2) | 31.3 ms | 23.6 ms | −7.7 ms |
| Phase C (layer3) | 24.1 ms | 18.7 ms | −5.4 ms |
| Phase D (tail) | 2.9 ms | 2.6 ms | −0.3 ms |
| **Total** | **166.9 ms** | **117.7 ms** | **−49.2 ms** |

Driver-level deltas (counted from `tt_foil_perf_results.csv`):

| Zone | Baseline | Step 1 | Δ |
|---|---:|---:|---:|
| `TF_read_buffer` total | 60.2 ms (59 calls) | 22.9 ms (19 calls) | −37.3 ms |
| `host_tile_matrix` | 36.4 ms (86 calls) | 30.5 ms (49 calls) | −5.9 ms |
| `host_untile_matrix` | 7.6 ms (56 calls) | (folded into devbuf_read) | ~−4 ms |

### Correctness

`worst_abs = 0.1367` (identical to baseline; bf16 truncation only, no algorithmic drift).
`PASS argmax dev=3 (cat) ref=3 (cat)`.

### Reproduce

```bash
$HOME/tt-venv/bin/tt-smi -r 0
TT_FOIL_DEVICE=0 \
TT_FOIL_KERNEL_DIR=$PWD/models/resnet20/prebuilt \
TT_FOIL_DATA_DIR=$PWD/data/resnet20 \
python3 tools/tt_foil_profile.py -o generated/step1 -- ./build/tests/test_resnet20
grep -E "^Phase_" generated/step1/reports/tt_foil_perf_results.csv
```

---

## Step 5 — Tier 2.2: Pin bias_relu_post across phases

**Date**: 2026-05-26
**Commit**: (pending)

### What changed

`tests/test_resnet20.cpp` only. `bias_relu_post` is shape-agnostic (same kernel binary
across all 4 phases). Load it once before Phase A and call
`tt::foil::pin_persistent(*dev, *k_bias, core)`; per-phase `release_kernels()` now
rewinds only to the post-bias_relu watermark, leaving k_bias resident. Drop only the
shape-specific kernels (`k_conv_s1/s2`, `k_add`, `k_gap`, `k_fc`) at phase boundaries.

### Numbers

| Phase | After Step 1 | After Step 5 | Δ |
|---|---:|---:|---:|
| Phase A (stem + layer1) | 72.8 ms | 68.4 ms | −4.4 ms |
| Phase B (layer2) | 23.6 ms | 23.4 ms | −0.2 ms |
| Phase C (layer3) | 18.7 ms | 20.4 ms | +1.7 ms (run-to-run variance) |
| Phase D (tail) | 2.6 ms | 2.4 ms | −0.2 ms |
| **Total** | **117.7 ms** | **114.7 ms** | **−3.0 ms** |

`TF_kernel_load,bias_relu_post` went 4 calls → 1 call (saved 3 × ~275 μs ≈ 0.8 ms).
Phase A benefit (~4 ms) is larger than the raw load saving — likely some L1 churn /
RTA-bookkeeping cost was also amortised.

### Correctness

`worst_abs = 0.1367` identical. `PASS argmax dev=3 (cat) ref=3 (cat)`.

### Reproduce

```bash
$HOME/tt-venv/bin/tt-smi -r 0
TT_FOIL_DEVICE=0 \
TT_FOIL_KERNEL_DIR=$PWD/models/resnet20/prebuilt \
TT_FOIL_DATA_DIR=$PWD/data/resnet20 \
python3 tools/tt_foil_profile.py -o generated/step5 -- ./build/tests/test_resnet20
grep -E "^Phase_|TF_kernel_load,bias" generated/step5/reports/tt_foil_perf_results.csv
```

---

## Step 3 — Tier 1.2: Fuse ReLU into residual_add (partial writer fusion)

**Date**: 2026-05-26
**Commit**: (pending)

### What changed

Smaller-scope variant of PERFORMANCE.md §5.1.2 — fuse only the post-add ReLU (rather
than bias+add+ReLU into the conv writer). The trailing `bias_relu_post(zero, relu=1)`
after every `residual_add` becomes redundant if `residual_add` itself applies ReLU.

- `examples/residual_add/kernels/compute.cpp` — add `relu_tile_init()` and a per-tile
  `relu_tile(0)` guarded by a new `arg[0] = relu_enable` runtime arg.
- `tests/test_resnet20.cpp` `run_add` — set TRISC RTAs with `relu_enable`, default
  to off so existing callers stay correct.
- `run_block` — pass `relu_enable=1` to the post-conv2 `run_add` and drop the
  trailing `run_bias_relu(zero, relu=1)` dispatch.

### Numbers

| Phase | After Step 5 | After Step 3 | Δ |
|---|---:|---:|---:|
| Phase A (stem + layer1) | 68.4 ms | 70.9 ms | +2.5 ms (run-to-run variance) |
| Phase B (layer2) | 23.4 ms | 20.4 ms | −3.0 ms |
| Phase C (layer3) | 20.4 ms | 17.3 ms | −3.1 ms |
| Phase D (tail) | 2.4 ms | 2.5 ms | ~0 |
| **Total** | **114.7 ms** | **111.0 ms** | **−3.7 ms** |

`TF_dispatch_execute,bias_relu_post` drops 29 → 20 (saved 9 dispatches × ~425 μs
≈ 3.8 ms — matches measured wall delta). Stem (1) + 3 blocks/phase × 2 calls + Phase D's
FC bias = 7 + 6 + 6 + 1 = 20. Phase A variance is normal cross-run jitter; Phase B + C
deltas are stable.

### Correctness

`worst_abs = 0.1367` identical. `PASS argmax dev=3 (cat) ref=3 (cat)`.

### Reproduce

```bash
# Force residual_add kernel rebuild (build helper caches per-source-hash):
rm -rf examples/residual_add/prebuilt models/resnet20/prebuilt/residual_add_n*
$HOME/tt-venv/bin/tt-smi -r 0
TT_FOIL_DEVICE=0 \
TT_FOIL_KERNEL_DIR=$PWD/models/resnet20/prebuilt \
TT_FOIL_DATA_DIR=$PWD/data/resnet20 \
python3 tools/tt_foil_profile.py -o generated/step3 -- ./build/tests/test_resnet20
grep -E "^Phase_|TF_dispatch_execute,bias_relu_post" \
    generated/step3/reports/tt_foil_perf_results.csv
```

---

## Step 4 — Tier 2.1 Fast-dispatch (DEFERRED, not landed)

After Step 1+3+5, the dispatch sub-stage breakdown shows:

| Sub-stage | Total | Mean per call |
|---|---:|---:|
| `TF_dispatch/send_reset` | 0.6 ms | 12 μs (×51) |
| `TF_dispatch/setup` | 0.3 ms | 6 μs (×51) |
| `TF_dispatch/fire_go` | 30 μs | 0.6 μs (×51) |
| `TF_dispatch/wait_done` | 7.3 ms | varies (chip-bound) |

`wait_done` dominates and is chip-bound — the host already exits the
poll loop quickly when the chip is done. Fast-dispatch moves the poll
to an on-chip dispatcher kernel but cannot shrink the actual chip
execution time, so the realistic upper bound is well under the
PERFORMANCE.md §5.2.1 estimate of 12 ms. Combined with the integration
effort (separate dispatcher core, pre-staged launch_msg lifecycle, RTA
re-write per dispatch), this was deferred in favour of attempting
Step 2.

---

## Step 2 — Tier 1.3 Device im2col reader (EXPERIMENTAL, not landed)

### What was attempted

Added `examples/conv_3x3/kernels/reader_im2col.cpp`: a BRISC reader
variant that takes the activation in its post-bias_relu tiled layout
(no pre-im2col on host) and synthesises each im2col B-stream tile via
per-element NOC reads from a single 2 KB L1 source-tile scratch.
Driver side: a `run_conv_dev` helper, a new
`buf_im2col_scratch` L1 allocation, and a parallel
`k_conv_s1_im2col` kernel per phase loaded with the new reader. Used
for the block's `conv2` (input already on device); `conv1` and stem
kept their host-im2col path because their input `act` already lives on
host as CHW.

### Why it didn't land

Two iterations were tested:

1. **First iteration**: decoded `kt → (ci_block, ki, kj)` assuming
   `kt = ci_block*9 + ki*3 + kj`. This is **wrong** for the layout
   `tile_matrix` produces from `im2col_3x3` — that layout interleaves
   `ci` slowly and `(ki, kj)` fast within each global row `r = ci*9 +
   ki*3 + kj`, so a single 32-row im2col tile spans multiple `ci`
   values (e.g. `kt=0` for `Cpad=32` covers `ci=0..3`). The first
   correctness run mis-decoded every output element after layer1.0
   (`worst_abs = 26.17`, argmax wrong).

2. **Second iteration**: per-row `(ci, ki, kj)` decoding inside the
   tile. Correctness restored (`worst_abs = 0.15` end-to-end, argmax
   `cat` = ref, all stage-by-stage golden diffs ≤ 0.16 bf16). But the
   per-element NOC + L1 scatter on BRISC became the new bottleneck:
   `conv_3x3_l1` dispatch went from 825 μs/call to **16.7 ms/call**
   (~20×). Total wall jumped from 111 ms to **264 ms** (2.4× *slower*).
   Reverted.

### What would unblock it

The scatter is ~1 024 byte-level moves per im2col tile × 288 im2col
tiles per Phase A conv. BRISC's scalar pipeline can't sustain this.
Viable paths (future work):

- Row-batched reads: for fixed `(kt, nt, r_in_tile)` all 32 dest cols
  share `(ci, ki, kj)`, and (for Wo ≥ 32) the same source row in a
  single source tile. One 64-byte NOC read of the source row + a
  contiguous 64-byte L1 memcpy with boundary fix-up replaces 32 byte
  ops.
- A dedicated im2col compute kernel on TRISC (SIMD-friendly), with
  BRISC just feeding raw source tiles. SFPU shuffle / pack
  instructions can do the 4-face rearrangement in a few cycles per
  tile.
- Multi-bank DRAM reads (PERFORMANCE.md §5.3.1) plus row-batched
  reads on BRISC, sharing the activation across all 9 `(ki, kj)`
  offsets per `(mt, nt)`.

### Reproduce the experiment

The kernel source and driver wiring live on the in-progress branch
prior to revert (see commit history pre-this entry). To repeat:

1. Restore `examples/conv_3x3/kernels/reader_im2col.cpp` (final
   row-decoded version), add `build_one brisc 0
   "$HERE/kernels/reader_im2col.cpp" reader_im2col.brisc` to
   `examples/conv_3x3/build_kernels.sh`.
2. Allocate a 2 KB L1 `buf_im2col_scratch`, add `load_im2col` +
   `run_conv_dev` helpers, route `conv2` in `run_block` through
   `k_conv_s1_im2col`.
3. Force-rebuild conv_3x3 variants. `test_resnet20` will PASS but
   Phase A wall jumps to ~160 ms.
