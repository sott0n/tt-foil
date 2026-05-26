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
