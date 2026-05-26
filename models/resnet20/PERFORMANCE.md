# ResNet-20 (CIFAR-10) — Performance Report & Improvement Plan

**Workload**: 1-image inference, ResNet-20 (akamaster-flavor, option-A skip),
input CHW 3×32×32 → 10-way logits.
**Platform**: Tenstorrent Blackhole, tt-foil slow-dispatch, single chip,
single Tensix core (0,0).
**Profiler**: Tracy host zones + on-chip `kernel_profiler.hpp` zones,
captured via `tools/tt_foil_profile.py`. See
[`docs/profiling.md`](../../docs/profiling.md) for how to reproduce.

The raw captures this report is built from live in
`generated/cifar10_p1/` (host-side host-helper instrumentation) and
`generated/cifar10_p2/` (device-side `read_act` / `read_w` + CB-stall
zones). Both runs PASS (argmax dev=3 cat = ref=3 cat).

---

## 1. Wall-time Overview

| Metric | Value |
|---|---:|
| **Inference wall (Phase A–D)** | **161.3 ms** |
| Phase A — stem + layer1 (3 basic_blocks) | 103.8 ms (64%) |
| Phase B — layer2 (downsample + 2 basic_blocks) | 31.4 ms (19%) |
| Phase C — layer3 (downsample + 2 basic_blocks) | 23.8 ms (15%) |
| Phase D — global_avg_pool + FC | 2.4 ms (1%) |
| **Chip compute time** (TRISC + BRISC kernel cycles) | **11.9 ms** |
| **Chip utilisation** (compute / wall) | **7.4%** |

→ **92.6% of wall is host / PCIe work.** The chip is essentially idle
most of the inference.

---

## 2. Host-side Bottlenecks

After the P1 instrumentation pass (tensor labels on `read_buffer` /
`write_buffer`, `TT_FOIL_ZONE` on every `host_*` helper and on each
`run_conv` / `run_bias_relu` / `run_add` / `run_block` lambda), the
profiler captures 99.8% of Phase A wall time — the previously
mysterious "50 ms gap" is fully attributed.

### 2.1 Top zones (`tt_foil_perf_results.csv`)

| Zone | Total | Per-call | % of wall |
|------|---:|---:|---:|
| `TF_read_buffer` | **55.5 ms** | 0.94 ms × 59 | **34%** |
| `host_tile_matrix` | **36.3 ms** | 0.42 ms × 86 | **22%** |
| `TF_dispatch_execute` | **29.2 ms** | 0.49 ms × 60 | **18%** |
| `host_im2col_3x3` | **20.1 ms** | 1.06 ms × 19 | **12%** |
| `TF_device_open` | 13.1 ms | (one-shot) | 8% |
| `TF_dispatch/wait_done` | 8.1 ms | 0.13 ms × 60 | 5% |
| `host_untile_matrix` | 7.6 ms | 0.14 ms × 56 | 5% |
| `TF_kernel_load` | 3.8 ms | 0.27 ms × 14 | 2% |
| `host_weight_3x3_reshape` | 2.1 ms | 0.11 ms × 19 | 1% |
| `TF_write_buffer` | 1.4 ms | 7.7 μs × 178 | 1% |

### 2.2 PCIe Read vs Write — 122× asymmetry

|  | calls | mean | effective BW |
|---|---:|---:|---:|
| `TF_read_buffer` | 59 | **942 μs** | **~30 MB/s** |
| `TF_write_buffer` | 178 | 7.7 μs | 5–13 GB/s |

Reading 64 KB from DRAM takes ~2.2 ms; writing the same 64 KB takes
~6 μs. The asymmetry is the slow-dispatch PCIe read path (polling-style
in UMD), not any chip-side effect.

### 2.3 Read breakdown by logical tensor

Tensor labels reveal that **the same DRAM slot is read back to host
every layer**:

| Tensor (addr) | Size | Calls | Total |
|---|---:|---:|---:|
| `bias_relu_out` (`@0x152040`) | 65 536 B | 10 | **19.0 ms** |
| `conv_out`      (`@0x132040`) | 65 536 B |  7 | 15.5 ms |
| `add_out`       (`@0x152040`) | 65 536 B |  3 |  6.0 ms |
| `bias_relu_out` (`@0x152040`) | 16 384 B |  9 |  5.3 ms |
| `conv_out`      (`@0x132040`) | 16 384 B |  6 |  3.6 ms |
| `bias_relu_out` (`@0x152040`) |  8 192 B |  9 |  2.4 ms |
| (smaller / unlabeled) | — | — | ~4.0 ms |

Every layer reads its output back to the host, then writes the (re-tiled)
same data to the next operation's input. **There is no data-flow reason
to do this** — the next op already reads from device DRAM.

---

## 3. Chip-side Bottleneck (Device Performance Report)

The on-chip zones reveal that **chip compute is bandwidth-bound, not
arithmetic-bound**.

| RISC | Zone | Mean cycles | % of `*-KERNEL` |
|------|------|---:|---:|
| TRISC0 | `TRISC-KERNEL` (whole compute kernel) | 191 566 | 100% |
| TRISC0 | **`CB-COMPUTE-WAIT-FRONT`** (waiting on BRISC) | **176 922** | **92%** |
| TRISC2 | **`CB-COMPUTE-RESERVE-BACK`** (waiting on NCRISC) | **176 713** | **99%** |
| BRISC | `BRISC-KERNEL` (whole reader kernel) | 191 453 | 100% |
| BRISC | **`read_act`** (`noc_async_read` of activation tiles) | **93 959** | **49%** |
| BRISC | **`read_w`** (`noc_async_read` of weight tiles) | **90 028** | **47%** |

Reading top → bottom:

- **TRISC math is only ~8% of its kernel time** (`191k − 176k = 15k`
  cycles). The other 92% is sitting idle on `cb_wait_front`, blocked
  on the BRISC reader.
- **BRISC spends 96% of its time inside two `noc_async_read` calls**
  (`read_act + read_w = 184k / 191k` cycles).
- TRISC2 (pack) waits 99% on `cb_reserve_back` — the NCRISC writer
  can't pull tiles fast enough either.

Conclusion: every RISC is serialised on DRAM bandwidth. The matmul
hardware is barely used.

At 1 GHz (Blackhole wall-clock): one conv_3x3 kernel invocation =
192 μs chip time, of which **only ~15 μs is actual arithmetic**.

---

## 4. Memory Report — healthy, no concerns

```
POOL                         ALLOC_COUNT  PEAK_LIVE_BYTES  LEAKED_BYTES
Device DRAM                          7      1,652,736           0
Device L1 (0,0)                      3          6,144           0
L1 CB_blob (0,0)                    60          1,088           0
L1 RTA (0,0)                        14          5,120           0
L1 kernel_text (0,0)                70         46,088           0
```

The bump allocator on (0,0) handles 60 dispatches cleanly, peak L1 use
is 46 KB out of 1.5 MB available. No leaks, no fragmentation concerns
(bump allocator design — see `docs/memory_management.md`).

---

## 5. Improvement Plan

Ordered by `Δ wall / engineering cost`. Numbers are estimates derived
from the table above.

### 🥇 Tier 1 — Eliminate host round-trips (expected: −110 to −120 ms wall)

The current "conv → host read → host tile→untile → host write → bias_relu
→ …" pipeline is ~75% of total wall. Switching to an on-chip pipeline
collapses `Phase_A_stem_layer1` from 103 ms to ~25 ms.

#### 1.1 Keep activations on-device across layers

- **Current**: `run_conv` ends with `read_buffer(conv_out)` →
  `untile_matrix` (host) → … → `tile_matrix` (host) →
  `write_buffer(bias_relu_in)`.
- **Proposed**: alias the `conv_out` DRAM slot as the next op's input
  buffer. Tiles never leave DRAM between consecutive ops.
- **Expected saving**:
  - `TF_read_buffer` 55.5 ms (most of it — the only legitimate reads
    are stem input + final logits read)
  - ~half of `host_tile_matrix` / `host_untile_matrix` (≈ 22 ms)
  - **Total ≈ 70 ms**
- **Cost**: cifar10 driver rewrite only. Kernels are already
  device-buffer based — no kernel changes required.

#### 1.2 Fuse `bias_relu_post` + `residual_add` into the conv writer kernel

- **Current**: 3 dispatches per basic_block (conv → bias_relu → add).
  Each dispatch costs 450 μs of host setup + 130 μs `wait_done`.
- **Proposed**: have the conv writer add bias, apply ReLU, and add the
  skip tensor as it packs output tiles back to DRAM.
- **Expected saving**: 60 dispatches → ~30 dispatches.
  `TF_dispatch_execute` 29 ms → ~14 ms, **≈ 15 ms**.
- **Cost**: writer.cpp kernel rewrite (medium). Compute kernel stays
  matmul-only — only the writer changes.

#### 1.3 Move `im2col_3x3` from host to device reader

- **Current**: `host_im2col_3x3` takes 20.1 ms on the CPU to expand
  `(C,H,W)` into the `(C×9, H×W)` matrix the conv kernel expects.
- **Proposed**: have BRISC reader compute the im2col index pattern
  on-the-fly during `noc_async_read` instead of host pre-expanding.
- **Expected saving**: **~18 ms** (all of `host_im2col_3x3` — the
  device-side cost is hidden inside the existing `read_act` zone
  which has slack to absorb it).
- **Cost**: BRISC reader.cpp changes (medium). tt-metal has reference
  implementations for this pattern.

**Tier 1 cumulative**: 161 → ~58 ms wall, **2.8× speedup**.

### 🥈 Tier 2 — Dispatch overhead reduction (expected: −12 to −15 ms)

#### 2.1 Adopt R5 G1 fast-dispatch in the cifar10 driver

- **Current**: 60 dispatches × `TF_dispatch_execute` 524 μs = 31 ms.
  Of which `wait_done` is 134 μs, and host setup (RTA + CB blob +
  launch_msg writes) is ~390 μs/call.
- **Proposed**: switch to the fast-dispatch path (`tt-foil` already
  has `test_fast_dispatch_smoke` proving the dispatcher kernel works).
  Host setup moves into the on-chip dispatcher.
- **Expected saving**: 60 × 390 μs ≈ **23 ms** standalone, or **~12 ms**
  combined with Tier 1.2's dispatch-count halving.
- **Cost**: cifar10 driver migrates to fast-dispatch API. Existing test
  already validates the underlying mechanism.

#### 2.2 Reuse shape-agnostic kernels across phases

- **Current**: 14 `TF_kernel_load` calls (3.8 ms). `bias_relu_post`
  reloads at each of 4 phases despite being shape-agnostic at compile
  time.
- **Proposed**: load `bias_relu_post`, `global_avg_pool`,
  `residual_add` once at `open_device` time; reuse the kernel binary
  across phases (only RTAs change per dispatch).
- **Expected saving**: **~2 ms**.
- **Cost**: kernel-cache layer in the driver (small).

### 🥉 Tier 3 — Chip-side bandwidth improvements (expected: −6 to −9 ms *after Tier 1*)

After Tier 1 lands, chip compute will be ~30% of the new ~55 ms wall.
Only at that point does chip-side optimisation move the needle.

#### 3.1 Multi-bank DRAM reads

- **Current**: BRISC reads both `read_act` and `read_w` from a single
  DRAM bank, serialised.
- **Proposed**: spread A and B across two banks; issue two
  `noc_async_read`s in parallel.
- **Expected saving**: `read_act + read_w` total 184k → ~100k cycles.
  Compute kernel time roughly halves (192 → 100 μs). **≈ 6 ms** wall
  saving (post-Tier-1).
- **Cost**: host allocator becomes bank-aware; reader kernel takes
  bank-ID args.

#### 3.2 Tile-batch reads to amortise NOC-barrier overhead

- **Current**: one `noc_async_read_barrier` per tile. For
  `conv_3x3_l1` (Mt=1, Kt=9, Nt=32) that's 288 barriers per kernel run.
- **Proposed**: batch the Kt tiles of one stream into a single
  `noc_async_read`, barrier once.
- **Expected saving**: `read_act` 94k → ~30k cycles. Combined with 3.1,
  compute kernel time → ~50 μs. **≈ 3 ms** wall saving (post-Tier-1).
- **Cost**: reader kernel inner-loop rewrite; CB depth expanded to Kt
  tiles.

---

## 6. Roadmap (cumulative wall)

| Step                                  | Δ ms | Wall after | Cumulative speedup |
|---------------------------------------|---:|---:|---:|
| Baseline                              |   – | **161.3** | 1.0× |
| 1.1 keep activations on device        | −70 |   91.3 | 1.8× |
| 1.2 writer fuses bias_relu + add      | −15 |   76.3 | 2.1× |
| 1.3 device-side im2col                | −18 |   58.3 | 2.8× |
| 2.1 fast-dispatch                     | −12 |   46.3 | 3.5× |
| 2.2 kernel reuse                      |  −2 |   44.3 | 3.6× |
| 3.1 2-bank DRAM reads                 |  −6 |   38.3 | 4.2× |
| 3.2 tile-batch reads                  |  −3 |   35.3 | **4.6×** |

Realistic engineering target: **~45 ms** (Tier 1 + Tier 2 only),
roughly **3.6× over baseline**. Tier 3 is reserved for after Tier 1
to avoid optimising the wrong side of the workload.

## 7. Recommended starting point

**Tier 1.1 (round-trip elimination) first.** Single-issue saving of ~70 ms,
all changes contained to `tests/test_resnet20.cpp`, low risk —
the kernels themselves don't move. Tier 1.3 (device im2col) is
independent of 1.1 and can run in parallel.

## 8. How to reproduce

```bash
# 1. Build with both profilers on
cmake -B build -DTT_FOIL_ENABLE_TRACY=ON \
               -DTT_FOIL_DEVICE_PROFILER=ON \
               -DTT_FOIL_PROFILE_KERNEL_BITMASK=9 \
               -DTT_FOIL_HW_TESTS=ON
cmake --build build -j$(nproc)

# 2. Reset chip + run under profiler wrapper
~/tt-venv/bin/tt-smi -r <chip_id>
TT_FOIL_DEVICE=<chip_id> \
TT_FOIL_KERNEL_DIR=$PWD/models/resnet20/prebuilt \
TT_FOIL_DATA_DIR=$PWD/data/resnet20 \
python tools/tt_foil_profile.py -o generated/cifar10_run -- \
    ./build/tests/test_resnet20

# 3. Reports
cat generated/cifar10_run/reports/tt_foil_perf_results.csv      # host
cat generated/cifar10_run/reports/tt_foil_device_perf.csv       # device
cat generated/cifar10_run/reports/tt_foil_memory_report.csv     # memory
cat generated/cifar10_run/reports/tt_foil_phases.csv            # auto-phases
```

`.tracy` file at `generated/cifar10_run/.logs/tracy_profile_log.tracy`
opens in the Tracy GUI for a visual timeline.

## 9. Open questions / follow-ups

- **Why is single-tile read so slow (30 MB/s)?** UMD slow-dispatch
  uses a polling-style PCIe path. Switching to interrupt-driven reads
  or batched UMD calls could help — but Tier 1 removes most reads
  anyway, so this is a low-priority investigation.
- **Can `host_tile_matrix` be SIMD-accelerated?** Currently a scalar
  reorder loop on the host. Worth measuring after Tier 1 — if it
  still dominates host time, AVX-512 row-to-tile permute kernels are
  a 5–10× win.
- **Does `bias_relu_post` warrant TRISC zones?** The compute kernel
  has known hang issues with `DeviceZoneScopedN` inside TRISC; reader
  / writer (BRISC / NCRISC) are safe. After Tier 1.2 (writer fuse),
  this becomes moot.
