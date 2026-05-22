# qwen3_run performance history

Standing benchmark: prompt `[3838, 374, 279, 6722, 315, 6435] + 26×PAD` (i.e. "What is the capital of Tokyo" + endoftext padding), `num_decode=4`, device 0, fresh `tt-smi -r` before each run.

Run with `bench/bench.sh <tag>`. Raw profile dumps land in `bench/runs/` (gitignored).

Tokens generated for this fixed prompt (must stay bit-identical across optimizations): `2303, 220, 220, 16, 13`.

| date | tag | commit | wall (s) | weights (s) | prefill (s) | decode (s, 4 step) | notes |
|------|-----|--------|----------|-------------|-------------|--------------------|-------|
| 2026-05-21 | baseline | b62624a + profiling | ~39.3 | 17.40 | 3.85 | 18.04 | per-op dispatch floor 3.5ms across small ops. Hot spots: dec:matmul_ffn 6.83s (336 × 20.3ms), dec:lm_head 1.75s (4 × 438ms), dec:matmul_qkv 2.23s (336 × 6.6ms). |
| 2026-05-21 | iter1-infra | (this commit) | ~23.3 | 3.93 | 3.35 | 16.02 | No perf change vs baseline — adds dispatch-cache infra + op_lib RTA-only setters + bench infra. Weights load drop is filesystem-cache warmth (file is gitignored, no longer cold). Bit-identical tokens 2303,220,220,16,13. |
| 2026-05-21 | iter2-matmul4 | (next commit) | ~17.5 | 4.18 | 3.25 | 10.07 | **Decode matmul sharded across 1×4 grid via make_matmul_grid + Nt_stride RTA.** dec:matmul_ffn 19.65 → 8.49 ms/call (-57%), dec:matmul_qkv 6.00 → 5.12 (-15%), dec:matmul_o 8.62 → 5.53 (-36%), dec:lm_head 437.6 → 112.1 (-74%). Net decode 16.0 → 10.1s (-37%). Prefill stays single-core. Bit-identical tokens. |
| 2026-05-22 | iter4-prefill-grid | (next commit) | ~20.6 | 4.73 | 2.94 | 12.89 | **Same 1×4 matmul grid extended to prefill.** Apples-to-apples on a single run isn't possible because the small-op dispatch floor varied this run (3.5 ms/call vs iter2's 2.4 ms/call — chip cold-state luck, all categories shifted by ~1 ms). Looking at compute-dominated ops only: pre:matmul_ffn 19.51 → 10.18 ms/call (-48%), pre:lm_head 439 → 114 ms (-74%), pre:matmul_o 7.77 → 7.31 (≈same), pre:matmul_qkv 5.82 → 6.79 (within noise — Nt=32 cases get Nt'=8 per core, dispatch overhead dominates). Prefill total 3.25 → 2.94 s (-10%). Bit-identical tokens. |
| 2026-05-22 | iter5-weights-parallel | (next commit) | **10.17** | 4.46 | 2.26 | 10.43 | **Weights load+upload pipelined.** `prepare_layer` (disk read + tile2d, CPU-only) runs on a 4-worker thread pool, semaphore-throttled to 6 in-flight, main thread consumes futures in order and does DRAM upload sequentially. embed_tokens (622 MB) + lm_head_tiled (622 MB) + final_norm gamma launched as `std::async` at startup so their disk reads overlap with UMD open + host RoPE/mask gen + the layer pipeline. **Wall 24.1 → 10.17 s (-58%)** on warm pagecache (paired same-session baseline rerun). Inner `weights:load+upload(28L)` timer only drops 4640 → 4463 ms because that timer wraps just the layer loop — the big win is the 1.24 GB of embed/lm_head I/O that previously ran serially before device open is now entirely hidden under it. Bit-identical tokens 2303,220,220,16,13. |
| 2026-05-22 | iter6-device-argmax | (next commit) | ~17.7 | 4.51 | 2.25 | 9.24 | **Device-side argmax replaces 9.7-MB tile readback + CPU argmax.** `ops/argmax_row0/` (BRISC scan + NCRISC writer) reads row 0 of each `[Mt=1, Vt]` logits tile from DRAM, tracks BF16 max + index by sign-magnitude rules, writes 4 bytes back. `dec:logits_readback` 1044 ms → `dec:argmax` 33 ms + `dec:argmax_readback` <1 ms (-97%). Net decode 10.43 → 9.24 s (-11%). Wall delta vs paired iter5 baseline run: 18.9 → 17.7 s. Also confirms iter3 "any extra dispatch corrupts" was a kernel bug (not a runtime bug) — same dispatch infra runs fine. **Critical landmine resolved**: a 4-byte DRAM alloc shifted subsequent tile-format tensors off 2048-byte boundaries and silently broke matmul/embedding reads. The fix is to pad non-tile DRAM allocations up to one tile (2048 B). Filed as a runtime invariant; the qwen3_run T_argmax alloc uses 2048 B explicitly. Bit-identical tokens. |

## Iter 2 finding

Nt-split sharding works without any kernel reorganization on the
data-path side: a single `Nt_stride` runtime arg in reader (B-row
stride) and writer (C-row stride) is enough to write per-core slices
into a single global C buffer. No weight resharding required.
Downstream RMSNorm/RoPE/eltwise see C as a normal contiguous DRAM
buffer.

`dec:matmul_ffn` did not hit the predicted ~5ms because the FFN
shapes (gate/up at Nt=192 and down at Nt=64) still pay the ~2.5ms
dispatch floor per core. The remaining matmul time is now dominated
by the same per-dispatch overhead that bit iter 1 — i.e. a watermark
allocator + persistent matmul handles could push these into the
2–3ms range. That's a future iteration.

Other movers worth noting in this profile:
- `dec:logits_readback` is now 260ms/call × 4 = 1.04s (3.3%). The
  148-MB readback is real — only relevant if we switch from greedy
  CPU argmax to a device-side argmax (iter 4+ candidate).
- Per-call latency of every small op dropped from 3.5ms to 2.4ms
  even though we changed nothing in their kernels. This is a happy
  side effect of `tt-smi -r` being cold-state-fresher on this run.
  The 1.1ms drop is consistent across categories; it is not the
  matmul-grid win.

## Iter 6 finding: iter3's "any extra dispatch corrupts" was a kernel bug, not a runtime bug

While re-attacking device-side argmax (priority 1, 1.04 s decode bar),
we built a minimal reproducer (`tests/test_post_matmul_dispatch.cpp`)
that runs `matmul_grid(1×4)` → `silu(single core)` in a loop and
checks both outputs for cross-step stability. **It passes** — no
corruption.

Then we exercised the full qwen3_run pipeline with a `TT_FOIL_PROBE_EXTRA_DISPATCH`
flag that injects 1 or 10 extra silu dispatches between `dec:lm_head`
and `dec:logits_readback`. Greedy tokens stay bit-identical
(`2303, 220, 220, 16, 13`) in every case. So **the dispatch protocol is
correct**; what iter3 hit was almost certainly a bug in the iter3
device-side argmax kernel itself (likely in the CB-staged write or
the host→L1 path of the result slot — the 0x35858A86 garbage observed
was deterministic, suggesting one specific uninitialized region kept
showing through).

Implications:
- We can write a fresh `ops/argmax_row0` kernel with confidence the
  runtime will dispatch it correctly.
- The `release_kernels` + `reset_l1` pattern stays valid and does not
  introduce hidden state-leak across decode steps.

`tests/test_post_matmul_dispatch.cpp` stays in tree as a regression
guard so we catch any future dispatch-protocol regression early.

## Iter 3 abandoned: tried 8-core grid + device-side argmax

**8-core grid** (1×8 instead of 1×4): tried for FFN/QKV. Decode regressed
from 10.07s to 11.44s. With Mt=1, per-core compute is so small that the
extra dispatch overhead from 8 cores outweighs the parallelism gain.
1×4 stays the sweet spot for these shapes. (No commit — finding is recorded
here.)

**Device-side argmax** (replace 9.7-MB host readback + CPU argmax with a
4-byte device argmax + 4-byte read): hit a hard-to-debug chip-side
state-corruption bug. Adding **any** extra kernel dispatch on **any** core
inside the decode body — even a literal no-op `void kernel_main(){}` —
causes subsequent decode steps to produce different tokens against the
same inputs. Tested:

- BRISC-only kernel → diverges
- BRISC + NCRISC (mirroring embedding's RISC list) → diverges
- No-op kernel bodies → diverges
- Dispatch on (0,1) instead of (0,0) → diverges
- 50 ms host-side sleep after lm_head (in case writers were still
  draining NOC traffic) → diverges
- Dispatch inside the layer body (after the last `add` of layer 0)
  rather than after `lm_head` → diverges (even step 0 corrupts)
- CB-staged structure exactly mirroring `ops/embedding/` (one staging
  CB, one result CB drained by NCRISC) → diverges (writes 0x35858A86
  garbage to the result slot every step — looks like the CPU L1 store
  isn't reaching NCRISC's read of the CB, even with a `fence`)

So the bug isn't argmax-specific; it's that something about the chip
state after lm_head makes the next single-core dispatch unsafe. Not the
existing layer-body single-core dispatches that work fine throughout
iter 2 — those happen between matmul grid calls *inside* the layer.

Filed as a known runtime gap; reverted the argmax exploration in full.
Next iter should attack a different bar (prefill grid, weight
pre-bake) and revisit argmax once the dispatch behavior is understood.

## Iter 1 finding: L1 budget caps persistent-op design at ~855 KB

We attempted to lift all op handles out of the layer/step loops to amortize the 3.5ms per-dispatch ELF-NOC-write cost. The dispatch cache (`Device::resident_kernels`) was put in place to make that possible. But the L1 *user* arena on Blackhole is only ~855 KB per Tensix core (HAL `DEFAULT_UNRESERVED` size), and a single persistent RMSNorm handle for the hidden-dim shape (NCHt=1, Wt=64) eats ~522 KB by itself (four Wt-deep CBs × 128 KB). Add gqa_decode (~70 KB), embedding (4–128 KB), three RMSNorm shapes, matmul, two ropes, eltwise×3, and we overflow.

Mixing one-shot ops (which need `release_kernels` + `reset_l1`) with persistent ops doesn't work either: `release_kernels` clears the whole dispatch cache, and `reset_l1` rewinds the user-arena bump pointer over persistent CBs. Future work would need a watermark-aware L1 allocator + per-kernel cache invalidation by L1 address overlap.

Pivot: **iter 2 attacks the matmul kernel's own latency via multi-core sharding** (much larger wins available — FFN is 6.8s/dec, lm_head 1.75s/dec). The dispatch-cache infra + op_lib RTA setters stay in tree for whoever next tries the persistent-handle direction, and are also useful for callers that fit in the L1 budget naturally.
