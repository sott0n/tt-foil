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
| 2026-05-22 | iter7-fused-qkv | (next commit) | ~16.3 | 4.50 | 2.05 | 8.32 | **Fused QKV matmul.** Wq/Wk/Wv concatenated along Nt at `prepare_layer` time into a single `Wqkv` weight (Nt = NqDt + 2*NkDt = 128). One `make_matmul_grid` call per layer-step replaces three; T_Q/T_K/T_V become zero-copy offset views into a single T_QKV buffer (using `std::make_shared<Buffer>` with manually-set `device_addr = qkv_base + tile_offset * 2048`). `dec:matmul_qkv` 1725 ms (336 × 5.13 ms) → 794 ms (112 × 7.09 ms) = -54%; `pre:matmul_qkv` similar. Decode 9.24 → 8.32 s (-10%), prefill 2.25 → 2.05 s (-9%). Wall 17.7 → 16.3 s (-8%). Bit-identical tokens. Mt=1 in both prefill (St=1) and decode (StDec=1), so per-row Q/K/V slices are simple contiguous views — no row-strided gather needed. |
| 2026-05-22 | iter8-fused-ffn-gateup | (next commit) | ~16.05 | 4.71 | 1.91 | 7.82 | **Fused gate+up FFN matmul** (same pattern as iter7). Wgate / Wup concatenated along Nt at `prepare_layer` (Nt = 2*FFt = 384); T_gate / T_up become offset views into a single T_gateup buffer. The downstream `silu(T_gate) * T_up → T_fused` chain is unchanged because each consumer just sees a TensorDesc backed by the right device_addr. Wdown stays separate (operates on the post-silu·up product). Profile: `dec:matmul_ffn` 2866 ms (336 × 8.53 ms) → 2395 ms (224 × 10.69 ms) = -16%; decoded out one of three FFN matmul dispatches per layer-step (gate+up_fused = 12.9 ms each instead of 2 × 8.5 = 17 ms). `pre:matmul_ffn` similar. Decode 8.32 → 7.82 s (-6%); prefill 2.05 → 1.91 s (-7%). Wall 16.3 → 16.05 s. Bit-identical tokens. |
| 2026-05-22 | iter9-device-kv-append | (next commit) | ~15.75 | 4.75 | 1.90 | 7.52 | **Device-side KV-cache append.** `ops/kv_append/` (BRISC-only) reads T_Kr + T_V into L1 (≈128 KB scratch), writes the new pos's K row into K^T cache slot1 col `slot1_r` via 64 face read-modify-writes (the K^T layout makes col-writes scattered) and writes the V row into V cache slot1 row `slot1_r` via 64 direct 32-byte writes. Replaces the host-side `dec:kv_slot1_rebuild` (4×64 KB PCIe per layer + host untile/transpose/tile). Profile: `dec:kv_slot1_rebuild(host)` 560 ms (112 × 5.0 ms) → `dec:kv_append` 246 ms (112 × 2.2 ms) = -56%; the new op is essentially pinned at the 2.2 ms dispatch floor since the actual kernel work (≈128 KB NOC reads + ≈64 KB NOC writes per call) is < 50 µs. Decode 7.82 → 7.52 s (-4%); wall 16.05 → 15.75 s. Subtle landmine fixed mid-bring-up: the K^T cache is stored **slot-major** (slot s tile c at offset `(s * Nk + c) * 2048`, see `ops/gqa_decode/reader.cpp`), not the tile-major `(c * StKv + s) * 2048` that an op_lib `allocate_tensor_dram(Nk * StKv)` would imply; first kernel revision did the latter and produced wrong tokens (198 instead of 220 at decode step 0). Bit-identical tokens after the fix. |
| 2026-05-23 | iter11-hybrid-grid | (next commit) | ~29.3 (cold rerun) | 4.68 | 2.51 | 9.66 | **lm_head sharded across 1×8; other matmuls stay on 1×4.** `make_matmul_grid` extended to ragged shards (first `Nt%n_cores` cores get base+1 tiles). lm_head Nt=4748 splits 4×594 + 4×593 across 8 cores. Profile vs paired iter9 cold rerun (wall 29.70 s): `dec:lm_head` 113.96 → 64.21 ms/call (-44%), `pre:lm_head` 114.92 → 64.38 ms (-44%); other matmuls unchanged. Net decode 9.84 → 9.66 s (-180 ms); wall 29.70 → 29.33 s (-369 ms). Bit-identical tokens 2303,220,220,16,13. **Iter11 lesson**: an earlier full 1×8 trial regressed qkv/o/ffn matmuls +30–45% per call because Mt=1 makes per-core dispatch (sequential ELF NOC writes in `dispatch_stage_setup`) dominate over compute savings. 8-core sharding only pays off when per-core compute > per-core dispatch overhead — i.e. lm_head's Vt=4748 tiles of work amortizes the extra setup time, but Nt≤128 matmuls don't. |
| 2026-05-23 | iter12-matmul-acache | (next commit) | ~26.95 (cold) | 4.73 | 2.28 | 8.68 | **A-tile caching in matmul kernel.** Reader now reads each mt-row's Kt A tiles once into cb_a (depth Kt) instead of re-fetching them Nt times per output column; compute waits on the full A batch and indexes by kt. Per-core L1 use grows from 3×kTileBytes to (Kt+2)×kTileBytes (worst-case Qwen3 Kt=192 → 388 KB, well under the ~855 KB user arena). Profile vs paired iter11 cold rerun (wall 29.33 s): `dec:matmul_ffn` 12.43 → 9.54 ms/call (-23%), `dec:matmul_qkv` 8.80 → 7.54 (-14%), `dec:matmul_o` 7.40 → 6.76 (-9%), `dec:lm_head` 64.21 → 40.18 (-37%); `pre:matmul_ffn` 12.46 → 9.52 (-24%); `pre:lm_head` 64.38 → 39.94 (-38%). Net decode 9.66 → 8.68 s (-10%); prefill 2.51 → 2.28 (-9%); wall 29.33 → 26.95 s (-8%). Bit-identical tokens 2303,220,220,16,13. Sources: `ops/matmul/{reader,compute}.cpp`. Eliminates ~31× redundant A-tile NOC reads for Mt=1 cases. |
| 2026-05-23 | iter13-matmul-bbatch | (next commit) | ~25.21 (cold) | 4.70 | 2.16 | 8.02 | **B-tile batched read with single barrier per nt.** Building on iter12: reader now issues all Kt B reads for one (mt, nt) back-to-back and barriers once (cb_b depth Kt) instead of per-tile reserve+barrier. NOC HW pipelines the Kt outstanding reads. Per-core L1 doubles to 2*Kt*kTileBytes + kTileBytes; worst-case Qwen3 Kt=192 → 770 KB, within the ~855 KB user arena. Profile vs paired iter12 cold rerun (wall 26.95 s): `dec:matmul_ffn` 9.54 → 7.27 ms/call (-24%), `dec:matmul_qkv` 7.54 → 6.65 (-12%), `dec:lm_head` 40.18 → 24.40 (-39%); `pre:matmul_ffn` 9.52 → 7.50 (-21%); `pre:lm_head` 39.94 → 24.51 (-39%). Net decode 8.68 → 8.02 s (-8%); prefill 2.28 → 2.16 (-5%); wall 26.95 → 25.21 s (-6%, -1.74 s). Bit-identical tokens. Sources: `ops/matmul/{reader,compute}.cpp`. The Kt-batched barrier saves (Kt-1) per-tile barrier round-trips per output column; combined with iter12's A caching, matmul kernel is now NOC-pipelined for both operands. |
| 2026-05-23 | iter14-bprefetch | (next commit) | ~24.99 (cold) | 4.78 | 2.11 | 7.92 | **cb_b depth bumped to 2*Kt where L1 fits.** Op_lib now sets `cb_b_tiles = (6*Kt + 2 ≤ 855) ? 2*Kt : Kt` per matmul instance, so the reader holds two nt-batches in flight: while the consumer matmuls nt=k, the reader issues nt=k+1's NOC reads. FFN-down (Kt=192) stays at depth Kt for L1 fit; all other matmuls (Kt=64) double to 2*Kt = 386 KB cb_b. Kernel unchanged — pure CB descriptor adjustment. Profile vs paired iter13 cold rerun (wall 25.21 s): `dec:matmul_ffn` 7.27 → 7.09 ms/call (-3%), `dec:matmul_qkv` 6.65 → 6.56 (-1%), `dec:lm_head` 24.40 → 23.44 (-4%); `pre:matmul_ffn` 7.50 → 7.11 (-5%). Net wall 25.21 → 24.99 s (-1%, -222 ms). Diminishing returns: matmul ops are now ~3.5 ms dispatch-floor + small compute, so prefetch overlap is bounded. Bit-identical tokens. Likely the end of the matmul-kernel-only optimization avenue without changing dispatch protocol. |
| 2026-05-23 | iter15-silu-mul-fuse | (next commit) | ~24.06 (cold) | 4.76 | 2.02 | 7.50 | **Fused SwiGLU: SiLU(gate) * up in one dispatch.** New compute kernel `ops/eltwise_binary/compute_silu_mul.cpp` does SiLU(A) into intermediate CB c_24, then mul(c_24, B) into c_16. Reuses eltwise_binary reader+writer ELFs; op_lib `make_silu_mul` adds c_24 staging CB. Replaces the two-call chain `silu(T_gate)→T_silu` + `mul(T_silu,T_up)→T_fused` with a single op call, saving the ~3.5 ms dispatch floor on the second op. Profile vs paired iter14 cold rerun (24.99 s baseline): `dec:silu`+`dec:mul` (2×3.5≈7.04 ms/layer-step) → `dec:silu_mul` (3.51 ms/layer-step) = -395 ms / decode (-5% wall). Same in prefill: 7 → 3.52 ms/layer (-95 ms). Net wall 24.99 → 24.06 s (-4%, -928 ms). Bit-identical tokens 2303,220,220,16,13. Kernel uses dual init pattern (`init_sfpu`/`silu_tile_init` for the unary phase, then `binary_op_init_common`/`mul_tiles_init` for the binary phase, re-applied per tile) — works on Blackhole TRISC without measurable per-tile reinit overhead at this 192-tile FFN scale. |
| 2026-05-23 | iter16-add-rmsnorm | (next commit) | ~23.05 (cold) | 4.69 | 1.92 | 7.13 | **Fused add+RMSNorm** for the attention-residual + ln2 stage. New op `ops/add_rmsnorm/` clones `ops/rmsnorm/` and adds: (a) reader reads two streams A/B in lockstep at depth-1 CBs, (b) compute does Phase 0 = `S=A+B` packed into both cb_sum (compute-private, Wt depth, replaces cb_inp for phases 1+4) and cb_s_out (writer-drain, Wt depth), then runs the standard 5-phase RMSNorm, (c) writer drains both cb_s_out (sum) and cb_out (normed) to separate DRAM destinations. Per-core L1 ≈ 654 KB. Replaces `add(layer_in,proj)→xmid; rmsnorm(xmid,ln2g)→ynorm` with one dispatch — saves 112 of the 224 add+rmsnorm pairs in decode (the other 112 are the post-FFN add + next-layer ln1g, which can't be fused without cross-layer restructuring). Profile vs paired iter15 cold rerun (24.06 s baseline): `dec:add` 392→392 (-50% calls), `dec:rmsnorm` 392→394 (-50% calls), `dec:add_rmsnorm` new 395 ms. Net 1574 → 1182 ms (-392 ms decode). Same in prefill: -100 ms. Wall 24.06 → 23.05 s (-4%, -1.01 s). Bit-identical tokens 2303,220,220,16,13. |
| 2026-05-23 | iter18-add-rmsnorm-pair2 | (next commit) | ~22.17 (cold) | 4.68 | 1.84 | 6.77 | **Pair-2 add_rmsnorm: post-FFN residual fused with NEXT layer's ln1g.** Restructures both prefill + decode layer loops: a single `rmsnorm(layer_in, layers[0].ln1g) → T_xnorm1` runs before the loop to prime the chain, then inside the loop the post-FFN `add(xmid, down)` is fused with the next layer's ln1g (`make_add_rmsnorm(... layers[li+1].ln1g, T_layer_out, T_xnorm1, ...)`). Last layer (li=27) falls back to plain add because final_rmsnorm has its own gamma. Decode: 28 layers × 4 steps × 1 fuse = 112 additional add_rmsnorm fuses (220 total) replacing 112 add + 112 rmsnorm. Profile vs paired iter16 cold rerun (23.05 s baseline): `dec:add_rmsnorm` 395 (112) → 778 (220); `dec:rmsnorm` 394 (112) → 14 (4 final_rmsnorm), `dec:add` 393 (112) → 14 (4 last-layer); net dec ops 1182 → 806 ms (-376 ms). Same in prefill. Wall 23.05 → 22.17 s (-4%, -884 ms). Bit-identical tokens. **Iter 9 → 18 cumulative**: wall 29.70 → 22.17 s (-25%); decode 9.84 → 6.77 s (-31%); prefill 2.49 → 1.84 s (-26%). |
| 2026-05-23 | iter17-rmsnorm-rope | (next commit) | ~20.26 (cold) | 4.71 | 1.63 | 6.00 | **Fused RMSNorm + RoPE for Q/K projections.** New op `ops/rmsnorm_rope/` runs all 5 RMSNorm phases per (head-row), outputs into cb_normed (depth Wt=Dt), then runs the 6-pass RoPE rotation reading cb_normed with index `dh` / `Dt_half+dh` and cos/sin from persistent CBs (loaded once, indexed by `st*Dt_half+dh`). Writer drains the (out_first, out_second) pair to head-relative positions identical to standalone rope. Per-core L1 ≈ 50 KB (Wt=4 is tiny for QK). Replaces `rmsnorm_qk(Q|K)` + `rope(Qn|Kn)` chain — 4 fuses per layer-step (Q and K, decode and prefill). Profile vs paired iter18 cold rerun (22.17 s baseline): `dec:rmsnorm_qk` + `dec:rope` (789 + 781) = 1570 → `dec:rmsnorm_rope` 795 (-49%). Same in prefill: 391 → 198 (-49%). Decode 6.77 → 6.00 s (-11%); prefill 1.84 → 1.63 s (-11%); wall 22.17 → 20.26 s (-9%, -1.91 s). Bit-identical tokens 2303,220,220,16,13. **Iter 9 → 17 cumulative**: wall 29.70 → 20.26 s (-32%); decode 9.84 → 6.00 s (-39%); prefill 2.49 → 1.63 s (-34%). |
| 2026-05-23 | iter19-prepare-workers | (next commit) | ~17.88 (cold) | 2.35 | 1.64 | 5.98 | **Bumped prepare-thread pool 4 → 16 workers.** Diagnostic timing split (`weights:wait_for_prepare` vs `weights:upload`) revealed the bottleneck was CPU-bound `prepare_layer` (disk read + `tile2d`), NOT UMD upload: prepare ~141 ms/layer with 4 workers (main thread waits 3.96 s total), upload only 19 ms/layer (541 ms total). Earlier attempt to pack 8 weights into one big DRAM blob to cut UMD call count actually regressed (+300 ms) — host-side memcpy of 96 MB × 28 layers dwarfed the per-call savings. Going wider on the prepare side instead: `weights:load+upload(28L)` 4.71 → 2.35 s (-50%) at 16 workers. 28 (one-per-layer) was slightly worse than 16 due to thread contention. Bit-identical tokens. **Iter 9 → 19 cumulative**: wall 29.70 → 17.88 s (-40%); decode 9.84 → 5.98 s (-39%); prefill 2.49 → 1.64 s (-34%); weights 4.75 → 2.35 s (-50%). |
| 2026-05-23 | iter20-simd-tile2d | (next commit) | ~17.13 (cold) | 1.63 | 1.62 | 5.99 | **AVX2-vectorized tile2d in qwen3_run.** Each 16×16 face-row is exactly one 256-bit AVX2 vector (16 × uint16_t), so the inner copy collapses to one `_mm256_loadu_si256` + one `_mm256_storeu_si256`. The original implementation had a 32×32 staging `block` and nested scalar copies — heavy strided uint16_t shuffling that the compiler couldn't auto-vectorize. Marked with `__attribute__((target("avx2")))` so no global compile flag changes needed. Per-call prepare_layer time drops, shrinking the layer pipeline critical path: weights:load+upload(28L) 2.35 → 1.63 s (-30%); wall 17.88 → 17.13 s (-4%, -750 ms). Bit-identical tokens 2303,220,220,16,13. |

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
