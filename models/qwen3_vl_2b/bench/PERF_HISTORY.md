# qwen3vl_run (Vision task) performance history

Standing benchmark for the **vision-image-pad prefill + greedy decode** path
of `qwen3vl_run`. Distinct from the text-only `HISTORY.md` because the
prefill is 3× larger (96 tokens = 64 IMAGE_PAD + 6 text + 26 pad, kSt=3
vs text-only's kSt=1), the visual-embed injection adds a host-side path,
and the prefill argmax / KV-cache snapshot bars are much heavier
relative to wall.

Run with `models/qwen3_vl_2b/bench/bench_vl.sh <tag> [num_decode]`. Default
`num_decode=4`. Raw profile dumps land in `models/qwen3_vl_2b/bench/runs/`
(gitignored).

Standing input:
- prompt = 64×`151655` (IMAGE_PAD) + `3838,374,279,6722,315,6435` ("What is the capital of Tokyo") + `151643`×26 (endoftext pad) = 96 tokens
- visual embeds = seeded-random bf16 `[64 × kH=2048]` (np seed 42)
- device 0, `tt-smi -r 0` before each run

Tokens generated for this fixed prompt (must stay bit-identical across
optimizations): `9640, 9640, 9640, 9640, 9640` (1 prefill-emitted + 4 decode).
The repeated 9640 reflects the seeded-random embeds — they don't carry any
real visual semantics, but greedy decode is deterministic so this acts as
the correctness oracle.

**Wall (s)** is the binary's own `wall_ms` line — measured from
`Clock::now()` at main entry through `decode_loop_end`, exactly one value
per run. **All deltas below are paired same-session vs the previous row's
re-measured baseline**, not absolute numbers chased across re-runs (the
chip's cold/warm state shifts the wall ±3% run-to-run; paired comparison
cancels that).

| date | tag | commit | wall (s, median of 3) | weights (s) | prefill (s) | decode (s, 4 step) | notes |
|------|-----|--------|-----------------------|-------------|-------------|--------------------|-------|
| 2026-05-27 | baseline | 054028e | 5.28 (5.26/5.28/5.42) | 1.37 | 1.08 | 1.61 | First captured VL baseline. Wall breakdown: device_open+UMD+teardown ~1.20 s, weights:load+upload(28L) 1.37 s, prefill:total 1.08 s, decode:total 1.61 s (sum ≈ wall, excl. the post-prefill host argmax 210 ms which sits between prefill:total and decode:total scopes). Largest tractable host bars (both in prefill region): pre:kv_cache_snapshot(host) 455 ms (28 × 16.26 ms, 8 small PCIe transactions/layer), pre:logits_readback+argmax(host) 210 ms (9.4 MB readback + 151,936-element BF16 host argmax). Top device bars: 7 dec:matmul_* (sum 1214 ms / 4 steps), dec:lm_head 62 ms (4 × 15.6), pre:lm_head 16 ms (1× 15.5). All matmuls are at the post-iter-percore-membar dispatch floor of 1.2-2 ms. Tracy: TF_dispatch/wait_done total ≈ 1.07 s (real device exec); host dispatch overhead (setup+send_reset+fire_go) ≈ 102 ms across 4954 dispatches. Memory: Device DRAM 3.5 GB peak (weights), L1 ~790 KB/core × 4 matmul cores. |
| 2026-05-27 | iter1-prefill-device-argmax | (this commit) | **5.07** (5.07/5.06/5.09) | 1.37 | 1.08 | 1.61 | **Generalized `argmax_row0` to accept row_in_tile and switched prefill to use device argmax.** The BRISC scanner in `ops/argmax_row0/reader.cpp` branches on face0/1 (row<16) or face2/3 (row≥16) and uses row-within-face offset `(row&15)*32`. Added optional `row_in_tile` (default 0) to `make_argmax_row0`/`set_argmax_row0_args` (existing decode callers are unchanged). Replaced `pre:logits_readback+argmax(host)` 210 ms (9.4 MB tile readback + 151,936-element BF16 host argmax) with `run1("pre:argmax", make_argmax_row0(..., row_in_tile=31))` + 4-byte readback. Profile vs paired baseline: pre:logits_readback+argmax(host) 210 ms → **pre:argmax 6.3 ms** (-97%, -203 ms). Wall 5.28 → 5.07 s (-205 ms, **-3.9%**). Bit-identical tokens 9640×5. Paired iter1 spread 30 ms (5.06/5.07/5.09) is below baseline noise 160 ms (5.26/5.28/5.42) — signal is clear. LOC ~50 (kernel ~10, op_lib ~10, qwen3vl_run ~10, header ~5). Next hot bar: pre:kv_cache_snapshot(host) 455 ms. |
| 2026-05-27 | iter2-prefill-device-kv-snapshot | (this commit) | **4.62** (4.61/4.62/4.62/4.63) | 1.37 | 0.62 | 1.61 | **Added `ops/kv_snapshot/` (BRISC-only) to replace host-side kv_cache_snapshot in prefill.** The kernel takes block-major K^T from `pre:transpose` (`T_Kt_pre`, tile (c,s) at `(c*kSt+s)*2048`) and slot-major V from the gqa-attention path (`T_V_pre`, tile (s,c) at `(s*kNkDt+c)*2048`), slurps both into L1 with one NOC read each, re-positions K^T tiles to slot-major `(s*kNkDt+c)*2048` in `T_Kt_cache`, copies V tiles to `T_V_cache`, and zero-fills decode slots (kSt..kStKv) with a reused 2 KB L1 zero tile. `make_kv_snapshot` caches kSt/kNkDt/kStKv via OpCache; runs 28 layers × 1 dispatch on dedicated core (1,6) `kPreKvSnapCore` (booted at open_device). Profile vs paired iter1: pre:kv_cache_snapshot(host) **455 ms (28×16.26)** → **pre:kv_snapshot 5.7 ms (28×0.20)** = **-98.7%**. Prefill total 1077 → 624 ms (-453 ms, -42%). Wall 5.07 → 4.62 s (-450 ms, **-8.9%**). Bit-identical tokens 9640×5. LOC ~210 (kernel 117, op_lib 113, qwen3vl_run -30+10, header 21, CMakeLists 1). Removed host helper `kt_to_slot_major` and related constants. **Bench noise note**: 1 of 5 consecutive runs hit 9.96 s (all phases ×2 — thermal/contention spike); the 3 post-cooldown runs (4.61/4.62/4.62) were stable at 20 ms spread. Allow 1 outlier and use median when pairing. **Cumulative iter1+iter2**: wall 5.28 → 4.62 s (-660 ms, **-12.5%**), prefill 1.08 → 0.62 s (-42%); both host hot bars resolved. Remaining: weights load+upload 1.37 s (26%), decode 1.61 s (31%, all ops at per-core membar dispatch floor). |
