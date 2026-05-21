# qwen3_run performance history

Standing benchmark: prompt `[3838, 374, 279, 6722, 315, 6435] + 26×PAD` (i.e. "What is the capital of Tokyo" + endoftext padding), `num_decode=4`, device 0, fresh `tt-smi -r` before each run.

Run with `bench/bench.sh <tag>`. Raw profile dumps land in `bench/runs/` (gitignored).

Tokens generated for this fixed prompt (must stay bit-identical across optimizations): `2303, 220, 220, 16, 13`.

| date | tag | commit | wall (s) | weights (s) | prefill (s) | decode (s, 4 step) | notes |
|------|-----|--------|----------|-------------|-------------|--------------------|-------|
| 2026-05-21 | baseline | b62624a + profiling | ~39.3 | 17.40 | 3.85 | 18.04 | per-op dispatch floor 3.5ms across small ops. Hot spots: dec:matmul_ffn 6.83s (336 × 20.3ms), dec:lm_head 1.75s (4 × 438ms), dec:matmul_qkv 2.23s (336 × 6.6ms). |
| 2026-05-21 | iter1-infra | (this commit) | ~23.3 | 3.93 | 3.35 | 16.02 | No perf change vs baseline — adds dispatch-cache infra + op_lib RTA-only setters + bench infra. Weights load drop is filesystem-cache warmth (file is gitignored, no longer cold). Bit-identical tokens 2303,220,220,16,13. |

## Iter 1 finding: L1 budget caps persistent-op design at ~855 KB

We attempted to lift all op handles out of the layer/step loops to amortize the 3.5ms per-dispatch ELF-NOC-write cost. The dispatch cache (`Device::resident_kernels`) was put in place to make that possible. But the L1 *user* arena on Blackhole is only ~855 KB per Tensix core (HAL `DEFAULT_UNRESERVED` size), and a single persistent RMSNorm handle for the hidden-dim shape (NCHt=1, Wt=64) eats ~522 KB by itself (four Wt-deep CBs × 128 KB). Add gqa_decode (~70 KB), embedding (4–128 KB), three RMSNorm shapes, matmul, two ropes, eltwise×3, and we overflow.

Mixing one-shot ops (which need `release_kernels` + `reset_l1`) with persistent ops doesn't work either: `release_kernels` clears the whole dispatch cache, and `reset_l1` rewinds the user-arena bump pointer over persistent CBs. Future work would need a watermark-aware L1 allocator + per-kernel cache invalidation by L1 address overlap.

Pivot: **iter 2 attacks the matmul kernel's own latency via multi-core sharding** (much larger wins available — FFN is 6.8s/dec, lm_head 1.75s/dec). The dispatch-cache infra + op_lib RTA setters stay in tree for whoever next tries the persistent-handle direction, and are also useful for callers that fit in the L1 budget naturally.
