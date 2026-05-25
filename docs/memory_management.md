# Memory Management: tt-foil vs. tt-metal

This document explains how device-side memory (L1 and DRAM) is managed in
tt-foil, contrasts it with tt-metal's allocator, and motivates the choice of
allocator. It also describes what the **Memory Report** produced by
[`tools/tt_foil_profile.py`](../tools/tt_foil_profile.py) does and does not
capture as a consequence of this design.

## 1. Two allocator designs

### tt-metal: free-list allocator (`tt_metal/impl/allocator/`)

tt-metal creates **one long-lived `Allocator`** per device, partitioned by
buffer type (`DRAM`, `L1`, `L1_SMALL`, …). The allocator runs a free-list
algorithm: a linked list of free regions sorted by address, with coalescing
on free.

```
device_open
└── Allocator (L1): [───────── 1.4 MiB free ─────────]

CreateBuffer(A, 512K)
└── Allocator (L1): [== A 512K ==|──── 896K free ────]

CreateBuffer(B, 256K)
└── Allocator (L1): [== A ==|= B 256K =|── 640K free ──]

free(B)        ← B's shared_ptr goes out of scope
└── Allocator (L1): [== A ==|## 256K hole ##|── 640K free ──]
                              ^^^^^ fragmentation
```

The allocator supports arbitrary alloc/free ordering and tracks fragmentation
explicitly. `largest_contiguous_free_block` is a real metric and answers the
practical question "can I still fit a buffer of size N?".

### tt-foil: bump allocator (`src/device.{hpp,cpp}`)

tt-foil creates **one per-core `L1Allocator`** and **one global `DramAllocator`**
that bump a pointer forward on every allocation. `free` is a no-op; the only
way to reclaim space is `reset()`, which rewinds the bump pointer to the
allocator's base (or to a `watermark` set by `pin_persistent`, per the iter21
contract — see [`include/tt_foil/runtime.hpp`](../include/tt_foil/runtime.hpp)).

```
op1 begin
└── bump: ptr = 0

buffer_alloc(A, 512K) → ptr = 512K
buffer_alloc(B, 256K) → ptr = 768K
buffer_alloc(C, 128K) → ptr = 896K

buffer_free(B)        ← bump pointer does NOT move back
└── bump: ptr = 896K   ← B's range is wasted until reset, but no "hole"

reset_l1()            → ptr = 0   ← everything goes away at once
op1 end
```

Fragmentation cannot occur: the pointer only goes forward within an op, and
`reset_l1` wipes the whole region between ops.

## 2. Why the designs differ

| Aspect                     | tt-metal                                       | tt-foil                                       |
| -------------------------- | ---------------------------------------------- | --------------------------------------------- |
| Allocator lifetime         | Per-device, lives until `CloseDevice`          | Per-op carve, reset on op boundary            |
| Free ordering              | Arbitrary (shared_ptr lifetime driven)         | LIFO-equivalent (one-shot reset)              |
| Fragmentation              | Real; tracked                                  | Cannot occur by construction                  |
| Per-bank tracking          | Yes (one free-list per DRAM bank)              | No (single bump per core)                     |
| `largest_free_block`       | Non-trivial; can be less than `total_free`     | Always equals `L1_size - watermark`           |
| Cost of `free`             | O(log N) merge into free-list                  | O(0) — no-op                                  |

The choice reflects the **runtime model** each project targets:

- **tt-metal** is a server-style runtime. A user creates several `Program`s,
  allocates large persistent buffers (model weights, KV cache), and enqueues
  programs many times. Buffers come and go in patterns the runtime cannot
  predict, so the allocator must handle arbitrary free ordering. Fragmentation
  is a real failure mode that ops can hit at runtime.

- **tt-foil** is a minimal-runtime "drive the chip from C++ directly" library
  targeting embedded / batch-style workloads. Each op `op_lib::*::run()`:
  1. Allocates the L1 buffers it needs (CB backing, scratch),
  2. Loads kernels, dispatches, waits for done,
  3. Releases everything and resets the L1 allocator.

  By construction, an op's L1 buffers all have the same lifetime, so a bump
  allocator with one `reset()` is sufficient and zero-cost. `pin_persistent`
  is the escape hatch for buffers that must live across op boundaries (e.g.
  KV cache slabs); it raises the post-`reset` watermark so those allocations
  survive.

The bump allocator is intentional and aligned with tt-foil's stated goal
("standalone C++ runtime for pre-compiled kernels", see [CLAUDE.md](../CLAUDE.md)):
no DRAM-interleaved buffers, no fast dispatch, no mesh — and correspondingly
no need for a free-list allocator.

## 3. Consequence: what the Memory Report can show

tt-metal's `MemoryReporter` (`tt_metal/detail/reports/memory_reporter.cpp`)
dumps the long-lived allocator's bookkeeping at program-compile time, which
exposes:

- Total allocatable / allocated / free per bank
- **Largest contiguous free block** per bank
- **Detailed block-level dump** showing the free-list shape

These metrics are meaningful *because the underlying allocator carries that
state*. They answer "is this device close to OOM on a specific bank?" and
"how badly fragmented is L1 after running these N programs?".

tt-foil's Memory Report (generated by `tools/tt_foil_profile.py` from
`tt_foil_memlog.csv`) reflects what its bump allocator can know:

| Column                       | Meaning                                                |
| ---------------------------- | ------------------------------------------------------ |
| `POOL`                       | `"Device L1"` or `"Device DRAM"`                       |
| `ALLOC_COUNT` / `FREE_COUNT` | Host-side `buffer_alloc` / `buffer_free` call counts   |
| `TOTAL_BYTES_ALLOCATED`      | Cumulative bytes ever handed out (not peak)            |
| `PEAK_LIVE_BYTES`            | Max of (live bytes from host's view) at any moment     |
| `LEAKED_BYTES`               | `alloc - free` byte balance at exit                    |

Useful for: "did I leak Buffer wrappers?", "what is the app-level peak L1
pressure across an op or a run?", "how often is the allocator hit?"

**Limitations** (a direct consequence of the bump allocator):

1. **Fragmentation is undefined.** Bump cannot fragment; the metric is omitted.
2. **`reset_l1` is invisible to the tracker.** If user code allocates buffers,
   does *not* free them, and then calls `reset_l1` / `release_kernels`, the
   tracker still believes those buffers are live. In practice tt-foil's op
   pattern is `alloc → use → free (via shared_ptr) → reset`, so the order
   prevents this from triggering, but it is a sharp edge.
3. **KERNEL_CONFIG region is not accounted for.** Kernel ELF text, runtime
   args, and the CB descriptor blob are written to L1 via a separate
   `kernel_config_allocs` bump allocator inside `kernel_load` /
   `dispatch_stage_setup`, bypassing `buffer_alloc`. Real on-chip L1 usage is
   therefore higher than `PEAK_LIVE_BYTES` reports.
4. **Per-core breakdown is not surfaced.** All cores' L1 allocations
   currently aggregate into a single `"Device L1"` pool (with the
   `(core.x, core.y)` encoded into the upper address bits to keep Tracy
   tracking unique). A future iteration could split into
   `"Device L1 (x,y)"` pools at low cost.

These are acceptable trade-offs for tt-foil's workload pattern: per-op
all-or-nothing allocation. If the runtime ever grows to support persistent
multi-program scenarios, replacing the bump allocator with a free-list
allocator would be the right move, and the Memory Report metrics could be
extended accordingly.

## 4. Performance implication: combining Performance + Memory reports

A direct consequence of the host-only bookkeeping is that tt-foil can capture
**both reports in a single run** with negligible cross-talk:

- Performance Report is a Tracy zone capture (`ZoneScopedN` on dispatch
  stages) → `.tracy` file → CSV export.
- Memory Report is a direct `fprintf` to `tt_foil_memlog.csv` from the
  `TF_ALLOC` / `TF_FREE` macros in `src/profiling.hpp`.

Neither path touches the chip; neither inserts barriers into the dispatch
sequence. Measured overhead on a qwen3 4-decode run is below run-to-run
variance: zone medians differ by < ~1% with memory logging on vs. off,
indistinguishable from chip thermal / PCIe jitter (see the verification log
in commit history if needed).

tt-metal traditionally needs separate runs for Performance Report and
Memory Report — not because of dispatch-path interference (the `MemoryReporter`
also runs entirely host-side and only at compile time) but because the two
tools are wired into different entry points (`python -m tracy -r` versus
`EnableMemoryReports()`) and their output paths overlap. tt-foil unifies them
in one wrapper script by design.

## 5. References

- `src/device.hpp` — `L1Allocator`, `DramAllocator` definitions
- `src/buffer.cpp` — `buffer_alloc` / `buffer_free` entry points, hosts the
  `TF_ALLOC` / `TF_FREE` instrumentation
- `src/profiling.hpp` — Tracy zone macros and the `emit_mem_event` CSV logger
- `tools/tt_foil_profile.py` — wrapper that produces both reports
- tt-metal: `tt_metal/impl/allocator/allocator.cpp`,
  `tt_metal/detail/reports/memory_reporter.cpp`
