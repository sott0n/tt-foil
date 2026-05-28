# R5 Feasibility Study — On-chip dispatcher for tt-foil

Date: 2026-05-24
Status: Pre-implementation feasibility / decision document
Author: Claude (under user direction)

## 1. Why this is on the table

After iter21 the standing benchmark is **wall 10.5 s, decode 5.94 s for 4 tokens
(1.5 s/token)** on a single Blackhole p150. tt-metal's Llama 3.1 8B on the same
silicon achieves **33 t/s/u = 30 ms/token** — a 50× gap on a 4× smaller model.

iter1–21 closed all the kernel-side gaps we could find within the existing
slow-dispatch architecture (matmul grid, A/B caching, op fusion,
persistent kernels for tiny ops, parallel weight upload, AVX2 tile2d).

The remaining bottleneck is structural: **per-dispatch ~3.3 ms floor**
that is dominated by **firmware-side BRISC state machine**
(RUN_MSG_INIT → setup_local_cb_* (5 RISCs) → subordinate-sync release →
user-kernel kick → RUN_MSG_DONE write-back). For our decode workload of
~285 dispatch/token × 4 tokens = 1140 dispatch/decode, this floor alone
is 1140 × 3.3 ms = **3.8 s of unavoidable cost** under the host-driven
RUN_MSG_* protocol.

tt-metal's runtime sidesteps this floor with **fast dispatch**: an
on-chip dispatcher kernel reads a command ring from system memory and
issues launches autonomously without round-tripping through the host.
R1–R4 in the prior plan cannot break this floor; only an on-chip
dispatcher can.

This document evaluates whether tt-foil can adopt a *minimal* version of
fast dispatch while preserving its core property (no `libtt_metal.so`
link, standalone HAL, single-chip Blackhole only).

## 2. tt-metal fast-dispatch architecture (what we'd be approximating)

Two specialised kernels, pinned to two cores (typically Tensix on a
single-chip configuration, or Ethernet cores on multi-chip):

```
                     PCIe (UMD writes)
                            │
   host hugepage cmd ring ──┴─►  PREFETCHER (cq_prefetch.cpp, 2.8k LoC)
                                      │ NOC
                                      ▼
                                DISPATCHER (cq_dispatch.cpp, 1.6k LoC)
                                      │ NOC
                       ┌──────────────┼──────────────┐
                       ▼              ▼              ▼
                   worker (0,0)   worker (0,1)   worker (0,2)  ...
```

Command set (`cq_commands.hpp`):
- **`CQ_PREFETCH_CMD_RELAY_LINEAR/PAGED`** — copy bytes from host hugepage
  (or DRAM) into dispatcher's L1.
- **`CQ_DISPATCH_CMD_WRITE_LINEAR/PAGED`** — dispatcher writes bytes to
  worker L1/DRAM.
- **`CQ_DISPATCH_CMD_GO_SIGNAL_MCAST`** — dispatcher mcasts a `RUN_MSG_GO`
  to a worker group.
- **`CQ_DISPATCH_CMD_WAIT`** — dispatcher waits for a counter to reach
  a value (worker DONE counters increment via NOC writes from BRISC).
- **`CQ_DISPATCH_CMD_EXEC_BUF`** — host points dispatcher at a "trace
  buffer" (pre-recorded command sequence in DRAM) and dispatcher
  replays it. **This is the key trace-mode primitive.**

Host side (`hardware_command_queue.cpp` + `system_memory_manager.cpp`,
~900 LoC core):
- Allocates two ring buffers in the host hugepage (issue & completion).
- Builds `DeviceCommand` byte-streams (see `device_command.hpp`, ~330 LoC)
  and copies them into the issue ring via cached `memcpy`.
- Bumps the prefetcher's "fetch_q" wptr via a single PCIe write.
- Polls the completion ring for events; the **completion ring is in host
  memory**, so polling does not cross PCIe.

End result on Llama 8B (host-perspective):
- Per-program launch ≈ a few hundred nanoseconds of host CPU (cached
  memcpy + 1 PCIe write).
- Multiple back-to-back launches stream through the dispatcher with
  no host-side gap.
- `EXEC_BUF` replays an entire pre-recorded decode-step layer in one
  host write — this is the "trace mode" we'd want for our decode loop.

## 3. Minimal subset feasible for tt-foil

The full tt-metal CQ supports many things we don't need:
- multi-chip routing / fabric / mesh / ethernet dispatch
- fast-dispatch *buffer* read/write (we keep host-direct UMD R/W for that)
- dynamic kernel JIT compilation
- semaphore / event / debug-data collection
- `tt::Cluster` integration (whole point of tt-foil is to avoid this)
- `MetalContext` lifecycle

Subset we *do* need for Qwen3-VL-2B decode:
- **`LAUNCH_PROGRAM(workers, launch_msg)`**: dispatcher writes launch_msg
  to each worker's `LAUNCH` ring slot, advances each worker's `wptr`,
  fires `RUN_MSG_GO` via the worker's `GO_MSG`.
- **`WAIT_DONE(workers)`**: dispatcher reads each worker's `GO_MSG.signal`
  until `RUN_MSG_DONE`. Reports completion back to host via a single
  L1 counter increment that host reads over PCIe.
- **`WRITE_RTA(worker, slot, bytes)`**: small inline write to a worker's
  RTA region (per-step RTAs that change between decode steps).
- **`EXEC_TRACE(dram_addr, size)`**: dispatcher reads command sub-stream
  from DRAM and processes it. Host pre-records the entire decode-step
  command sequence here.

What we **drop** vs tt-metal:
- Prefetcher core entirely: host writes commands directly into the
  dispatcher's L1 cmd ring via UMD `write_to_device`. The prefetcher
  exists in tt-metal to pre-stage program data (kernel binaries, CB
  configs) from host hugepage so the dispatcher doesn't stall on PCIe.
  For our workload, kernel binaries already live in worker L1 (cached
  via `resident_kernels`) and per-step RTAs are tiny (<64 B), so we
  can inline them in commands without a prefetcher.
- Completion ring in host memory: not strictly needed; we keep a
  single `completion_count` in L1 that host polls. One PCIe read /
  decode-step (not per op).
- All multi-CQ / multi-chip / fabric infrastructure.

## 4. Concrete design sketch

### 4.1 Layout

| Component | Where | Size | Owner |
|---|---|---|---|
| Dispatcher kernel ELF | (0,7) L1 KERNEL_TEXT | ~8 KB | tt-foil firmware build |
| Cmd ring | (0,7) L1 SCRATCH | 64 KB | dispatcher reads, host writes |
| Cmd ring wptr (host→device) | (0,7) L1 mailbox | 4 B | host writes |
| Cmd ring rdptr (device→host) | (0,7) L1 mailbox | 4 B | dispatcher writes |
| Completion counter | (0,7) L1 mailbox | 4 B | dispatcher writes, host reads |
| Trace buffer | DRAM bank N | ~256 KB | host records once, dispatcher reads many times |

### 4.2 Commands

```c++
enum CqCmd : uint8_t {
    CMD_LAUNCH      = 1,   // launch one or more workers
    CMD_WAIT_DONE   = 2,   // wait until all launched workers report DONE
    CMD_WRITE_RTA   = 3,   // small (<= 256 B) RTA write
    CMD_EXEC_TRACE  = 4,   // recurse into a DRAM trace blob
    CMD_NOTIFY_HOST = 5,   // bump completion_count
};

struct LaunchCmd {
    uint8_t  cmd;
    uint8_t  num_workers;
    uint16_t reserved;
    struct { uint8_t x, y; } workers[N];
    launch_msg_t launch_msg;     // inline copy
};

struct ExecTraceCmd {
    uint8_t  cmd;
    uint8_t  reserved[3];
    uint32_t dram_addr;
    uint32_t size_bytes;
};
```

### 4.3 Dispatcher kernel (BRISC on (0,7))

Single-file kernel, ~250 LoC target:

```c++
void kernel_main() {
    uint32_t rd = 0;
    while (true) {
        // wait until host wptr > rd
        while (read_volatile(&host_wptr) == rd) noop();
        // process commands up to host_wptr
        while (rd < read_volatile(&host_wptr)) {
            CqCmd* cmd = (CqCmd*)(cmd_ring + (rd % kRingSize));
            switch (cmd->op) {
                case CMD_LAUNCH:        handle_launch(...);  break;
                case CMD_WAIT_DONE:     handle_wait_done(...); break;
                case CMD_WRITE_RTA:     handle_write_rta(...); break;
                case CMD_EXEC_TRACE:    handle_exec_trace(...); break;
                case CMD_NOTIFY_HOST:   completion_count++; break;
            }
            rd += cmd_size(cmd);
            write_volatile(&dispatcher_rdptr, rd);
        }
    }
}
```

`handle_launch` writes `launch_msg` to each worker's LAUNCH ring slot
(via NOC unicast — we already have this in tt-foil's
`make_noc_unicast_addr`), advances the per-worker wptr (kept locally in
dispatcher), and writes `GO_MSG.signal = RUN_MSG_GO` with appropriate
`go_msg_idx`. NOC writes from one core to another are <1 µs each.

`handle_wait_done` polls each worker's `GO_MSG.signal` over NOC reads.
These are also <1 µs each — the dispatcher's poll loop runs at NOC
speed, not PCIe speed. Polling time becomes pure device-side kernel
execution.

`handle_exec_trace` switches `rd` to a DRAM-backed alternate ring,
processes commands from there until end-marker, then returns to the
main ring.

### 4.4 Host side

Add a new `FastDispatch` module (~400 LoC target):

```c++
struct FastDispatch {
    Device& dev;
    uint8_t* cmd_ring;         // mmaped staging, flushed to L1 in batches
    uint32_t host_wptr;        // local mirror

    void launch(const Kernel& k);
    void wait_done(const Kernel& k);
    void write_rta(const Kernel& k, std::span<const uint32_t> rta);
    TraceHandle record_begin();
    void        record_end(TraceHandle);
    void        exec_trace(TraceHandle);
    void        flush();        // PCIe-write cmd ring up to host_wptr, then bump device wptr
    void        sync();          // poll completion_count
};
```

Existing `dispatch_execute_multi` keeps working (it's the
slow-dispatch fallback we use during cold boot and tests). qwen3_run
decode loop opts into `FastDispatch` explicitly:

```c++
// Once before the decode loop:
auto fd = FastDispatch::open(*dev);
auto trace = fd.record_begin();
//   "fake-run" one decode step — all ol::execute calls route through
//   fd instead of dispatch_execute_multi.
run_one_decode_step(fd, ...);
fd.record_end(trace);

// Per token:
update_rta_patches(trace, pos);          // patch position-dependent RTAs
fd.exec_trace(trace);
fd.sync();                               // one PCIe read
uint32_t next = read_argmax();
```

## 5. Estimated effort

| Component | LoC | Risk | Notes |
|---|---:|---|---|
| Dispatcher kernel (BRISC) | ~250 | M | New custom firmware; uses existing tt-foil NOC primitives. Hardest part is debugging without printf — need scratch-pad tracing. |
| Dispatcher kernel build wiring | ~80 | L | New entry in `scripts/build_firmware.sh` / new `ops/cq_dispatch/` build script. |
| `FastDispatch` host module | ~400 | M | Owns cmd ring, command serializer, trace recorder. |
| Trace patching | ~150 | M | Identifying which RTA bytes vary per step and where they live in the trace blob. Needs op_lib API extension (`get_rta_patch_points(kernel)`). |
| `qwen3_run` integration | ~100 | L | Wrap decode loop. |
| Regression coverage | ~300 | M | New `tests/test_fast_dispatch.cpp`: cold-launch single kernel via FD, multi-kernel concurrent, trace record/replay, RTA patching. Keep all existing slow-dispatch tests green. |
| Diagnostics / NOC scratch tracing | ~100 | M | For dispatcher debug. |
| **Total** | **~1380** | — | (lower than the 2000 LoC earlier estimate because we skip prefetcher + multi-CQ + buffer R/W paths) |

Time estimate: **3–5 focused sessions** for first end-to-end Qwen decode
on FD (assuming each session ≈ 1 implementation push of one of the
rows above + bringup debug). Plus a 6th session for trace patching to
be perf-effective.

## 6. Expected wall reduction

Per-op host time eliminated entirely (replaced with one cmd-ring write
shared by N ops). Dispatcher-side NOC ops take ~1 µs each so the
per-op dispatcher cost is ~5 µs. Worker firmware floor remains
(setup_local_cb_*, subordinate sync) but **only once per "program"** —
trace replay does not re-invoke the full RUN_MSG_INIT path because
the dispatcher fires GO directly on top of pre-staged launch_msgs.

| Bench | wall | decode | source |
|---|---:|---:|---|
| iter21 baseline | 10.53 s | 5.94 s | current |
| FD without trace | ~5 s | ~1.5 s | per-op floor ~1.5 ms (NOC sync, not PCIe) |
| FD + trace replay | ~3 s | ~0.5 s | per-op floor ~0.5 ms (back-to-back firmware launches inside trace) |

If the firmware floor turns out to be intrinsic even with FD (i.e.
`setup_local_cb_*` is unavoidable per launch), we land at the ~5 s
side. If trace replay lets us reuse CB setup across calls (matmul vs
non-matmul switching), we get closer to 3 s.

Either outcome is a **3–5× wall improvement** over iter21. Both are
multiples better than the 10–20% R1–R4 could deliver.

## 7. Risks and unknowns

### 7.1 Hard risks

- **Dispatcher kernel debug is painful.** No printf on BRISC, no GDB.
  Mitigation: dedicated L1 scratch region for "phase markers"
  (already used in tt-foil per CLAUDE.md "NOC checkpoint pattern").
  Allocate ~256 B of scratch in the dispatcher's L1; bump-write
  sentinels at each `handle_*` entry. Host reads after timeout.

- **Worker firmware semantics under dispatcher-driven launch.**
  Currently tt-foil sends `RUN_MSG_RESET_READ_PTR_FROM_HOST` per op
  (see `dispatch.cpp:151`) because firmware otherwise mis-handles
  `setup_local_cb_*` writes (long comment in source). We must verify
  the dispatcher-driven path doesn't require this reset, or replicate
  it as a dispatcher cmd. tt-metal's dispatcher mostly skips it
  (workers stay in a steady "wait for launch_msg" state). Bringup
  experiment: send 2 back-to-back launches via FD without reset, see
  if second kernel's CB interface is sane.

- **L1 budget on (0,7).** Dispatcher needs ~64 KB cmd ring + ~8 KB
  kernel text + ~256 B scratch ≈ 73 KB. Doesn't conflict with lm_head
  (which uses (0,0..7) in 1×8 grid) — wait, **it does**. lm_head's
  per-core arena is ~107 KB (855 KB / 8) and our dispatcher would
  squat ~73 KB of that. Either move lm_head to 1×7 grid (slight perf
  hit) or pin dispatcher to a row-1 core (1, anything) since
  matmul/lm_head use row-0 only on Blackhole p150 (verify HAL).

- **Trace patch points.** Identifying every byte in the trace that
  depends on `pos` / `cur_token`. Easy to miss one and get wrong
  outputs that look "almost right". Mitigation: bit-identical token
  guard in `bench/bench.sh` already catches this. Plus: instrument
  `set_*_args` per op_lib to emit "trace patch annotations" instead
  of writing bytes when in record mode.

### 7.2 Soft risks

- **Scope creep into tt-metal-like infrastructure.** Once we have a
  cmd ring and trace replay, the temptation is to add buffer R/W
  cmds, semaphores, event signals, multi-CQ, etc. Stay disciplined:
  Qwen3 decode is the only target. Anything else stays slow-dispatch.

- **`libtt_metal.so` re-entry.** The CB blob layout, launch_msg
  layout, and CQ command layout all come from tt-metal headers. As
  long as we read them at host build time (via HAL or directly via
  headers, no runtime link), we stay standalone. The vendored
  `ll_api::memory` precedent shows this is doable. The new firmware
  build will need `cq_commands.hpp` from tt-metal source tree — same
  pattern as `chlkc_list.h`.

- **Bringup may stall for days on firmware issues.** This is the
  highest-risk component. If after 2 sessions the dispatcher kernel
  isn't reliably launching even one worker, drop to a smaller
  R3-α-style increment.

## 8. Recommendation

**Proceed, but with explicit go/no-go gates.**

### Gate 1 (after ~1 session): Single-kernel FD launch works
- Deliverable: `tests/test_fast_dispatch_smoke.cpp` launches a
  no-op kernel on (0,0) via FD on (0,1), waits for completion,
  verifies done-counter increments.
- **Pass**: per-launch host time < 50 µs (vs slow-dispatch ~2 ms).
- **Fail**: revisit subordinate-sync interaction; consider falling
  back to R3-α.

### Gate 2 (after ~3 sessions): Qwen3 decode runs on FD without trace
- Deliverable: bench/bench.sh iterR5a runs end-to-end with bit-identical
  tokens (`2303,220,220,16,13`).
- **Pass criterion**: wall ≤ 7 s.
- **Fail**: ~5 ms / op floor still present → R5 has fundamentally
  same problem as R1–R4; abandon and accept iter21 as ceiling.

### Gate 3 (after ~5 sessions): Trace replay enabled
- Deliverable: bench/bench.sh iterR5b shows decode in trace mode.
- **Stretch goal**: wall ≤ 4 s, decode ≤ 1.5 s.

### Branch hygiene
- Work on a feature branch `r5-fast-dispatch` off `main`. Do NOT
  merge until at least Gate 2 passes.
- Keep slow-dispatch as the default path. FD is opt-in via a
  `Device::enable_fast_dispatch()` flag, gated initially behind
  `TT_FOIL_FAST_DISPATCH=1`.
- New firmware target `tt_foil_cq_dispatch` is independent of
  `tt_foil_firmware`; can be built or skipped without affecting
  slow-dispatch tests.

### Non-goals (for this branch)
- Prefill speedup (1.6 s is below 15% of wall; ROI low).
- Multi-chip / mesh.
- Buffer R/W via FD (keep host-direct UMD R/W).
- Backwards compatibility with anyone outside qwen3_run (no other
  models exist yet; this is greenfield).

## 9. Open questions

1. **Where exactly does the 3.3 ms firmware floor come from?** It would
   be worth one targeted measurement: launch a no-op BRISC kernel
   (`void kernel_main() {}`) via slow-dispatch in a tight loop, measure
   per-op wall time. If the floor is still ~3 ms, it's purely firmware.
   If it drops to ~1 ms, then user-kernel exec (even tiny) accounts
   for the rest and FD's expected gains may be smaller than projected.
   *Defer this to Gate 1.*

2. **Does the dispatcher kernel need NCRISC too?** BRISC on (0,1) (or
   wherever) handles cmd parsing + NOC writes; it has enough bandwidth.
   NCRISC could be added later if we want overlap (cmd-parse on BRISC
   while NCRISC issues writes). Start with BRISC-only.

3. **Trace storage budget.** Worst-case trace blob ≈ (per-op cmd ~64 B
   inline launch_msg + small RTAs) × 285 ops + per-step patch table.
   ~25 KB / decode-step. Fits comfortably in a single DRAM allocation.

4. **Should we land R3-α (persistent pin extension) as an interim
   improvement while R5 is in bringup?** R3-α is ~200 LoC and gives
   ~300 ms wall improvement; useful if R5 takes longer than expected.
   Decision: yes, land R3-α first as the "safety net", then start R5
   on top of it.

## 10. Decision requested

Sign-off on the staged approach:
- Land R3-α (persistent decode-core extension) first — modest gain, low risk.
- Branch `r5-fast-dispatch` and pursue Gates 1 → 2 → 3.
- Abort at any gate that fails per its criteria; record finding in
  `models/qwen3_vl_2b/bench/PERF_HISTORY.md` and `docs/`.
