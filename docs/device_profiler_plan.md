# Device-Side Cycle Profiler for tt-foil — Implementation Plan

Status: **Implemented** (Phases 1–7). The end-user guide moved to
[`profiling.md` §4 "Device Performance Report"](profiling.md). This
document is kept as the original design record — the phase breakdown,
trade-offs, and the lessons learned along the way.
Target: Blackhole, slow dispatch (CLAUDE.md current scope)
Companion docs: [`profiling.md`](profiling.md), [`memory_management.md`](memory_management.md)

## 1. Context

The host-side profiler (improvements 1–9, currently shipped) tells us *which
op* is slow and *whether* a dispatch is PCIe-bound or chip-bound. It cannot
say *why* a chip-bound dispatch is slow: we only see one number
("wait_done = 270μs") instead of the per-RISC breakdown of where those
cycles went.

```
host: fire_go ──→ [chip is opaque] ──→ done received
                  └── 270μs ──┘            wait_done report

what we want to see instead:
  BRISC   │█ args │█ noc_read █│░ wait_cb ░░│ ...
  NCRISC  │█ args │░ wait ░░░░│█ noc_write █│
  TRISC1  │       │░░░░░░░░░░░│ █ math ███████│  ← critical path
```

The device profiler reads RISC-V wall-clock registers at zone enter/exit
and stores the cycle stamps in an L1 region the host reads back after the
dispatch finishes. This proposal adapts tt-metal's existing
`kernel_profiler.hpp` rather than rewriting one.

## 2. What we get from tt-metal for free

tt-foil already builds against tt-metal's source tree, so several pieces
are reusable as-is:

| Component | Path | Notes |
|---|---|---|
| Macros | `tt_metal/tools/profiler/kernel_profiler.hpp` | `DeviceZoneScopedN`, `DeviceTimestampedData`, no-op when `PROFILE_KERNEL` is undefined |
| Buffer layout | `tt_metal/hw/inc/hostdev/dev_msgs.h` (`profiler_msg_t`) | Lives **inside the mailbox struct** that tt-foil already initialises |
| HAL exposure | `HalL1MemAddrType::PROFILER` | `hal->get_dev_addr(TENSIX, PROFILER)` returns the buffer base; size via `get_dev_size(PROFILER)` |
| Hardware regs | `RISCV_DEBUG_REG_WALL_CLOCK_L/H` in `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h` | Two 32-bit reads; reading L latches H |
| Constants | `tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h` | `PROFILER_L1_VECTOR_SIZE`, marker indices, control-vector layout |
| Sync kernel | `tt_metal/tools/profiler/sync/sync_kernel.cpp` | Reference for our calibration kernel |

The buffer is **per-core**, **per-RISC slots**, and the address is
HAL-derived — none of that needs to change.

## 3. What tt-foil has to build

Five distinct pieces, in order of dependency:

### A. Firmware: compile with `PROFILE_KERNEL` defined

- `scripts/build_firmware.sh` already produces our 5 firmware ELFs.
  We add an opt-in `-DPROFILE_KERNEL=1` (or appropriate bitmask) to
  the SFPI g++ command line when the new CMake option is on.
- Firmware-internal calls like `kernel_profiler::init_profiler()` need
  to fire at FW boot — typically called from `do_crt1` or BRISC `main`.
  We patch this in via the same mechanism we use for the
  `weakened_elf` workflow (or via SFPI compile-line `-include` of a
  tiny shim header).
- Where to put `init_profiler()` and `finish` calls in our firmware:
  to be determined by reading `tt_metal/hw/firmware/src/brisc.cc`
  and equivalents — this is the single biggest risk area, see §6.

### B. Kernel build: same `-DPROFILE_KERNEL` flag

- `examples/*/build_kernels.sh` and `ops/*/build.sh` need the flag
  too, plus `-I` to `kernel_profiler.hpp`'s directory.
- Without the flag, `DeviceZoneScopedN` expands to nothing — kernels
  not updated for profiling still build cleanly.
- We expose **one new macro for kernel authors**:
  `TF_DEVICE_ZONE("name")` in a new
  `include/tt_foil/device_profiling.h` header. It is just an alias
  for `DeviceZoneScopedN` so we control the public API surface.

### C. Dispatch path: zero changes during launch, read-back after DONE

- `dispatch_stage_wait_done` already polls until `RUN_MSG_DONE`. After
  it returns, we add an optional read-back step in
  `dispatch_execute_multi`: for each kernel's core, read
  `sizeof(profiler_msg_t)` bytes from `hal->get_dev_addr(TENSIX, PROFILER)`
  via `umd_driver->read_from_device`. This is gated on
  `dev.profiler_enabled` so non-profile builds pay nothing.
- The read is **one PCIe transaction per core**, sized ~16 KB. For a
  60-dispatch cifar10 run this adds ~1 ms total — acceptable for an
  opt-in mode. For qwen3-scale this could add 1–2 % wall, also fine.

### D. Host parser: turn raw L1 dump into structured events

- New file `src/device_profile.cpp` reads the `profiler_msg_t` layout,
  walks each RISC's `data[]`, decodes the packed
  `[1 bit | 19 bits timer_id | 12 bits cycle_high]` + 32 bits
  `cycle_low` marker pairs, and produces a `std::vector<DeviceZone>`.
- Each `DeviceZone` carries: `core (x,y)`, `risc_id`, `zone_name_hash`,
  `start_cycle`, `end_cycle`.
- The zone-name hash table is host-side: kernels use string literals,
  the macro hashes them at compile time (16-bit FNV via
  `Hash16_CT` in tt-metal's header). We maintain a manually-curated
  map from hash → name in `src/device_profile_zones.cpp`. Adding
  a new zone in a kernel means adding one line to this map. (Better
  long-term: emit the hash→name table at kernel build time. Out of
  scope for v1.)

### E. Memlog integration + CSV

- Append events to a new file `tt_foil_device_zones.csv` with
  columns: `core_x, core_y, risc, zone_name, start_ns, end_ns,
  duration_ns`. Cycle → ns conversion uses the clock sync from step F.
- `tt_foil_profile.py` aggregates by `(risc, zone_name)` into a new
  report `reports/tt_foil_device_perf.csv` analogous to
  tt_foil_perf_results.csv.

### F. Clock sync

- One-time calibration during `open_device`: load a tiny
  `device_sync_kernel` on every booted core. It reads
  `WALL_CLOCK_L/H` once, writes the 64-bit value into a known L1
  scratch slot, exits.
- Host records `steady_clock::now()` immediately before fire_go and
  immediately after wait_done, reads the device value, stores the
  triplet `(host_t_pre, host_t_post, device_cycle)`. Repeats N=10
  times to average out PCIe latency.
- Assume linear mapping: `host_ns = a * device_cycle + b` (Blackhole
  cycle counter is a free-running 1 GHz timer, so `a` is fixed near
  1.0 — we only need `b`). One per-core offset is enough; cores share
  a chip clock.
- Save the constants in `Device` and use them at parse time.

## 4. CMake + build wiring

New option in top-level `CMakeLists.txt`:

```cmake
option(TT_FOIL_DEVICE_PROFILER
    "Enable device-side cycle profiling (firmware + kernels) — implies TT_FOIL_ENABLE_TRACY" OFF)
```

When ON:
- Adds `TT_FOIL_DEVICE_PROFILER_PROFILE_KERNEL_BITMASK=1` to
  `target_compile_definitions(tt_foil ...)` (host side knows
  read-back must happen)
- Passes `-DPROFILE_KERNEL=1` and the relevant `-I` to
  `scripts/build_firmware.sh` and `examples/*/build_kernels.sh`
  through a new env var `TT_FOIL_PROFILE_KERNEL`
- Asserts `TT_FOIL_ENABLE_TRACY=ON` (device zones feed into the same
  Performance Report pipeline)

When OFF (default):
- Firmware + kernels built without profiler; host-side read-back code
  compiled out via `#if defined(TT_FOIL_DEVICE_PROFILER)` guards.
- Zero overhead, matches the current zero-overhead Tracy story.

## 5. Phase breakdown (implementation order)

### Phase 1 — Plumbing without profiling output (1 day)
- Add CMake option and env-var plumbing
- Build firmware with `PROFILE_KERNEL=1` defined but no kernel calls
  the macros yet. Confirm firmware still boots and `test_add_kernel`
  still passes.
- Goal: prove we can flip the flag without breaking anything.

### Phase 2 — Capture buffer + raw dump (1 day)
- Add the host read-back in `dispatch_execute_multi`
- Print the raw `profiler_msg_t` bytes for one core, one dispatch
- Add `DeviceZoneScopedN("test")` to one kernel (e.g. `add_two_numbers`
  reader.brisc.cc) and confirm the dump shows non-zero markers
- Goal: end-to-end byte path works.

### Phase 3 — Parser + CSV (1 day)
- `device_profile.cpp` decodes the buffer into `DeviceZone` events
- Emit raw `tt_foil_device_zones.csv` per dispatch
- Verify start_cycle < end_cycle and durations look plausible
- Goal: structured device data flowing out.

### Phase 4 — Clock sync (1 day)
- Implement the calibration kernel + host loop
- Convert cycles → ns
- Cross-check: total wait_done time should be ≥ max device zone duration
- Goal: device zones expressible on the host wall clock.

### Phase 5 — User-facing zones in real ops (0.5 day)
- Add `TF_DEVICE_ZONE` macros to the conv_3x3 family kernels in
  models/cifar10_resnet20 (one math zone, one noc-wait zone)
- Re-run cifar10 profile, examine the breakdown
- Goal: validate the bottleneck-analysis loop end to end on a real
  workload.

### Phase 6 — Aggregator + report integration (0.5 day)
- `tt_foil_profile.py` reads the device_zones CSV and produces the
  per-RISC, per-zone aggregate report
- Sort by total cycles to surface the chip-side hot path

### Phase 7 — Docs (0.5 day)
- Update `docs/profiling.md` with the new option, the new report
  file, and the kernel-author convention.
- Add a "limitations" note: 16 KB / core L1 cost, hash collision
  caveat, requires kernel-source changes to be useful.

Total estimated: **4–5 days** of focused work.

## 6. Risks and unknowns

1. **Firmware boot path** — Where exactly `init_profiler()` and
   `finish` need to land in `brisc.cc` is the highest-risk question.
   Worst case: we have to patch the firmware source tree (= maintain
   a small `.patch` file in `scripts/`). Acceptable but ugly.

2. **Hash collisions** — `Hash16_CT` is 16-bit FNV. Two distinct zone
   names can collide; the manual hash→name map will be silently
   wrong. Mitigation: a build-time check that hashes are unique
   across a kernel build, fail-fast if not.

3. **Buffer overflow** — `PROFILER_L1_VECTOR_SIZE` (512 entries =
   256 markers per RISC) is plenty for a single dispatch but could
   overflow if a kernel hits a hot loop with a zone inside. The
   tt-metal macro silently drops markers past the buffer end. We
   should at least flag this in the host parser when `wIndex` is
   at the cap.

4. **Coexistence with other mailbox users** — `dprint`, `watcher`,
   and a hypothetical future fast-dispatch all share the mailbox
   struct. The profiler buffer is part of `mailboxes_t` so the
   layout is the same as tt-metal's, but we should sanity-check
   no other tt-foil component is writing past its own region.

5. **Read-back cost** — 16 KB × num_cores per dispatch is small for
   single-core tests but qwen3 has 13 cores. At 5 us per PCIe read
   that's 65 μs per dispatch overhead. Mitigation: only enable
   profiler on the dispatches you care about (per-frame
   `enable_profiler(...)` API), or read N dispatches at a time.

6. **CLAUDE.md scope** — The doc currently says "Not in scope:
   DRAM interleaved, fast dispatch, mesh, Wormhole/Quasar."
   Device profiling isn't called out either way. Treat this plan
   as a scope extension that should be explicitly mentioned in
   CLAUDE.md after Phase 7 lands.

## 7. Verification plan

- **Phase 1**: existing `ctest -L hw` all pass with `TT_FOIL_DEVICE_PROFILER=ON`
- **Phase 3**: `test_add_kernel` with a synthetic 1-zone kernel —
  verify the parsed start/end are within a microsecond of the
  host-measured wait_done.
- **Phase 5**: cifar10 conv_3x3_l1 with math zone — verify
  `sum(device math zones) ≤ wait_done` and the ratio is plausible
  (>50% would mean math-bound, <50% would suggest NOC-bound).
- **Phase 7**: full pipeline run on cifar10 and qwen3, side-by-side
  with the existing host-only profiler — the answers should refine,
  not contradict, what we see today.

## 8. Out of scope (deliberately)

- Fast-dispatch realtime profiler (`cq_realtime_profiler.cpp` flow).
  tt-foil is slow-dispatch only per CLAUDE.md; the simpler
  per-dispatch read-back is sufficient.
- Tracy GUI integration of device zones (TracyTTDevice). We'll feed
  the data into our own CSV pipeline only; Tracy GUI device
  timeline is a future polish item.
- Automatic per-instruction zones (e.g. instrumenting every
  `matmul_tiles`). Manual `DeviceZoneScopedN` placement keeps the
  signal-to-noise ratio high; we don't want every `ZoneScopedN` to
  hide a 100-cycle overhead.
- Wormhole / Quasar support. The plan is Blackhole-only; HAL constants
  already differ.

## 9. Decision points before starting

Before Phase 1 starts I'd like confirmation on:

1. Is the 16 KB / core L1 cost acceptable? Cifar10 currently uses
   ~64 KB peak L1, so adding 16 KB pushes us to 80 KB out of 1.4 MB
   — totally fine for current workloads, possibly tight if a future
   workload is L1-bound.
2. Is patching `brisc.cc` (vs. via wrapper) acceptable if needed?
3. Should `TF_DEVICE_ZONE` live in `include/tt_foil/` (public API,
   long-term commitment) or stay internal (free to rename)?

Items 2 and 3 we can defer to Phase 1 (we'll know more after the
firmware-build experiment). Item 1 is a yes-or-no upfront.
