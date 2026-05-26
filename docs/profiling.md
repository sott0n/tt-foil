# Profiling tt-foil with Tracy

This guide explains how to capture **Performance Reports**, **Memory
Reports** and **Device-side Performance Reports** from a tt-foil binary
using the optional Tracy integration plus an in-process L1 readback of
tt-metal's `kernel_profiler.hpp` zones. The design rationale and the
host-only bookkeeping model are documented in
[`memory_management.md`](memory_management.md); this file is the practical
how-to.

## TL;DR

```bash
# 1. Build tt-foil with the profiler features you want (one-time)
#    TT_FOIL_ENABLE_TRACY    → host-side TF_* zones + Tracy memory events
#    TT_FOIL_DEVICE_PROFILER → kernel_profiler.hpp on every RISC + L1 readback
cmake -B build -DTT_FOIL_ENABLE_TRACY=ON -DTT_FOIL_DEVICE_PROFILER=ON \
               -DTT_FOIL_HW_TESTS=ON
cmake --build build -j$(nproc)

# 2. Run any tt-foil binary under the profiler wrapper
python tools/tt_foil_profile.py ./build/tests/test_add_kernel

# 3. Inspect the reports
cat generated/profiler/reports/tt_foil_perf_results.csv     # host zones
cat generated/profiler/reports/tt_foil_memory_report.csv    # buffers
cat generated/profiler/reports/tt_foil_device_perf.csv      # device zones
```

The two CMake options are **independent**: turn on only the side you need.
Without `TT_FOIL_ENABLE_TRACY`, host-side zones and memory events are
zero-cost. Without `TT_FOIL_DEVICE_PROFILER`, no kernel-side cycle markers
are recorded and the device report stays empty.

## 1. Prerequisites

- A tt-metal build with `ENABLE_TRACY=ON` is available at
  `$TT_METAL_BUILD_DIR` (default: `third_party/tt-metal/build_Release/`).
  tt-foil links the `libtracy.so` produced there and reuses
  `capture-release` and `csvexport-release` from
  `${TT_METAL_BUILD_DIR}/tools/profiler/bin/`. No separate Tracy install is
  required.
- Python 3.10+ for the wrapper script.

`tt_foil_profile.py` exits with a clear error if either binary is missing.

## 2. Building tt-foil with profiling enabled

Two independent CMake options control the two profilers. Both are **off
by default** — production builds pay zero overhead. Enable each only when
you want its specific output.

### 2.1 `TT_FOIL_ENABLE_TRACY` — host-side zones + memory

```bash
cmake -B build -DTT_FOIL_ENABLE_TRACY=ON
cmake --build build -j$(nproc)
```

When enabled:

- `libtracy.so` is linked into binaries that use `libtt_foil.a`.
- The `TRACY_ENABLE` macro activates the `TF_ZONE_N`, `TF_FRAME_MARK`,
  `TF_ALLOC` / `TF_FREE` and `TF_POOL_RESET` macros in
  [`src/profiling.hpp`](../src/profiling.hpp).
- The compile flag `-fno-omit-frame-pointer` is added so Tracy can resolve
  call stacks.

When disabled, the same macros expand to `do {} while (0)` and no Tracy
symbols enter the binary. Verify with:

```bash
ldd build/tests/test_add_kernel | grep tracy   # empty when Tracy is off
```

### 2.2 `TT_FOIL_DEVICE_PROFILER` — on-chip cycle markers

```bash
cmake -B build -DTT_FOIL_DEVICE_PROFILER=ON \
               -DTT_FOIL_PROFILE_KERNEL_BITMASK=1
cmake --build build -j$(nproc)
```

When enabled:

- `scripts/build_firmware.sh` recompiles BRISC/NCRISC/TRISC firmware with
  `-DPROFILE_KERNEL=<bitmask>` plus `-flto=auto -ffat-lto-objects`
  (required to keep BRISC under its 0x2200-byte region with profiler
  code linked in).
- The `tt_foil_hw_test()` CMake helper propagates `TT_FOIL_PROFILE_KERNEL`
  to every `examples/*/build_kernels.sh` and `models/*/build_kernels.sh`
  invocation, so kernel ELFs are rebuilt against the matching
  `*_weakened.elf` — a mismatch silently hangs `cb_reserve_back` (see
  `CLAUDE.md`'s firmware-vs-kernel invariant).
- `src/device_profile.cpp` reads each kernel's per-RISC L1 profiler
  region after every dispatch and writes raw zone events to
  `tt_foil_device_zones.csv` (configurable via
  `TT_FOIL_DEVICE_ZONES_CSV`).

The kernel auto-rebuild infrastructure (`scripts/kernel_build_helpers.sh`
+ `.build_stamp` per `prebuilt/`) recompiles kernels exactly when
needed when you toggle this option — there is no manual `rm -rf
prebuilt/` step. See [`docs/memory_management.md`](memory_management.md)
for the host-side memory model; the device profiler section below
covers what's emitted.

## 3. Running the profiler wrapper

Invoke `tools/tt_foil_profile.py` with the binary and its args appended:

```bash
python tools/tt_foil_profile.py <profiler-flags> <target_binary> [target args...]
```

Examples:

```bash
# 1. A unit-style HW test
python tools/tt_foil_profile.py ./build/tests/test_tile_copy

# 2. Qwen3 inference (use -- to separate wrapper args from the binary's args)
TT_FOIL_QWEN3_DATA=$PWD/data/qwen3_vl_2b \
TT_FOIL_OPS_DIR=$PWD/ops \
python tools/tt_foil_profile.py -o generated/qwen3_profile -- \
    ./build/models/qwen3_vl_2b/qwen3_run /tmp/prompt_ids.bin 4

# 3. Different output directory + non-default zone prefix
python tools/tt_foil_profile.py \
    -o /tmp/myrun --zone-prefix TF_ \
    ./build/tests/test_multi_kernel
```

What it does internally:

1. Launches `capture-release -o <out>/.logs/tracy_profile_log.tracy -f -p 8086`
   in the background.
2. Runs the target binary with `TT_FOIL_MEM_LOG` pointing at
   `<out>/.logs/tt_foil_memlog.csv`, so the in-process buffer-event
   logger is enabled.
3. Sends `SIGINT` to `capture-release` after the target exits — this
   triggers its "Save & Quit" path which flushes the `.tracy` file.
4. Runs `csvexport-release -u -f TF_ <trace>` to extract zone events.
5. Post-processes both CSVs into the final report files.

### Wrapper flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-o, --output-folder` | `generated/profiler` | Where reports are written |
| `--zone-filter` | empty (all zones) | Substring filter passed to `csvexport-release -f` |
| `--capture-port` | `8086` | Tracy capture-release listening port |
| `--kernel-source-dirs` | `examples,models,src` | Comma-separated dirs (relative to repo root) scanned for `DeviceZoneScopedN("...")` to resolve hashes to names in the device perf report |
| `-v, --verbose` | off | Show capture-release's own log output |

If you forget `--`, argparse will swallow the first positional flag-looking
arg of the target. Use `--` to be safe with multi-arg binaries.

## 4. Output files

```
generated/profiler/
├── .logs/
│   ├── tracy_profile_log.tracy       # raw Tracy capture (open in Tracy GUI)
│   ├── tracy_ops_times.csv           # csvexport-release output (per-event host zones)
│   ├── tt_foil_memlog.csv            # raw buffer alloc/free events
│   ├── tt_foil_device_zones.csv      # raw device-side zone events (one row per START/END)
│   └── tt_foil_device_clock_sync.csv # one-line (host_ns_first, cycle_first) anchor
└── reports/
    ├── tt_foil_perf_results.csv      # host Performance Report (per-zone × context)
    ├── tt_foil_perf_per_call.csv     # per-call timeline (seq, ts, zone, dur, ctx)
    ├── tt_foil_phases.csv            # auto-detected phases (TF_kernel_load bursts)
    ├── tt_foil_memory_report.csv     # Memory Report (per-pool aggregate)
    └── tt_foil_device_perf.csv       # device Performance Report (per RISC × zone)
```

The `.tracy`, `.logs/tt_foil_device_zones.csv` and
`.logs/tt_foil_device_clock_sync.csv` files are intermediate artifacts —
they're useful for ad-hoc digging but the `reports/` CSVs are the curated
outputs. The Tracy GUI can also open the `.tracy` file directly for a
visual timeline (see §6).

### Performance Report

Per-zone aggregate produced from `tracy_ops_times.csv`:

```csv
ZONE_NAME,CALL_COUNT,TOTAL_NS,MEAN_NS,MIN_NS,MAX_NS
TF_device_open,1,27324728,27324728,27324728,27324728
TF_dispatch_execute_multi,1564,1084649732,693510,60719,13965493
TF_dispatch/wait_done,3279,895894472,273221,2090,13495312
TF_dispatch/setup,3279,36193483,11037,2089,114287
TF_dispatch/fire_go,3279,2567459,783,270,25540
TF_kernel_load,2356,771556041,327485,70139,567490
```

Zones currently emitted (see [`src/profiling.hpp`](../src/profiling.hpp) and
the call sites for the source of truth):

| Zone | Location | Covers |
|------|----------|--------|
| `TF_device_open` / `TF_device_close` | `device.cpp` | Boot + teardown of `umd::Cluster` and per-core firmware |
| `TF_device/core_boot` | `device.cpp` | Per-core firmware load + reset deassert + INIT wait |
| `TF_firmware_load` | `firmware_load.cpp` | One ELF (BRISC / NCRISC / TRISC0-2) load via `ll_api::memory` |
| `TF_kernel_load` | `kernel.cpp` | User kernel ELF + RTA + CB blob staging |
| `TF_dispatch_execute` / `TF_dispatch_execute_multi` | `dispatch.cpp` | Outer scope of one dispatch call |
| `TF_dispatch/send_reset` | `dispatch.cpp` | Stage 0: GO_MSG reset + GO_MSG_INDEX zero |
| `TF_dispatch/setup` | `dispatch.cpp` | Stage 1: ELF + RTA + launch_msg L1 writes |
| `TF_dispatch/fire_go` | `dispatch.cpp` | Stage 2: `GO_MSG = RUN_MSG_GO` write |
| `TF_dispatch/wait_done` | `dispatch.cpp` | Stage 3: poll until `GO_MSG.signal == RUN_MSG_DONE` |
| `TF_dispatch_launch_async` | `dispatch.cpp` | Fire-and-forget launch (R5 G1) |

### Memory Report

Per-pool aggregate produced from `tt_foil_memlog.csv`:

```csv
POOL,ALLOC_COUNT,FREE_COUNT,TOTAL_BYTES_ALLOCATED,PEAK_LIVE_BYTES,LEAKED_BYTES
Device DRAM,346,346,4147138560,4147081216,0
Device L1,7308,7308,1124784640,5038592,0
```

What each column means and what the bump-allocator design does and does
not let it capture is covered in
[`memory_management.md §3`](memory_management.md#3-consequence-what-the-memory-report-can-show).

### Device Performance Report

Per `(RISC, zone)` aggregate produced from `tt_foil_device_zones.csv`,
sorted by total cycles descending so the hottest zones are at the top.
Cycles are 1 GHz on Blackhole, so cycle ≈ ns.

```csv
RISC,ZONE,ZONE_HASH,CALL_COUNT,TOTAL_CYCLES,MEAN_CYCLES,MIN_CYCLES,MAX_CYCLES
BRISC,BRISC-FW,0xc29f,60,3761526,62692,2161,324518
NCRISC,NCRISC-FW,0xf57b,60,3750090,62501,1971,324325
TRISC2,TRISC-FW,0x7d7e,60,3716946,61949,1417,323770
NCRISC,NCRISC-KERNEL,0x1607,17,3079667,181156,81206,323862
TRISC0,TRISC-KERNEL,0xb77c,17,3070089,180593,80668,323286
BRISC,BRISC-KERNEL,0xbdb7,17,3068225,180483,80540,323189
```

Columns:

| Column | Meaning |
|--------|---------|
| `RISC` | One of BRISC / NCRISC / TRISC0 / TRISC1 / TRISC2 |
| `ZONE` | Resolved zone name (blank when the hash isn't in any scanned source) |
| `ZONE_HASH` | The 16-bit FNV-1a hash that `kernel_profiler.hpp` emitted at the START/END markers — useful for grepping the kernel source |
| `CALL_COUNT` | Number of START/END pairs observed across all cores |
| `TOTAL_CYCLES` | Sum of cycle deltas (END − START) for that zone |
| `MEAN_CYCLES` | Integer mean = `TOTAL_CYCLES / CALL_COUNT` |
| `MIN_CYCLES` / `MAX_CYCLES` | Bounds across all observed pairs |

Two zone families dominate a typical capture:

- **`<RISC>-FW`** — fired once per dispatch on every RISC, around the
  whole kernel lifecycle (firmware entry → user kernel → exit). 60
  calls in the cifar10 example because 60 dispatches each ran on 5
  cores × 5 RISCs (with the same hash deduplicated by `(risc, zone)`).
- **`<RISC>-KERNEL`** — fired only when that RISC actually has a user
  kernel loaded for the dispatch. So fewer calls than the FW zone:
  the cifar10 example shows 17 calls for NCRISC-KERNEL but 60 for
  NCRISC-FW because many dispatches in that workload are BRISC-only
  (e.g., `add_two_numbers`-style passes).

The remaining rows surface your own `DeviceZoneScopedN("name")` calls.
Unresolved hashes (`ZONE` blank) usually mean the source line where the
zone is defined isn't under any directory passed to
`--kernel-source-dirs` — extend that list or read the hash off the
kernel `.cpp` directly.

### Adding cycle markers to your kernels

`tt-metal/tt_metal/tools/profiler/kernel_profiler.hpp` is included
implicitly when `TT_FOIL_DEVICE_PROFILER=ON`. To time a code path inside
one of your kernels, wrap it with `DeviceZoneScopedN`:

```cpp
// examples/add_two_numbers/kernels/add_brisc.cpp
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("add_two_numbers");
    // ... actual work ...
}
```

Notes:

- The hash is `kernel_profiler::Hash16_CT(name "," __FILE__ "," __LINE__
  ",KERNEL_PROFILER")` — moving the call site to another line changes
  the hash. `tt_foil_profile.py` recomputes the same hash by scanning
  source files, so name resolution survives line edits as long as the
  file still contains the macro.
- **Compute kernels (TRISC0/1/2) are sensitive** to `DeviceZoneScopedN`
  RAII inside the compute body — matmul-style kernels we tried hung at
  `cb_reserve_back`. Reader/writer kernels (BRISC/NCRISC) are safe.
  Until that's narrowed down, restrict instrumentation in compute
  kernels to function-entry zones at most, or skip TRISC entirely.
- Each RISC's L1 profiler region holds 512 words. Zones beyond that
  get dropped silently; keep instrumentation focused on what you want
  to measure.

### Clock-sync caveat

`tt_foil_device_clock_sync.csv` records the first `(host_steady_ns,
device_cycle)` pair the runtime sees, so device cycles can be projected
onto the host timeline as `host_ns ≈ device_cycle + (host_ns_first −
cycle_first)` (1 ns ≈ 1 cycle on Blackhole). The pairing is sampled
post-dispatch, so the offset includes a few µs of PCIe round-trip — fine
for "which zone is the hottest" questions, off by enough that
strict sub-µs host/device alignment needs a dedicated calibration kernel.

## 5. Adding your own zones

### 5.1 Inside `libtt_foil` (runtime internals)

In any `.cpp` file under `src/` (do not include from public headers):

```cpp
#include "profiling.hpp"

void my_hot_path() {
    TF_ZONE_N("TF_my_hot_path");
    // ... work ...
}
```

For memory tracking, use the existing `buffer_alloc` / `buffer_free` path —
those already emit `TF_ALLOC` / `TF_FREE`. If you allocate device memory
outside that path (rare; today only `kernel_config_allocs` does), wire
`TF_ALLOC` / `TF_FREE` in directly.

Keep zone names prefixed with `TF_` so they're trivial to filter from
third-party Tracy-instrumented code (`tt_foil_profile.py --zone-filter
TF_`).

### 5.2 From user code (binaries that link `libtt_foil`)

Public macros in [`include/tt_foil/profiling.h`](../include/tt_foil/profiling.h)
let model / test binaries add their own zones without depending on the
private `src/profiling.hpp`:

```cpp
#include "tt_foil/profiling.h"

int main() {
    TT_FOIL_ZONE("Phase_A_stem_layer1");
    // ... dispatch calls ...
    TT_FOIL_ZONE_TEXT("conv_3x3_l1");   // optional per-zone context
    TT_FOIL_FRAME_MARK();               // optional frame boundary
}
```

These compile to no-ops when `TT_FOIL_ENABLE_TRACY=OFF`, so user code can
stay annotated unconditionally.

### 5.3 Inside device kernels

See *Adding cycle markers to your kernels* under §4 above.

## 6. Viewing the trace in Tracy GUI (optional)

`.tracy` capture files open directly in the Tracy profiler GUI (download
from <https://github.com/wolfpld/tracy/releases>). All zones plus
`TracyAllocN` / `TracyFreeN` events show up under the **Memory** tab; the
GUI is useful for visually exploring nested zone timing and per-thread
flow.

For live (rather than offline) capture, run the Tracy GUI in capture mode
**instead of** `tt_foil_profile.py`. tt-foil's Tracy client will connect
automatically when the first zone fires.

## 7. Common gotchas

- **"ERROR: Tracy capture file not produced"** — the target binary was
  built without `TT_FOIL_ENABLE_TRACY=ON`, so it has no Tracy client to
  connect to capture-release. Rebuild tt-foil with the option enabled.

- **Memory Report is empty** — the binary was run without
  `TT_FOIL_MEM_LOG` set. The wrapper script sets it automatically; if
  you ran the binary directly, the in-process memory logger stays off
  by design (no I/O overhead unless explicitly enabled).

- **Capture port collision** — if 8086 is already in use, pass
  `--capture-port` to `tt_foil_profile.py`.

- **HW tests need a chip reset between runs** if a previous run crashed.
  `~/tt-venv/bin/tt-smi -r <device>` clears the state. ctest already
  serialises HW tests via `RESOURCE_LOCK chip`.

- **Device perf report is empty** — the binary was built without
  `TT_FOIL_DEVICE_PROFILER=ON`, or `kernel_profiler.hpp` never armed
  for some reason. Confirm by checking that `.logs/tt_foil_device_zones.csv`
  exists and has rows.

- **Device zone names are blank (`0xABCD` only)** — the source file that
  emitted the zone isn't under any directory passed via
  `--kernel-source-dirs`. Pass `--kernel-source-dirs examples,models,
  src,my_dir` to widen the scan.

- **Compute kernel hang after enabling device profiler** — instrumenting
  TRISC0/1/2 kernels with `DeviceZoneScopedN` has been seen to wedge
  `cb_reserve_back` for matmul-style flows. Remove TRISC instrumentation
  and rely on the auto `TRISC-FW` / `TRISC-KERNEL` zones from
  firmware until that's narrowed.

## 8. Performance impact

Empirically measured on a qwen3 4-decode benchmark (N=3, `TT_FOIL_MEM_LOG`
on vs. off): dispatch zone medians differ by < 1%, within run-to-run
jitter (PCIe, chip thermal). The Performance Report and Memory Report
can be captured from a single run with no measurable trade-off — see
[`memory_management.md §4`](memory_management.md#4-performance-implication-combining-performance--memory-reports)
for the design reasoning.

## 9. References

- [`tools/tt_foil_profile.py`](../tools/tt_foil_profile.py) — wrapper source (host + device aggregation)
- [`src/profiling.hpp`](../src/profiling.hpp) / [`src/profiling.cpp`](../src/profiling.cpp) — Tracy macros + per-core pool names
- [`include/tt_foil/profiling.h`](../include/tt_foil/profiling.h) — public `TT_FOIL_ZONE` macros for user code
- [`src/device_profile.cpp`](../src/device_profile.cpp) — L1 profiler readback + raw zone CSV writer
- [`scripts/build_firmware.sh`](../scripts/build_firmware.sh) — profiler-aware firmware build (LTO, `PROFILE_KERNEL`)
- [`scripts/kernel_build_helpers.sh`](../scripts/kernel_build_helpers.sh) — kernel auto-rebuild stamp logic
- [`docs/memory_management.md`](memory_management.md) — host-side memory model
- tt-metal: `tools/tracy/__main__.py` — host-zone wrapper this is modelled on
- tt-metal: `tt_metal/tools/profiler/kernel_profiler.hpp` — device-zone macros + L1 marker format
