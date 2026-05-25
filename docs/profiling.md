# Profiling tt-foil with Tracy

This guide explains how to capture **Performance Reports** and **Memory
Reports** from a tt-foil binary using the optional Tracy integration. The
design rationale and the host-only bookkeeping model are documented in
[`memory_management.md`](memory_management.md); this file is the practical
how-to.

## TL;DR

```bash
# 1. Build tt-foil with Tracy enabled (one-time)
cmake -B build -DTT_FOIL_ENABLE_TRACY=ON -DTT_FOIL_HW_TESTS=ON
cmake --build build -j$(nproc)

# 2. Run any tt-foil binary under the profiler wrapper
python tools/tt_foil_profile.py ./build/tests/test_add_kernel

# 3. Inspect the reports
cat generated/profiler/reports/tt_foil_perf_results.csv
cat generated/profiler/reports/tt_foil_memory_report.csv
```

## 1. Prerequisites

- A tt-metal build with `ENABLE_TRACY=ON` is available at
  `$TT_METAL_BUILD_DIR` (default: `third_party/tt-metal/build_Release/`).
  tt-foil links the `libtracy.so` produced there and reuses
  `capture-release` and `csvexport-release` from
  `${TT_METAL_BUILD_DIR}/tools/profiler/bin/`. No separate Tracy install is
  required.
- Python 3.10+ for the wrapper script.

`tt_foil_profile.py` exits with a clear error if either binary is missing.

## 2. Building tt-foil with Tracy

The `TT_FOIL_ENABLE_TRACY` CMake option is **off by default** — production
builds pay zero overhead. Pass it explicitly to opt in:

```bash
cmake -B build -DTT_FOIL_ENABLE_TRACY=ON
cmake --build build -j$(nproc)
```

When enabled:

- `libtracy.so` is linked into `libtt_foil.a`'s downstream binaries.
- The `TRACY_ENABLE` macro activates the `TF_ZONE_N` and `TF_ALLOC` /
  `TF_FREE` macros in [`src/profiling.hpp`](../src/profiling.hpp).
- The compile flag `-fno-omit-frame-pointer` is added so Tracy can resolve
  call stacks.

When disabled, the same macros expand to `do {} while (0)` and no Tracy
symbols enter the binary. Verify with:

```bash
ldd build/tests/test_add_kernel | grep tracy   # empty when Tracy is off
```

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
| `--zone-prefix` | `TF_` | Filter zones by prefix when exporting |
| `--capture-port` | `8086` | Tracy capture-release listening port |
| `-v, --verbose` | off | Show capture-release's own log output |

If you forget `--`, argparse will swallow the first positional flag-looking
arg of the target. Use `--` to be safe with multi-arg binaries.

## 4. Output files

```
generated/profiler/
├── .logs/
│   ├── tracy_profile_log.tracy     # raw Tracy capture (open in Tracy GUI)
│   ├── tracy_ops_times.csv         # csvexport-release output (per-event zones)
│   └── tt_foil_memlog.csv          # raw buffer alloc/free events
└── reports/
    ├── tt_foil_perf_results.csv    # Performance Report (per-zone aggregate)
    └── tt_foil_memory_report.csv   # Memory Report (per-pool aggregate)
```

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

## 5. Adding your own zones

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

Keep zone names prefixed with `TF_` so the default `--zone-prefix TF_`
filter picks them up while ignoring any zones from third-party code that
happens to be Tracy-instrumented.

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

## 8. Performance impact

Empirically measured on a qwen3 4-decode benchmark (N=3, `TT_FOIL_MEM_LOG`
on vs. off): dispatch zone medians differ by < 1%, within run-to-run
jitter (PCIe, chip thermal). The Performance Report and Memory Report
can be captured from a single run with no measurable trade-off — see
[`memory_management.md §4`](memory_management.md#4-performance-implication-combining-performance--memory-reports)
for the design reasoning.

## 9. References

- [`tools/tt_foil_profile.py`](../tools/tt_foil_profile.py) — wrapper source
- [`src/profiling.hpp`](../src/profiling.hpp) — Tracy macros
- [`docs/memory_management.md`](memory_management.md) — design rationale
- tt-metal: `tools/tracy/__main__.py` — the inspiration for the wrapper
