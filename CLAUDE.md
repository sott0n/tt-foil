# tt-foil Codebase Guide (for Claude)

## What this is

A C++ runtime for running pre-compiled kernels on Tenstorrent **Blackhole**.
Standalone — does NOT link `libtt_metal.so`. Owns its own `umd::Cluster` and
`tt::tt_metal::Hal` instances; the HAL .cpp sources are compiled into a local
static lib from the tt-metal source tree.

**Scope:** single chip, multi Tensix core (v3), BRISC + NCRISC, slow-dispatch,
NOC unicast L1↔L1 between cores. **Not** in scope: TRISC compute, Circular
Buffers, DRAM interleaved, fast dispatch, mesh.

## Repository layout

```
include/tt_foil/runtime.hpp            # Public API
src/
  device.{hpp,cpp}                     # umd::Cluster + Hal ownership, multi-core cold boot
  buffer.{hpp,cpp}                     # Bump allocators, write_l1/read_l1, write_dram/read_dram
  kernel.{hpp,cpp}                     # ELF parsing (XIP), per-RISC RTA staging
  dispatch.{hpp,cpp}                   # setup → fire_go → wait_done stages; multi-kernel variant
  firmware_load.{hpp,cpp}              # boot firmware ELF → fw_base (relocate_dev_addr)
  firmware_paths.{hpp,cpp}             # pre-compiled/<hash>/ discovery + TT_FOIL_FIRMWARE_DIR override
  mailbox_init.{hpp,cpp}               # LAUNCH/GO_MSG(RUN_MSG_INIT)/rd_ptr/go_idx
  core_info_init.{hpp,cpp}             # minimal CORE_INFO (abs_logical_x/y + WORKER magic)
  bank_tables_init.{hpp,cpp}           # BANK_TO_NOC_SCRATCH + LOGICAL_TO_VIRTUAL_SCRATCH zero-fill
  reset.{hpp,cpp}                      # assert_tensix_reset / deassert_brisc_reset / wait_tensix_init_done
  noc_addr.{hpp,cpp}                   # NOC unicast 64-bit address packing
  umd_boot.{hpp,cpp}                   # umd::Cluster open helper (used by test_umd_open)
  runtime_impl.cpp                     # Thin delegation from public API
  llrt_local/                          # Vendored from tt_metal/llrt; ll_api → tt::foil::ll_api
    tt_memory.{cpp,h}                    (MetalContext::instance() call patched out; getenv)
    tt_elffile.{cpp,hpp}                 (unchanged except namespace)
    hal_stubs.cpp                        (throw stubs for initialize_wh / initialize_qa)
tests/                                 # see below
examples/
  add_two_numbers/                     # single core, BRISC + NCRISC, no NOC
  noc_passthrough/                     # 2 cores, BRISC producer/consumer, NOC unicast
cmake/tt_metal_deps.cmake              # tt_foil_llrt INTERFACE + tt_foil_hal_local STATIC
```

## Key invariants (read before editing)

### `libtt_metal.so` is NEVER linked
The shared lib `libtt_metal.so` from tt-metal is not in the link line. HAL is
built from `tt_metal/llrt/hal.cpp` + `hal/tt-1xx/hal_1xx_common.cpp` + the five
`hal/tt-1xx/blackhole/bh_hal*.cpp` files directly into `tt_foil_hal_local`.
Wormhole/Quasar HAL paths are stubbed (`src/llrt_local/hal_stubs.cpp`).
**Adding any call into `MetalContext` or `tt::Cluster` re-introduces the
runtime conflict that motivated B3** — see "the dual-UMD wall" below.

### `ll_api::memory` is the vendored copy
The version of `ll_api::memory` used by kernel.cpp / firmware_load.cpp / etc.
lives in `src/llrt_local/`, with namespace renamed to `tt::foil::ll_api` so
it doesn't fight libtt_metal.so's copy in the unlikely case a downstream
re-links it. **Do not switch to `<llrt/tt_memory.h>` — the upstream version
calls `MetalContext::instance()` from a debug-dump path** and that triggers a
fresh `umd::Cluster` open against every PCIe chip.

### Cores must be booted at `open_device` time
`open_device(pcie_idx, fw_dir, cores)` boots each entry in `cores` (default
`{{0,0}}`). `kernel_load` does **not** lazily boot — calling it on an unbooted
core gives garbage. `Device::booted_cores` is the source of truth.

### UMD `BarrierAddressParams` must be set
`umd::Cluster::l1_membar` is a no-op without `set_barrier_address_params`. We
call it in `device_open` right after constructing the cluster, with the
TENSIX/DRAM/ACTIVE_ETH `BARRIER` addresses from HAL. Forgetting this makes
`dispatch_execute` hang on the GO_MSG poll because the prior reg write never
flushes.

### BRISC is the only RISC the host deasserts
On Blackhole, BRISC firmware brings NCRISC + TRISC0/1/2 up via the
`subordinate_sync` mailbox. `deassert_brisc_reset` (defined for BRISC only) is
what `device_open` calls; do not also deassert NCRISC manually unless you
intend to bypass BRISC firmware.

### NOC address layout (Blackhole)
`NOC_XY_ADDR(x, y, addr) = (y<<42) | (x<<36) | addr` — `NOC_ADDR_NODE_ID_BITS = 6`,
`NOC_ADDR_LOCAL_BITS = 36`. `tt::foil::make_noc_unicast_addr` does this
packing on the host. Producer kernels receive the result split into hi/lo
32-bit halves via RTA and reassemble. **Do not call
`get_noc_addr_from_logical_xy` from a kernel** — the worker_logical_to_virtual
scratch is zero-filled, so it would resolve to (0,0).

## The dual-UMD wall (B3 lesson, do not repeat)

`MetalContext::instance()` is lazy-initialised on first call. When it runs it
constructs a fresh `umd::Cluster` with **no `target_devices`** restriction,
opening every PCIe chip. If tt-foil has already opened the same chip via its
own `umd::Cluster{target_devices={N}}`, the second open re-inits PCIe TLBs
and breaks BRISC firmware mid-flight — subsequent `dispatch_execute` hangs.

The bundled `tt_memory.cpp` was the last code path that called
`MetalContext::instance()`. It's now patched to `std::getenv("TT_FOIL_XIP_DUMP")`.
**If you add another translation unit that pulls in tt-metal `*.cpp` files,
audit it for `MetalContext::` / `rtoptions()` references**.

## Tests

All in `tests/`. CMake gates HW tests behind `-DTT_FOIL_HW_TESTS=ON`.

| Test                          | HW? | What it covers                                  |
| ----------------------------- | --- | ----------------------------------------------- |
| `test_bump_alloc`             | No  | Bump allocator semantics                        |
| `test_elf_load`               | No  | `ll_api::memory` round-trip (set TT_FOIL_FW_DIR)|
| `test_firmware_paths`         | No  | `pre-compiled/<hash>` discovery + env override  |
| `test_hal_standalone`         | No  | `new Hal(BLACKHOLE, ...)` succeeds standalone   |
| `test_add_kernel`             | Yes | Single BRISC kernel — add two L1 words          |
| `test_brisc_ncrisc`           | Yes | BRISC + NCRISC concurrent on one core           |
| `test_umd_open`               | Yes | umd::Cluster direct open + L1 round trip        |
| `test_multi_core_boot`        | Yes | open_device boots 2 cores to RUN_MSG_DONE       |
| `test_multi_kernel`           | Yes | Two BRISC kernels on different cores            |
| `test_noc_passthrough`        | Yes | Producer/consumer via noc_async_write           |

Run HW tests on Blackhole with:

```bash
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal \
TT_FOIL_FIRMWARE_DIR=/path/to/tt-metal/tt_metal/pre-compiled/<hash> \
TT_FOIL_DEVICE=3 \
TT_FOIL_KERNEL_DIR=examples/.../prebuilt \
./build/tests/test_<name>
```

`TT_FOIL_FIRMWARE_DIR` matters when more than one `pre-compiled/<hash>/` exists
— the auto-discovery picks the most recently modified one and that may not be
the one tt-metal's own `BuildEnvManager` would pick.

## Dispatch protocol summary

```
execute(dev, kernel):  → dispatch_execute_multi(dev, &k, 1)
execute(dev, {k1,k2}): → dispatch_execute_multi(dev, [k1,k2], 2)

dispatch_execute_multi:
  setup    for each kernel: write ELF + RTA + launch_msg
  sfence
  fire_go  for each kernel: write GO_MSG.signal = RUN_MSG_GO
  l1_membar
  wait     for each kernel: poll GO_MSG.signal == RUN_MSG_DONE
```

All GOs are fired before any DONE check — producer/consumer kernels need
that. Polling is sequential per kernel but device-side execution is concurrent,
so wall time is `max(per-kernel run time)`.

## Cold boot summary (single core)

```
1.  assert all 5 RISCs reset
2.  load brisc.elf, ncrisc.elf, trisc0.elf, trisc1.elf, trisc2.elf via
    ll_api::memory(DISCRETE) + process_spans + hal.relocate_dev_addr(addr,
    jit_cfg.local_init_addr, false)
3.  zero-fill BANK_TO_NOC_SCRATCH (2048 B) and LOGICAL_TO_VIRTUAL_SCRATCH (32 B)
4.  write CORE_INFO with abs_logical_x/y + magic = WORKER (other fields zero)
5.  write 8 zero launch_msg_t copies, GO_MSG.signal = RUN_MSG_INIT,
    LAUNCH_MSG_BUFFER_RD_PTR = 0, GO_MSG_INDEX = 0
6.  deassert BRISC reset
7.  poll GO_MSG.signal until RUN_MSG_DONE (10 s timeout)
```

For multi-core boot, `device_open` runs steps 1-6 for every core, then runs
step 7 for every core (so cores boot in parallel on the device side).

## Common surprises / FAQ

**"GO_MSG poll times out"** — usually `BarrierAddressParams` wasn't set on
the umd::Cluster, or the chip is in a stale state from a previous run that
crashed. Re-running once usually fixes the latter; check
`set_barrier_address_params` for the former.

**"My kernel ELF byte-mismatches what tt-metal writes"** — pick
`<risc>/<risc>.elf`, not `<risc>/<risc>_weakened.elf`. The weakened ELF is
the input to user-kernel linking, not the firmware itself, and has unresolved
relocations.

**"NOC write reaches address (0,0)"** — you're probably calling
`get_noc_addr_from_logical_xy` from a kernel. The logical→virtual scratch is
zero-filled. Use the host-side `make_noc_unicast_addr` and pass the 64-bit
result via RTA instead.

**"Why is `pre-compiled/<hash>/` numerical?"** — `BuildEnvManager::compute_build_key()`
hashes dispatch_core_type + num_hw_cqs + harvesting_mask + the SFPI compile-
hash string. Recomputing it outside tt-metal would couple us to the internal
hasher; we scan the dir instead and let `TT_FOIL_FIRMWARE_DIR` override.

## Important constants (Blackhole)

From `tt_metal/hw/inc/internal/tt-1xx/blackhole/`:

| Constant                  | Value          | Where                            |
| ------------------------- | -------------- | -------------------------------- |
| `NOC_ADDR_NODE_ID_BITS`   | 6              | `noc/noc_parameters.h`           |
| `NOC_ADDR_LOCAL_BITS`     | 36             | `noc/noc_parameters.h`           |
| `MaxProcessorsPerCoreType`| 5              | `core_config.h`                  |
| `launch_msg_buffer_num_entries` | 8       | `hal/generated/dev_msgs.hpp`     |

HAL addresses pulled at runtime (Blackhole Tensix):

| Mailbox / Region            | Offset    | Notes                       |
| --------------------------- | --------- | --------------------------- |
| `LAUNCH`                    | 0x70      | 8 × launch_msg_t            |
| `LAUNCH_MSG_BUFFER_RD_PTR`  | 0x6C      | uint32                      |
| `CORE_INFO`                 | 0x948     | 248 B                       |
| `GO_MSG`                    | 0x3F0     | 4 B (signal byte at offset 0)|
| `GO_MSG_INDEX`              | 0x420     | uint32                      |
| `KERNEL_CONFIG`             | 0x9E00    | RTA + kernel text region    |
| `BANK_TO_NOC_SCRATCH`       | 0x12E00   | 2048 B (zero-fill on boot)  |
| `LOGICAL_TO_VIRTUAL_SCRATCH`| 0x13600   | 32 B (zero-fill on boot)    |

## When making changes

- **Add a new runtime API function**: declare in `include/tt_foil/runtime.hpp`,
  implement in the relevant `src/*.cpp`, thin-wrap in `src/runtime_impl.cpp`.
- **Add a new boot step**: put the helper in its own `src/<name>_init.{hpp,cpp}`
  pair matching the existing per-step files; call from `device_open` between
  the firmware load and the BRISC deassert.
- **Touch dispatch**: the three stages (`dispatch_stage_setup`,
  `dispatch_stage_fire_go`, `dispatch_stage_wait_done`) are in an anonymous
  namespace in `dispatch.cpp`; `dispatch_execute_multi` is the only caller
  worth modifying, single-kernel forwards.
- **Add a new example**: copy `examples/noc_passthrough/build_kernels.sh`
  as the template; it's the up-to-date one that builds against the local
  SFPI toolchain + firmware-weakened ELFs.

## Commit hygiene

- No "Phase X" / "Step Y" / "Session N" language in commit messages or code.
- Squash incremental commits into 3-5 logical units per phase.
- Co-Authored-By: Claude <noreply@anthropic.com> trailer on AI-assisted
  commits (this is recorded in the user's MEMORY.md too).
