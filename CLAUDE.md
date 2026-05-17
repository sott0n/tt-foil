# tt-foil Codebase Guide (for Claude)

## What this is

A C++ runtime for running pre-compiled kernels on Tenstorrent **Blackhole**.
Standalone — does NOT link `libtt_metal.so`. Owns its own `umd::Cluster` and
`tt::tt_metal::Hal` instances; the HAL .cpp sources are compiled into a local
static lib from the tt-metal source tree.

**Scope:** single chip, multi Tensix core, BRISC + NCRISC + TRISC0/1/2 (full
5-RISC), slow-dispatch, NOC unicast L1↔L1 between cores, Circular Buffers.
**Not** in scope: DRAM interleaved, fast dispatch, mesh, Wormhole/Quasar.

## Repository layout

```
include/tt_foil/runtime.hpp            # Public API
src/
  device.{hpp,cpp}                     # umd::Cluster + Hal ownership, multi-core cold boot
  buffer.{hpp,cpp}                     # Bump allocators, write_l1/read_l1, write_dram/read_dram
  kernel.{hpp,cpp}                     # ELF parsing (XIP), per-RISC RTA staging
  dispatch.{hpp,cpp}                   # reset → setup → fire_go → wait_done; multi-kernel variant
  cb_config.{hpp,cpp}                  # CB descriptor blob layout + L1 write (v4)
  firmware_load.{hpp,cpp}              # boot firmware ELF → fw_base (relocate_dev_addr)
  firmware_paths.{hpp,cpp}             # ~/.cache > tt_metal/pre-compiled discovery + override
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
  tile_copy/                           # 1 core, 5 RISCs: reader→CB c_0→TRISC copy_tile→CB c_16→writer (v4)
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

### Firmware ELFs: tt-foil self-build (Plan L)
`firmware_paths.cpp` resolves in this order:
1. `$TT_FOIL_FIRMWARE_DIR` (explicit override)
2. `$TT_FOIL_BUILD_FIRMWARE_DIR`, or the path baked in via
   `TT_FOIL_FIRMWARE_BUILD_DIR` at CMake configure time
   (`<build>/firmware/` when `TT_FOIL_BUILD_FIRMWARE=ON`, default)
3. `$HOME/.cache/tt-metal-cache/<hash>/firmware/` (newest, when present)
4. `$TT_METAL_RUNTIME_ROOT/tt_metal/pre-compiled/<hash>/` (fallback)

(2) is the default source. `scripts/build_firmware.sh` (invoked by the
`tt_foil_firmware` CMake target) compiles `brisc.cc`, `ncrisc.cc`, and
`trisc.cc` (3× for UNPACK/MATH/PACK) via SFPI g++, then runs
`tools/tt_foil_weaken` to weaken all data symbols except
`__fw_export_*` and `__global_pointer$` — mirroring tt-metal's
`JitBuildState::weaken()`. The result is a `<build>/firmware/` tree with
the same `<risc>/<risc>.elf` + `<risc>/<risc>_weakened.elf` layout as
tt-metal's JIT cache, so the existing kernel build flow drops in
unchanged.

Lesson from v4: firmware ELFs and the `*_weakened.elf` the kernel was
linked against must come from the same build. Mismatched bytes silently
break `setup_local_cb_read_write_interfaces` — the kernel runs
end-to-end but every CB interface reads as zero, so `cb_reserve_back`
hangs forever. The `build_kernels.sh` scripts under `examples/*/` use
the same priority order, so by default they link against tt-foil's
self-built `build/firmware/`.

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

### BRISC writes to peer L1 must use NOC=1 (v6-3 lesson)
BRISC's default `noc_index = 0` is the **read** direction on Blackhole.
`noc_async_write` / `noc_async_write_one_packet` issued from BRISC on
NOC 0 to a peer Tensix L1 endpoint stalls in `noc_async_write_barrier`
forever — the ACK never returns. (4-byte writes occasionally complete
but flakily; anything bigger always hangs.) Pass `noc=1` (NCRISC's
default writer NOC) explicitly:
```cpp
constexpr uint8_t kWriteNoc = 1;
noc_async_write_one_packet(src, dst_noc_addr, size, kWriteNoc);
noc_async_write_barrier(kWriteNoc);
```
Reads (`noc_async_read`) stay on the default NOC 0 from BRISC. NCRISC,
by contrast, defaults to NOC 1 for both directions and the
matmul_dram-style writer kernels work without passing an explicit
`noc=...`. See `examples/matmul_2core_mcast/kernels/reader_producer.cpp`
for the working pattern.

### Tile-stream offset math: `(kt * Nt + nt)`, not `kt` (v6-3 lesson)
For DRAM-resident B (laid out as Kt rows × Nt columns of tiles,
kt-major), the (kt, nt) tile is at offset `(kt * Nt + nt) * kTileBytes`.
Reading at `kt * kTileBytes` only is a **silent bug** when Nt > 1: half
the matmul iterations grab the right B tile, half grab a wrong one, and
the resulting outputs look "systematically scaled" rather than random
— which makes the error look like a NOC/cache issue and burns hours.
Cross-check any new reader against `examples/matmul_dram/kernels/
reader.cpp`'s `b_dram_base + (kt * Nt + nt) * kTileBytes` form before
running the test.

### NOC checkpoint pattern for debugging
When a producer/consumer kernel hangs and you can't tell where, give
the kernel a small (16 × uint32_t) L1 scratch via RTA and have it write
fixed-sentinel values (`0x11111111`, `0x22222222`, …) at each phase
boundary. After the timeout, `read_buffer` the scratch from the host
and look at the last non-zero entry. That's how v6-3's "consumer
staging bytes match B but matmul output is wrong" insight was found
quickly — confirms NOC fan-out works and the bug is downstream.

### CB blob covers [0..max_idx], not [min_idx..max_idx]
Firmware's `setup_local_cb_read_write_interfaces` walks descriptors at
`cb_l1_base + cb_id * 16` and indexes them by absolute CB id. The blob
must lay out a 16-byte descriptor for every CB id from 0 up to the
highest one set in `local_cb_mask`; unused slots are zeros. There is no
`min_local_cb_start_index` field in `launch_msg` (only the remote-CB
path has one). `src/cb_config.cpp` does the right thing.

### TRISC compute build needs three non-obvious flags
For `compute.*.elf` (TRISC0/1/2 sharing one source compiled 3× with
`-DTRISC_UNPACK`/`MATH`/`PACK`):
- `-mcpu=tt-bh-tensix` (not `tt-bh`) — enables Tensix-extension intrinsics so
  `__builtin_rvtt_*` lowers inline.
- `-O3` (not `Os`) — required for `INSTRUCTION_WORD("n")` `asm` constraints
  to be resolvable at compile time.
- `-ffast-math -ftt-nttp -ftt-constinit -ftt-consteval` — tt-metal's
  standard compute flags.

Plus the stub `chlkc_list.h` we ship in `examples/tile_copy/build_kernels.sh`
substitutes for tt-metal's JIT-generated chlkc files (DST_ACCUM_MODE,
APPROX, MATH_FIDELITY, and per-CB pack/unpack format + tile-shape
arrays). For tile_copy this is bf16-fixed; a more general flow would
plug in real per-program values.

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
| `test_firmware_paths`         | No  | Firmware ELF discovery + env override           |
| `test_hal_standalone`         | No  | `new Hal(BLACKHOLE, ...)` succeeds standalone   |
| `test_risc_ids`               | No  | RiscId → HAL (class, type, idx) mapping (v4)    |
| `test_add_kernel`             | Yes | Single BRISC kernel — add two L1 words          |
| `test_brisc_ncrisc`           | Yes | BRISC + NCRISC concurrent on one core           |
| `test_umd_open`               | Yes | umd::Cluster direct open + L1 round trip        |
| `test_multi_core_boot`        | Yes | open_device boots 2 cores to RUN_MSG_DONE       |
| `test_multi_kernel`           | Yes | Two BRISC kernels on different cores            |
| `test_noc_passthrough`        | Yes | Producer/consumer via noc_async_write           |
| `test_cb_config`              | Yes | CB descriptor blob round-trip via L1 (v4)       |
| `test_tile_copy`              | Yes | End-to-end 5-RISC tile_copy pipeline (v4)       |

Run tests via ctest (preferred — `TT_FOIL_KERNEL_DIR` is wired per
test by the CMake test-helper functions, no manual env juggling):

```bash
cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_DEVICE=3
cmake --build build -j$(nproc)
tt-smi -r 3                        # one-shot, ensures clean chip state
ctest --test-dir build             # all tests, serialised on chip
ctest --test-dir build -L unit     # only host-side unit tests
ctest --test-dir build -L hw       # only Blackhole integration tests
ctest --test-dir build -R matmul   # by name regex
```

All HW tests share `RESOURCE_LOCK chip` so they never run in parallel
within one ctest invocation. If a kernel crashes the chip, `ctest`
cannot recover it — rerun with another `tt-smi -r N` first.

To run a single test binary directly:

```bash
TT_FOIL_DEVICE=3 \
TT_FOIL_KERNEL_DIR=examples/.../prebuilt \
./build/tests/test_<name>
```

`TT_FOIL_FIRMWARE_DIR` is optional — auto-discovery picks tt-foil's own
`<build>/firmware/` first (built by the `tt_foil_firmware` CMake target).
Set it explicitly only when you need to pin against a different build or
have moved the build tree.

## Dispatch protocol summary

```
execute(dev, kernel):  → dispatch_execute_multi(dev, &k, 1)
execute(dev, {k1,k2}): → dispatch_execute_multi(dev, [k1,k2], 2)

dispatch_execute_multi:
  reset    for each kernel: write RUN_MSG_RESET_READ_PTR_FROM_HOST to GO_MSG,
                            then zero GO_MSG_INDEX (matches tt-metal
                            send_reset_go_signal in llrt.cpp:115)
  l1_membar
  setup    for each kernel: write ELF + RTA + launch_msg
                            (launch_msg.kernel_config gets local_cb_offset /
                             local_cb_mask from kernel.cb_alloc when valid)
  sfence
  l1_membar
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

**"`cb_reserve_back` hangs in TRISC-compute kernels"** — kernel ELF was
linked against a `*_weakened.elf` that doesn't match the firmware
actually loaded onto the chip (different build hashes →
`setup_local_cb_*` writes silently miss `cb_interface[]`). Rebuild
kernels with `examples/*/build_kernels.sh` and make sure they pick the
same firmware tree tt-foil's `firmware_paths.cpp` will resolve at
runtime — by default both are `<tt-foil-build>/firmware/`.

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

**"BRISC NOC write to peer L1 hangs in `noc_async_write_barrier`"** —
on Blackhole, BRISC's default NOC 0 doesn't reliably handle writes to
peer Tensix L1 (ACK never returns). Pass `noc=1` explicitly to
`noc_async_write_one_packet` / `noc_async_write` and to the matching
`noc_async_write_barrier`. Reads stay on NOC 0. See the "BRISC writes
to peer L1 must use NOC=1" invariant above.

**"Multi-core matmul output is systematically wrong by a constant
factor"** — your reader's B-tile offset probably has `kt` instead of
`(kt * Nt + nt)`. With Nt > 1 the reader grabs the wrong B tile for
some (mt, nt) iterations, and the output looks scaled rather than
random. Diff against `examples/matmul_dram/kernels/reader.cpp`.

**"TRISC compute kernel won't compile: `impossible constraint in 'asm'`"** —
you're using `-Os`. Switch the TRISC build to `-O3`. tt-metal's `INSTRUCTION_WORD`
relies on `"n"((x))` constraints that require constexpr propagation; only
`-O3` gives the inliner room to do it.

**"TRISC compute link error: `__builtin_rvtt_ttreplay` undefined"** —
you're using `-mcpu=tt-bh`. Switch to `-mcpu=tt-bh-tensix` for the TRISC
build only (BRISC/NCRISC keep `tt-bh`).

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
| `NUM_CIRCULAR_BUFFERS`    | 64             | `circular_buffer_constants.h`    |
| `launch_msg_buffer_num_entries` | 8       | `hal/generated/dev_msgs.hpp`     |

HAL addresses pulled at runtime (Blackhole Tensix):

| Mailbox / Region            | Offset    | Notes                       |
| --------------------------- | --------- | --------------------------- |
| `LAUNCH`                    | 0x70      | 8 × launch_msg_t            |
| `LAUNCH_MSG_BUFFER_RD_PTR`  | 0x6C      | uint32                      |
| `CORE_INFO`                 | 0x948     | 248 B                       |
| `GO_MSG`                    | 0x3F0     | 4 B (signal byte at offset 0)|
| `GO_MSG_INDEX`              | 0x420     | uint32                      |
| `KERNEL_CONFIG`             | 0x9E00    | RTA + kernel text + CB blob |
| `BANK_TO_NOC_SCRATCH`       | 0x12E00   | 2048 B (zero-fill on boot)  |
| `LOGICAL_TO_VIRTUAL_SCRATCH`| 0x13600   | 32 B (zero-fill on boot)    |

## When making changes

- **Add a new runtime API function**: declare in `include/tt_foil/runtime.hpp`,
  implement in the relevant `src/*.cpp`, thin-wrap in `src/runtime_impl.cpp`.
- **Add a new boot step**: put the helper in its own `src/<name>_init.{hpp,cpp}`
  pair matching the existing per-step files; call from `device_open` between
  the firmware load and the BRISC deassert.
- **Touch dispatch**: the four stages
  (`dispatch_stage_send_reset`, `dispatch_stage_setup`,
  `dispatch_stage_fire_go`, `dispatch_stage_wait_done`) are in an anonymous
  namespace in `dispatch.cpp`; `dispatch_execute_multi` is the only caller
  worth modifying, single-kernel forwards.
- **Add a new example**: copy `examples/tile_copy/build_kernels.sh` if you
  need TRISC; otherwise `examples/noc_passthrough/build_kernels.sh`.
  Both already have the `<tt-foil-build>/firmware/ > JIT cache > pre-compiled`
  preference baked in.

## Commit hygiene

- No "Phase X" / "Step Y" / "Session N" language in commit messages or code.
- Squash incremental commits into 3-5 logical units per phase.
- Co-Authored-By: Claude <noreply@anthropic.com> trailer on AI-assisted
  commits (this is recorded in the user's MEMORY.md too).
