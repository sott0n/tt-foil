# Porting tt-metal kernels to tt-foil

Step-by-step guide for taking a kernel from the tt-metal source tree
(under `tt_metal/programming_examples/` or `ttnn/cpp/ttnn/operations/`)
and running it on the tt-foil runtime, **without modifying the kernel
sources** in the typical case.

The first kernel ported this way was the `eltwise_sfpu` programming
example — see [examples/eltwise_sfpu/](../examples/eltwise_sfpu/) and
[tests/test_eltwise_sfpu.cpp](../tests/test_eltwise_sfpu.cpp) for the
worked reference.

---

## What works today

| Kernel feature | Works as-is? | Notes |
|---|---|---|
| Classic `dataflow_api.h` (`noc_async_read/write`, `cb_*`) | Yes | tt-foil's existing examples use this |
| `experimental::Noc` / `experimental::CircularBuffer` | Yes | Header-only wrappers over the classic API |
| `TensorAccessor` for **DRAM-interleaved** buffers | Yes | Depends on `BANK_TO_NOC_SCRATCH` populated by `src/bank_tables_init.cpp` (commit `1f2c948`) |
| `TensorAccessor` for **L1-interleaved** buffers | No | `l1_bank_to_noc_xy` is still zero-filled; populate it the same way as DRAM when needed |
| SFPU intrinsics (`exp_tile`, `silu_tile`, etc.) | Yes | TRISC compute build is fully supported (see `chlkc_descriptors.h` / `chlkc_list.h` stubs) |
| `get_tile_size(cb_id)` | Yes | Requires `chlkc_descriptors.h` on the include path |
| Compile-time args (`get_compile_time_arg_val(N)`) | Yes | Wire via `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...` in `build_kernels.sh` |
| `get_noc_addr_from_logical_xy()` | **No** | `LOGICAL_TO_VIRTUAL_SCRATCH` is zero-filled; would resolve to (0, 0). Use `tt::foil::make_noc_unicast_addr` host-side and pass via RTA |
| BRISC `noc_async_write` to peer Tensix L1 on NOC 0 | **No** | Blackhole quirk — pass `noc=1` explicitly. Reads on NOC 0 are fine |
| Sharded buffers | Partial | Bank table covers interleaved only; sharded paths need extra CTA wiring |
| Multi-page DRAM-interleaved with `n_tiles > 1` | Partial | Kernel side works; host needs to fan writes across DRAM channels (not yet helper'd — see [Limitations](#limitations)) |
| Fast dispatch / mesh / multi-CQ | No | tt-foil is single-chip, slow-dispatch by default (`TT_FOIL_FAST_DISPATCH=1` is op-lib only) |

If your target kernel uses anything in the **No** rows, you'll need to
either modify the kernel or extend tt-foil. The rest of this doc
assumes the kernel is in the **Yes** rows.

---

## How to decide what to look for

Open the tt-metal kernel source and scan for:

1. **API style** — modern `experimental::Noc` / `TensorAccessor` (need
   bank-table + CTAs) or classic `noc_async_read(addr, ...)`
   (no bank table needed, simpler).
2. **Compile-time args** — every `get_compile_time_arg_val(N)` call
   becomes a slot in `KERNEL_COMPILE_TIME_ARGS=...`. For
   `TensorAccessorArgs<CTA_OFFSET>` the minimum is two slots: an
   `ArgsConfig` flag bitmask and the page size in bytes.
3. **Logical-coord NOC** — `get_noc_addr_from_logical_xy()` won't work
   under tt-foil (see [Limitations](#limitations)). Grep for it.
4. **Sharded vs interleaved** — sharded TensorAccessors use additional
   CTAs (rank, shard shape, bank coords). If you see
   `RuntimeRank | RuntimeShardShape | ...` in the source, expect
   more wiring.
5. **BRISC NOC writes** — if a BRISC kernel does
   `noc_async_write(...)` to peer L1, check whether it passes
   `noc=1` or relies on the default. If default, the kernel needs
   a one-line change.

---

## Porting recipe

This is the same flow used for `eltwise_sfpu`. Substitute your
kernel name everywhere you see `<NAME>` (e.g. `gelu`, `concat`).

### 1. Lay out the example directory

```bash
mkdir -p examples/<NAME>/kernels
cp <tt-metal>/kernels/<reader>.cpp  examples/<NAME>/kernels/reader.cpp
cp <tt-metal>/kernels/<writer>.cpp  examples/<NAME>/kernels/writer.cpp
cp <tt-metal>/kernels/<compute>.cpp examples/<NAME>/kernels/compute.cpp
```

These three names (`reader.cpp`, `writer.cpp`, `compute.cpp`) are a
soft convention from tt-foil's existing examples — `build_kernels.sh`
references them. Pure-dataflow ops can omit `compute.cpp`.

**Do not modify the files.** If you find yourself needing to (e.g.
because the kernel uses an unsupported feature), document the change
clearly in a comment — that's a signal the porting story has a gap.

### 2. Identify the CTA values for `TensorAccessor`

If the kernel does `TensorAccessor<...>` or `TensorAccessorArgs<...>`,
the compile-time args slot layout is:

| CTA index | Meaning |
|---|---|
| `CTA_OFFSET + 0` | `ArgsConfig` flag bitmask (see `tensor_accessor::ArgConfig` enum) |
| `CTA_OFFSET + 1` | Aligned page size in bytes |
| `CTA_OFFSET + 2..N` | Only used for sharded buffers — rank, shape, bank coords |

For a vanilla DRAM-interleaved tensor (the common case):

```
flags = ArgConfig::IsDram = 1 << 1 = 0x2
page_size = 32 * 32 * sizeof(bf16) = 2048
```

so `KERNEL_COMPILE_TIME_ARGS="2,2048"`. For interleaved L1, use
`flags = 0x0`. For sharded, see `tensor_accessor_args.h`.

### 3. Write `examples/<NAME>/build_kernels.sh`

Start from `examples/eltwise_sfpu/build_kernels.sh` — it has:

- Firmware path auto-resolution (tt-foil build → JIT cache → pre-compiled fallback).
- BRISC / NCRISC compile with `KERNEL_COMPILE_TIME_ARGS` injected.
- TRISC compute compile (3 variants: UNPACK/MATH/PACK).
- The shared `chlkc_descriptors.h` stub for tile size + data format.
- The TRISC `chlkc_list.h` stub.
- The manifest write at the end (stale-ELF guard).

Things you will likely change:

- **CB formats / shapes** in the `chlkc_descriptors.h` heredoc. The
  stub assumes bf16 (`format=5`) on CB 0 (input) and CB 16 (output)
  with `tile_size=2048`. If your kernel uses different CB indices,
  formats, or tile shapes, update those array slots. Other slots stay
  at `255` (invalid format) so any accidental access trips.
- **`MATH_FIDELITY` and `DST_ACCUM_MODE`**:
  - `MATH_FIDELITY` lives in the `#if defined(UCK_CHLKC_MATH) || ...`
    block of `chlkc_descriptors.h`. For pure SFPU ops `255` (LoFi)
    is fine. For reduce / sum-of-squares set it to `4` (HiFi4) — LoFi
    has ~5% relative error on those (see RMSNorm invariant in CLAUDE.md).
  - `DST_ACCUM_MODE` is in `chlkc_list.h`. Set `true` if your compute
    kernel uses `acc_to_dest` paths.
- **Compile-time args**: pass to `build_one` as the 5th arg, e.g.
  `build_one brisc 0 "$HERE/kernels/reader.cpp" reader.brisc "$YOUR_CT_ARGS"`.

### 4. Test the kernel build standalone

```bash
TT_METAL_ROOT=$PWD/third_party/tt-metal \
    bash examples/<NAME>/build_kernels.sh
```

You should get 5 ELFs in `examples/<NAME>/prebuilt/`:
- `reader.brisc.elf`
- `writer.ncrisc.elf`
- `compute.trisc{0,1,2}.elf` (skip if no compute)

If this step fails, see [Troubleshooting](#troubleshooting).

### 5. Write the host runner / test

Create `tests/test_<NAME>.cpp`. The skeleton (matches
`test_eltwise_sfpu.cpp`):

```cpp
#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"        // register_cbs (private header)
#include "tile_utils.hpp"       // bf16 helpers, tile layout

namespace tf = tt::foil;
namespace tut = tt::foil::test;

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    auto dev = tf::open_device(/*pcie=*/0, "", {{0, 0}});
    tf::CoreCoord core{0, 0};

    // DRAM buffers + L1 CB backing buffers
    auto dram_in  = tf::allocate_buffer(*dev, tf::BufferLocation::DRAM, kBytes);
    auto dram_out = tf::allocate_buffer(*dev, tf::BufferLocation::DRAM, kBytes);
    auto cb_in    = tf::allocate_buffer(*dev, tf::BufferLocation::L1, kTileBytes, core);
    auto cb_out   = tf::allocate_buffer(*dev, tf::BufferLocation::L1, kTileBytes, core);

    // ... fill dram_in with your test input ...
    tf::write_buffer(*dev, *dram_in, input.data(), kBytes);

    using R = tf::RiscBinary;
    auto kernel = tf::load_kernel(*dev, {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }}, core);

    // CB descriptors. Indices match the kernel's `tt::CBIndex::c_N`.
    std::array<tf::CbConfig, 2> cbs = {{
        {0,  cb_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tf::register_cbs(*dev, *kernel, cbs);

    // Runtime args. Match the kernel's get_arg_val(N) order.
    // For DRAM TensorAccessor: pass the buffer's device_addr as the
    // bank_base_address — the kernel adds bank_to_dram_offset (zero) and
    // strides per page itself.
    tf::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(dram_in->device_addr), kNumTiles});
    tf::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC,
        std::array<uint32_t, 2>{
            static_cast<uint32_t>(dram_out->device_addr), kNumTiles});
    // TRISC0/1/2 each have their own RTA slot — set the same value 3x
    // if all three call get_arg_val(N).
    const std::array<uint32_t, 1> ra_trisc{kNumTiles};
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC0, ra_trisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC1, ra_trisc);
    tf::set_runtime_args(*dev, *kernel, R::RiscId::TRISC2, ra_trisc);

    tf::execute(*dev, *kernel);

    // ... read dram_out, compare against reference ...
}
```

Key tt-foil API surface you'll touch:

| Function | Purpose |
|---|---|
| `tf::open_device(pcie, fw_dir, cores)` | Cold-boot. `cores` lists which Tensix cores to bring up. |
| `tf::allocate_buffer(dev, location, bytes, core)` | Bump allocator. `BufferLocation::DRAM` uses channel 0. |
| `tf::write_buffer` / `tf::read_buffer` | Host ↔ device copy. |
| `tf::load_kernel(dev, binaries, core)` | Parse + relocate ELFs, stage RTAs. |
| `tf::register_cbs(dev, kernel, cbs)` | Build the CB descriptor blob in L1. |
| `tf::set_runtime_args(dev, kernel, risc, args)` | Per-RISC RTA. |
| `tf::execute(dev, kernel)` | Reset → setup → fire GO → poll DONE. |
| `tf::make_noc_unicast_addr(dev, core, l1_addr)` | Build a NOC address host-side when the kernel doesn't use TensorAccessor. |

### 6. Register the test

In `tests/CMakeLists.txt`:

```cmake
tt_foil_hw_test(test_<NAME> test_<NAME>.cpp "${EX}/<NAME>/prebuilt")
```

`tt_foil_hw_test` auto-detects `build_kernels.sh` in the parent dir
and wires it as a CMake dependency, so a stale kernel rebuild gets
triggered on the next `cmake --build`.

### 7. Run + verify regression

```bash
cmake --build build -j$(nproc)
$HOME/tt-venv/bin/tt-smi -r 0
ctest --test-dir build -R test_<NAME>     # your new test
ctest --test-dir build                    # full runtime regression — must stay 100% green
```

If your test passes but the full sweep regresses, something in your
change leaked beyond the new example. Likely culprits: edits to
`scripts/kernel_build_helpers.sh`, `cmake/tt_foil_kernels.cmake`,
or shared headers. Bisect the diff.

---

## Limitations

### Multi-page DRAM-interleaved (`n_tiles > 1`)

`TensorAccessor` will stripe pages across all 8 DRAM channels:
page 0 → bank 0 → channel 0, page 1 → bank 1 → channel 1, …,
page 8 → bank 0 channel 0 at offset `page_size`, etc.

`tt::foil::allocate_buffer(BufferLocation::DRAM, ...)` allocates
from channel 0 only, and `write_buffer` writes to channel 0 only.
So `n_tiles > 1` with the modern API requires either:

1. **Single-tile workloads** (`n_tiles == 1`) — page 0 is in channel
   0, the existing allocator + writer suffice. This is what
   `test_eltwise_sfpu` does.
2. **A new multi-channel host helper** — needs to allocate the same
   logical offset in every channel, fan host writes across all
   channels, and read back in the same shape. Not yet implemented
   in tt-foil. When you need it, add a `write_dram_interleaved` /
   `read_dram_interleaved` pair next to `write_buffer` in
   `src/buffer.cpp`; the per-channel DRAM cores can be enumerated
   from `umd::soc_descriptor::get_dram_core_for_channel` (the same
   call `src/bank_tables_init.cpp` already uses).

Until then, keep `n_tiles == 1` or use the legacy
`make_noc_dram_addr` + RTA path that tt-foil's existing examples
use (`examples/matmul_dram` is the canonical reference).

### L1-interleaved

`l1_bank_to_noc_xy[]` is still zero-filled in
`src/bank_tables_init.cpp`. To support L1-interleaved
`TensorAccessor`, mirror the DRAM block in `init_bank_tables`:

```cpp
for (uint32_t noc = 0; noc < num_nocs; ++noc) {
    for (uint32_t bank = 0; bank < kNumL1Banks; ++bank) {
        // L1 bank → worker core mapping comes from
        // tt-metal's L1BankingAllocator (l1_noc_coord_per_bank in
        // RiscFirmwareInitializer::generate_device_bank_to_noc_tables).
        // For a tt-foil-side implementation, see the soc_descriptor
        // worker grid; need to mirror the L1 remap logic.
    }
}
```

Also note `IS_NOT_POW2_NUM_L1_BANKS` is not currently defined
in `build_firmware.sh` despite `NUM_L1_BANKS=140` being non-pow2.
Fix that when wiring L1-interleaved.

### `get_noc_addr_from_logical_xy`

`LOGICAL_TO_VIRTUAL_SCRATCH` is zero-filled. If a kernel calls
this, it resolves to NOC (0, 0). Two options:

1. Modify the kernel to take a TRANSLATED NOC coord via RTA (using
   `tt::foil::make_noc_unicast_addr` host-side). This is the
   convention in all tt-foil examples that need NOC inter-core
   communication (`examples/noc_passthrough`, `examples/matmul_2core_mcast`).
2. Populate `LOGICAL_TO_VIRTUAL_SCRATCH` in `init_bank_tables` —
   same pattern as the DRAM bank table. Not yet done because no
   tt-foil kernel needs it.

### BRISC NOC writes to peer L1

On Blackhole, BRISC's default NOC 0 doesn't reliably handle writes
to peer Tensix L1 (ACK never returns). Pass `noc=1` explicitly:

```cpp
constexpr uint8_t kWriteNoc = 1;
noc_async_write_one_packet(src, dst_noc_addr, size, kWriteNoc);
noc_async_write_barrier(kWriteNoc);
```

NCRISC defaults to NOC 1 already, so writer kernels on NCRISC need
no change. See `examples/matmul_2core_mcast/kernels/reader_producer.cpp`
for the working pattern. tt-metal's stock kernels usually default to
NOC 0 on BRISC — when porting one that writes to peer L1, this is
the rare case where the kernel source needs a small edit.

---

## Troubleshooting

### `error: '<NAME>' was not declared in this scope`

The kernel uses an API that needs a TU-level include we haven't set up.

- `get_tile_size`, `get_tile_num_faces`, etc. → add
  `chlkc_descriptors.h` to `$BUILD` (the include path already covers
  it via `-I"$BUILD"`). See `examples/eltwise_sfpu/build_kernels.sh`
  for the stub layout.
- `TensorAccessor`, `experimental::Noc` → confirm the include path
  has `tt_metal/hw/inc/experimental/`. Already in the example's
  COMMON_CFLAGS.

### `cb_reserve_back` hangs forever

Almost always means the kernel ELF was linked against a
`*_weakened.elf` that doesn't match the firmware actually loaded on
the chip. `setup_local_cb_read_write_interfaces` silently misses the
`cb_interface[]` writes and every CB read returns zero.

Fix: rebuild the kernels against the same firmware tree tt-foil's
`firmware_paths.cpp` resolves at runtime. By default both default to
`<build>/firmware/`. Setting `TT_FOIL_VERIFY_MANIFEST=1` makes the
runtime assert on hash mismatch — turn it on while iterating.

### Output is all zeros

Usually the input never made it to the kernel:

1. **Wrong bank table** — was the bank table populated? Check
   `src/bank_tables_init.cpp` is calling `init_bank_tables`, not
   leaving zeros. Quick smoke test: run
   `ctest --test-dir build -R test_eltwise_sfpu` — that test
   asserts non-zero output and acts as a canary.
2. **`row_major_to_tile` trap** — the helper *appends* to its output
   vector, doesn't overwrite. The destination must start empty.
   See the [CLAUDE.md](../CLAUDE.md) invariant for the full
   gotcha.
3. **`CbConfig.fifo_size`** — must equal `num_pages * page_size`,
   not just `page_size`. With `Wt=1` the two coincide and bugs hide;
   they surface at `Wt > 1`.
4. **DRAM write to wrong channel** — confirm you wrote to channel 0
   (the only one `BufferLocation::DRAM` allocates from today).

### TRISC compute won't compile

| Error | Cause | Fix |
|---|---|---|
| `impossible constraint in 'asm'` | TRISC built with `-Os` | Switch to `-O3` for the TRISC step (already in `build_compute` in eltwise_sfpu's script). |
| `__builtin_rvtt_ttreplay undefined` | `-mcpu=tt-bh` (not `tt-bh-tensix`) | Use `-mcpu=tt-bh-tensix` for TRISC only. BRISC/NCRISC stay on `tt-bh`. |
| `chlkc_list.h not found` | TRISC `chlkc_list.h` stub not written to `$BUILD` | The `build_compute` function in `build_kernels.sh` writes it as a heredoc — confirm `$BUILD` is on `-I`. |

### NOC write reaches (0, 0)

The kernel called `get_noc_addr_from_logical_xy()` or
`get_noc_addr(bank_id)` on an L1-interleaved buffer. Both rely on
zero-filled scratches in tt-foil. See [Limitations](#limitations).

---

## Reference: the eltwise_sfpu walkthrough

The complete change set for the first port is split across 3
commits on `main` (run `git log --oneline --grep="eltwise_sfpu\|BANK_TO_NOC"`):

- `74173af test(regression): split runtime sweep from model integration tests` —
  housekeeping so the sweep is meaningful.
- `1f2c948 feat(boot): populate BANK_TO_NOC_SCRATCH with real DRAM bank table` —
  the foundational change that makes `TensorAccessor` work for
  DRAM-interleaved.
- `623858b feat(examples): eltwise_sfpu — first tt-metal kernel running unchanged` —
  the actual kernel port + host runner + test.

Read those diffs back-to-back for a concrete sense of what each
porting step looks like in practice.
