# tt-foil

A lightweight C++ runtime for running pre-compiled kernels on a Tenstorrent
Blackhole chip. Designed for **embedding** into larger systems where the full
tt-metal stack is too heavy.

## Why

The standard tt-metal runtime is ~660K lines and includes JIT kernel compilation,
fast-dispatch firmware, profiling, multi-device mesh support, and Python bindings.
For embedding use cases most of that is unnecessary. tt-foil keeps only what's
needed to cold-boot a chip, load pre-compiled kernels, and dispatch them across a
small set of Tensix cores. Dispatch defaults to slow-dispatch (direct mailbox);
an opt-in on-chip dispatcher (`TT_FOIL_FAST_DISPATCH=1`) is available for
latency-sensitive decode loops.

| Feature                          | tt-metal                   | tt-foil                                |
| -------------------------------- | -------------------------- | -------------------------------------- |
| Codebase size                    | ~660K lines                | ~6.9K lines (core + op-lib + vendored llrt)|
| JIT kernel compilation           | Yes                        | No — pre-compiled ELFs only            |
| Dispatch firmware                | Yes                        | Slow-dispatch (default) + fast-dispatch (opt-in)|
| `libtt_metal.so` link            | Required                   | **Not linked**                         |
| `MetalContext` / IDevice         | Required                   | **Not used** — own UMD + HAL directly  |
| Multi-device / Mesh              | Yes                        | No — single chip                       |
| Multi-Tensix core dispatch       | Yes                        | Yes                                    |
| NOC inter-core (`noc_async_*`)   | Yes                        | Yes (unicast)                          |
| TRISC compute + Circular Buffers | Yes                        | Yes                                    |
| Target hardware                  | WH, BH, Quasar             | Blackhole only                         |
| Runtime dynamic deps             | `libtt_metal.so`, UMD, ... | UMD + fmt (`libtt-umd.so`, `libfmt.so`)|
| Runtime image on disk            | ~27 MB shared libs         | **~5 MB** (binary + libtt-umd.so + libfmt.so)|
| Test binary (stripped)           | n/a (linked against .so)   | **~600 KB**                            |

## Dependencies

**The headline:** tt-foil **needs tt-metal at build time only**. At
runtime the binary has exactly one TT-specific dynamic dep —
`libtt-umd.so`. No `libtt_metal.so`, no `MetalContext`, no
`tt::Cluster`.

### Side-by-side at each phase

|                       | tt-metal app                              | tt-foil app                             |
| --------------------- | ----------------------------------------- | --------------------------------------- |
| **Build inputs**      | tt-metal headers + libs                   | tt-metal headers + libs                 |
| **Build output**      | your binary (links `libtt_metal.so`)      | your binary (statically holds tt-foil)  |
| **Runtime TT deps**   | `libtt_metal.so` (22 MB) + UMD (4.4 MB)   | `libtt-umd.so` (4.4 MB) + `libfmt.so` (0.18 MB) |
| **Runtime image**     | ~27 MB                                    | **~5 MB**                               |

### Phase diagram

```
┌─ BUILD TIME ────────────────────────────────── tt-metal is required here ─┐
│                                                                           │
│  tt-metal source tree                tt-metal build_Release/              │
│  ┌───────────────────────────┐      ┌────────────────────────────────┐    │
│  │ headers  (.h, .hpp)       │      │ libtt-umd.so        (.so)      │    │
│  │ HAL      (.cpp)           │      │ libfmt.so           (.so)      │    │
│  │ firmware (.cc, .ld)       │      │ SFPI g++   (cross compiler)    │    │
│  │ ll_api/* (vendored .cpp)  │      └────────────────────────────────┘    │
│  └───────────────────────────┘                                            │
│             │                                  │                          │
│             │  compile into tt-foil's          │  link libtt-umd /        │
│             │  static libs + firmware ELFs     │  libfmt as deps          │
│             ▼                                  ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                       tt-foil CMake build                        │     │
│  │                                                                  │     │
│  │  src/*.cpp + HAL .cpp + vendored ll_api::memory + tt_elffile     │     │
│  │     ──► libtt_foil.a + libtt_foil_hal_local.a       [STATIC]     │     │
│  │                                                                  │     │
│  │  tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc         │     │
│  │     ──► <build>/firmware/<risc>/<risc>.elf   ×5  [via SFPI g++]  │     │
│  │                                                                  │     │
│  │  tests/*.cpp + examples/build_kernels.sh                         │     │
│  │     ──► test binaries  +  kernel ELFs               [STATIC]     │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                           │
└──────────  At this point tt-metal is no longer needed.  ─────────────────-┘

                                  │
                                  ▼  ship just these to the target host:

┌─ RUNTIME ──────────────────────────────────────── tt-metal NOT needed ───┐
│                                                                          │
│   your_test_binary  (~600 KB stripped)                                   │
│   ┌──────────────────────────────────────────────────────────┐           │
│   │ all of tt-foil statically linked in:                     │           │
│   │   • src/* (device open, dispatch, ELF loader, NOC, ...)  │           │
│   │   • HAL .cpp from tt-metal                               │           │
│   │   • ll_api::memory + tt_elffile (vendored)               │           │
│   │   • tile/bf16 helpers                                    │           │
│   └──────────────────────────────────────────────────────────┘           │
│                       │                                                  │
│                       │ dynamic-links two TT-specific .so files:         │
│                       ▼                                                  │
│              libtt-umd.so.0   (4.4 MB)   ──► PCIe / TLB / DMA            │
│              libfmt.so.11     (0.18 MB)  ──► logging / formatting        │
│                                                  │                       │
│                                                  ▼                       │
│                                          Blackhole chip                  │
│                                            • firmware ELFs (loaded into  │
│                                              Tensix L1 by tt-foil at     │
│                                              device_open)                │
│                                            • kernel ELFs    (loaded      │
│                                              into Tensix L1 by tt-foil   │
│                                              at execute())               │
│                                                                          │
│   Not linked at all:                                                     │
│     ✗ libtt_metal.so   ✗ MetalContext   ✗ tt::Cluster   ✗ JIT cache      │
└──────────────────────────────────────────────────────────────────────────┘
```

In other words: tt-metal acts as a **build-time SDK** (source, HAL,
firmware sources, SFPI compiler, and one shared lib called UMD). After
the CMake build finishes, the only TT-specific files that need to ride
along with the binary are `libtt-umd.so` and `libfmt.so`. See [Firmware ELF
selection](#firmware-elf-selection) for how the runtime resolves the
self-built firmware ELFs.

### Footprint

Measured against `tests/test_matmul_1tile` on Blackhole x86_64:

| Artifact                                        | Size      |
| ----------------------------------------------- | --------- |
| `libtt_foil.a` (static, release build)          | **~7 MB** |
| `libtt_foil_hal_local.a` (HAL .cpp, static)     | ~4 MB     |
| Test binary `test_matmul_1tile` (stripped)      | **~600 KB**|
| `libtt-umd.so.0` (TT PCIe/DMA runtime dep)      | 4.4 MB    |
| `libfmt.so.11` (logging runtime dep)            | 0.18 MB   |
| **Total runtime image on disk** (binary + deps) | **~5 MB** |

An equivalent tt-metal test binary loads `libtt_metal.so` (22 MB)
**and** `libtt-umd.so` (4.4 MB) at runtime — about **27 MB** of
TT-specific shared objects, ~5–6× the tt-foil footprint. tt-foil's
static lib bundles HAL + llrt subset + op-lib; the *image actually
mapped to run a kernel* is a single ~600 KB binary plus `libtt-umd.so`
and `libfmt.so`.

Source-line count: ~3.1K lines of own core runtime + ~2.4K lines op-lib
+ ~1.4K lines vendored from tt-metal (`tt_memory.cpp`, `tt_elffile.cpp`).

### Size-minimized builds

For embedding scenarios — edge deployment, container images, or applications that
bundle multiple large binaries — enable the size-minimized build mode:

```bash
cmake -B build \
  -DTT_FOIL_MINIMIZE_SIZE=ON \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DTT_FOIL_HW_TESTS=ON          # optional
```

`TT_FOIL_MINIMIZE_SIZE=ON` applies the following to `libtt_foil.a` and
`libtt_foil_hal_local.a`, and propagates `--gc-sections` to every downstream
executable that links `TT::Foil`:

| Technique | What it does |
| --------- | ------------ |
| `-ffunction-sections` + `-fdata-sections` | One ELF section per function/variable |
| `-Wl,--gc-sections` (propagated) | Linker discards every unreachable section |
| `-fvisibility=hidden` + `-fvisibility-inlines-hidden` | Internal symbols hidden; aids GC, shrinks export table |
| LTO (`INTERPROCEDURAL_OPTIMIZATION`) | Cross-TU dead code elimination and inlining (graceful fallback if unavailable) |

`MinSizeRel` adds `-Os -DNDEBUG` on top for code-size-optimised codegen.

**Hard floor:** `libtt-umd.so` (4.4 MB) is a runtime dynamic dependency and
cannot be reduced from the tt-foil side. Strip the final binary to drop debug
info (~3.5 MB unstripped → ~600 KB stripped):

```bash
strip --strip-unneeded build/my_app
```

**Incompatibility:** `TT_FOIL_MINIMIZE_SIZE` and `TT_FOIL_DEVICE_PROFILER`
cannot both be ON — Tracy profiler zone hooks require default symbol visibility.
CMake will error if both flags are set.

**Build time:** LTO significantly increases link time on large translation units
(especially the HAL sources). Expect a 2–4× longer link step vs. a standard
Release build.

## Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+ or Clang 14+)
- Blackhole PCIe device
- A built tt-metal source tree on disk (for headers, firmware/kernel
  source, SFPI cross-compiler, and `libtt-umd.so` — tt-foil bundles the
  .cpp it needs but the tree is still the source of those artifacts). No
  prior tt-metal *run* is required: tt-foil compiles the RISC firmware
  itself.

## Getting Started

### 1. Build tt-metal once (for headers + libtt-umd.so + SFPI compiler)

```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
cmake -B build_Release -DCMAKE_BUILD_TYPE=Release
cmake --build build_Release -j$(nproc) --target tt_metal
```

Produces: `lib/libtt-umd.so`, `include/` headers, and the SFPI
cross-compiler at `libexec/tt-metalium/runtime/sfpi/compiler/bin/`.

### 2. Build tt-foil

```bash
git clone https://github.com/tenstorrent/tt-foil.git
cd tt-foil
cmake -B build -DTT_METAL_BUILD_DIR=/path/to/tt-metal/build_Release \
               -DTT_FOIL_HW_TESTS=ON \
               -DTT_FOIL_DEVICE=3
cmake --build build -j$(nproc)
```

tt-foil compiles the 5 RISC firmware ELFs as part of its own build,
under `<tt-foil-build>/firmware/<risc>/<risc>.elf` (+ `*_weakened.elf`
for kernel linking).

### 3. Pre-compile your kernel

Use the SFPI cross-compiler from tt-metal's build tree. The
[`examples/`](examples/) directory has working build scripts; the
simplest is
[`examples/add_two_numbers/build_kernels.sh`](examples/add_two_numbers/build_kernels.sh)
(BRISC only) and the most complete is
[`examples/tile_copy/build_kernels.sh`](examples/tile_copy/build_kernels.sh)
(5 RISCs + CB descriptors).

### 4. Run a kernel from C++

```cpp
#include "tt_foil/runtime.hpp"

auto dev = tt::foil::open_device(/*pcie_index=*/3, "", {{0, 0}});
tt::foil::CoreCoord core{0, 0};

auto a_buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);
auto r_buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);

uint32_t v = 42;
tt::foil::write_buffer(*dev, *a_buf, &v, 4);

std::array<tt::foil::RiscBinary, 1> bins = {{
    {tt::foil::RiscBinary::RiscId::BRISC, "my_kernel.elf"},
}};
auto kernel = tt::foil::load_kernel(*dev, bins, core);
tt::foil::set_runtime_args(*dev, *kernel,
    tt::foil::RiscBinary::RiscId::BRISC,
    std::vector<uint32_t>{
        static_cast<uint32_t>(a_buf->device_addr),
        static_cast<uint32_t>(r_buf->device_addr),
    });

tt::foil::execute(*dev, *kernel);

uint32_t result = 0;
tt::foil::read_buffer(*dev, *r_buf, &result, 4);
tt::foil::close_device(std::move(dev));
```

Full API reference: [`docs/api.md`](docs/api.md).

Optional Tracy-based profiling for Performance + Memory Reports:
[`docs/profiling.md`](docs/profiling.md) (design rationale in
[`docs/memory_management.md`](docs/memory_management.md)).

### Running tests with `ctest`

`tests/CMakeLists.txt` wires the kernel-dir + device env vars per
test, so HW integration tests run end-to-end through `ctest`:

```bash
tt-smi -r 3                       # one-shot, ensures clean chip state
ctest --test-dir build            # runtime regression sweep (default)
ctest --test-dir build -L unit    # host-only unit tests
ctest --test-dir build -L hw      # Blackhole integration tests
ctest --test-dir build -LE model  # everything except model tests
```

The default `ctest` invocation is the **runtime regression sweep** —
it must stay 100% green. It covers core runtime, examples, ops, and
op_lib integration tests. Model-specific integration tests live under
`models/` and are opt-in (see below) because they need exported model
weights/fixtures that aren't in the repo.

HW tests share `RESOURCE_LOCK chip` so they never run concurrently
within a ctest invocation. `TT_FOIL_DEVICE` falls back to the env var
if unset on the CMake line, and to `0` otherwise.

### Running model integration tests

Model-specific tests (Qwen3-VL-2B, etc.) live in
`models/<model>/tests/` and are gated by the `TT_FOIL_MODEL_TESTS`
CMake option. They depend on real-weight binaries produced by the
per-model export script (e.g. `scripts/qwen3_export_weights.sh`),
which aren't in the repo:

```bash
# 1. Export weights (one-time, ~5 min, needs HF cache)
scripts/qwen3_export_weights.sh

# 2. Re-configure with model tests enabled
cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_MODEL_TESTS=ON
cmake --build build -j$(nproc)

# 3. Run just the model tests (label "model")
tt-smi -r 3
ctest --test-dir build -L model
```

Without `-DTT_FOIL_MODEL_TESTS=ON`, the model tests aren't even
built, so a fresh checkout's runtime regression stays green on
machines that haven't materialised the weights.

### Runtime env vars

| Env var                  | Meaning                                                  |
| ------------------------ | -------------------------------------------------------- |
| `TT_METAL_RUNTIME_ROOT`  | tt-metal source root (firmware ELF auto-discovery)       |
| `TT_FOIL_DEVICE`         | PCIe device index, e.g. `3` (default: `0`)               |
| `TT_FOIL_FIRMWARE_DIR`   | Explicit firmware dir; overrides auto-discovery          |
| `TT_FOIL_KERNEL_DIR`     | Directory containing your kernel ELFs (test convention)  |
| `TT_FOIL_FAST_DISPATCH`  | Set to `1` to enable on-chip fast-dispatch (persistent BRISC dispatcher on core (1,0)); recommended for decode loops |
| `TT_FOIL_OPS_DIR`        | Directory containing op kernel ELFs (required for fast-dispatch; e.g. `$PWD/ops`) |

### Firmware ELF selection

`tt-foil` resolves firmware ELFs in this order:

1. `$TT_FOIL_FIRMWARE_DIR` if set (must contain
   `brisc/brisc.elf`, `ncrisc/ncrisc.elf`, `trisc{0,1,2}/trisc{0,1,2}.elf`).
2. `$TT_FOIL_BUILD_FIRMWARE_DIR` if set, or the path baked in at CMake
   configure time when `TT_FOIL_BUILD_FIRMWARE=ON` (default) —
   `<tt-foil-build>/firmware/`. **This is the default source**: tt-foil
   compiles each firmware ELF directly from
   `tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc` via
   SFPI g++ and runs `tools/tt_foil_weaken` to produce the
   `*_weakened.elf` companion.
3. The newest matching dir under `$HOME/.cache/tt-metal-cache/<hash>/firmware/`
   (tt-metal JIT cache).
4. Fallback: the newest matching dir under
   `$TT_METAL_RUNTIME_ROOT/tt_metal/pre-compiled/<hash>/`.

Kernel ↔ firmware ABI consistency matters: a kernel ELF and the
firmware it runs against must come from the same build (their
`*_weakened.elf` must match the firmware loaded onto the chip). The
`build_kernels.sh` scripts under `examples/*/` follow the same
precedence order, so by default they link against tt-foil's self-built
`build/firmware/`.

## Supported Models

End-to-end neural network forward passes that run on a single Tensix
core. Each model lives under [`models/`](models/) and is exercised by
a paired test under [`tests/`](tests/).

| Model | Description | Test |
| --- | --- | --- |
| **ResNet-20** | Pretrained akamaster checkpoint + one CIFAR-10 test image. Full 19-conv + GAP + FC forward pass on device, BN folded into per-conv biases by [`models/resnet20/export.py`](models/resnet20/export.py). Worst per-class logit drift 0.14 vs the fp32 reference; dev argmax matches ref argmax = "cat". Data generated automatically on first build (requires Python + torch). | [`test_resnet20`](tests/test_resnet20.cpp) |

### Single-op kernel demos

Lower-level building blocks — one device kernel each, isolated for
study or copy-paste — live under [`examples/`](examples/): matmul
variants, conv via im2col (1×1 / 3×3 / 3×3 stride 2 / 7×7), maxpool
2×2 and 3×3, SFPU eltwise (ReLU, add, bias broadcast, fused
bias+ReLU), global average pool, multi-tile residual add, NOC
passthrough, multi-core sharded matmul, and a couple of "five RISCs
plus circular buffers" tutorials. Each example ships its own
`build_kernels.sh` and `tests/test_<name>.cpp`.

## API

See [`docs/api.md`](docs/api.md) for the full reference. Public
surface lives in `<tt_foil/runtime.hpp>`:

```
open_device / close_device
allocate_buffer / write_buffer / read_buffer
load_kernel / set_runtime_args / release_kernels
register_cbs               // circular buffer descriptors
execute                    // single-kernel + multi-kernel variants
make_noc_unicast_addr      // pack a 64-bit peer-L1 NOC address
make_noc_dram_addr         // pack a 64-bit DRAM-bank NOC address
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
