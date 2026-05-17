# tt-foil

A lightweight C++ runtime for running pre-compiled kernels on a Tenstorrent
Blackhole chip. Designed for **embedding** into larger systems where the full
tt-metal stack is too heavy.

## Why

The standard tt-metal runtime is ~660K lines and includes JIT kernel compilation,
fast-dispatch firmware, profiling, multi-device mesh support, and Python bindings.
For embedding use cases most of that is unnecessary. tt-foil keeps only what's
needed to cold-boot a chip, load pre-compiled kernels, and dispatch them across a
small set of Tensix cores.

| Feature                          | tt-metal                   | tt-foil                                |
| -------------------------------- | -------------------------- | -------------------------------------- |
| Codebase size                    | ~660K lines                | ~2.5K lines (incl. bundled llrt subset)|
| JIT kernel compilation           | Yes                        | No — pre-compiled ELFs only            |
| Dispatch firmware                | Yes                        | No — slow-dispatch via direct mailbox  |
| `libtt_metal.so` link            | Required                   | **Not linked**                         |
| `MetalContext` / IDevice         | Required                   | **Not used** — own UMD + HAL directly  |
| Multi-device / Mesh              | Yes                        | No — single chip                       |
| Multi-Tensix core dispatch       | Yes                        | Yes                                    |
| NOC inter-core (`noc_async_*`)   | Yes                        | Yes (unicast)                          |
| TRISC compute + Circular Buffers | Yes                        | Yes                                    |
| Target hardware                  | WH, BH, Quasar             | Blackhole only                         |
| Runtime dynamic deps             | `libtt_metal.so`, UMD, ... | UMD only (`libtt-umd.so`)              |
| Runtime image on disk            | ~27 MB shared libs         | **~5 MB** (binary + libtt-umd.so)      |
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
| **Runtime TT deps**   | `libtt_metal.so` (22 MB) + UMD (4.3 MB)   | `libtt-umd.so` only (4.3 MB)            |
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
│                       │ dynamic-links one TT-specific .so:               │
│                       ▼                                                  │
│              libtt-umd.so.0   (4.3 MB)   ──► PCIe / TLB / DMA            │
│                                                  │                       │
│                                                  ▼                       │
│                                          Blackhole chip                  │
│                                            • firmware ELFs (loaded into │
│                                              Tensix L1 by tt-foil at    │
│                                              device_open)               │
│                                            • kernel ELFs    (loaded     │
│                                              into Tensix L1 by tt-foil │
│                                              at execute())             │
│                                                                          │
│   Not linked at all:                                                     │
│     ✗ libtt_metal.so   ✗ MetalContext   ✗ tt::Cluster   ✗ JIT cache      │
└──────────────────────────────────────────────────────────────────────────┘
```

In other words: tt-metal acts as a **build-time SDK** (source, HAL,
firmware sources, SFPI compiler, and one shared lib called UMD). After
the CMake build finishes, the only TT-related file that needs to ride
along with the binary is `libtt-umd.so`. See [Firmware ELF
selection](#firmware-elf-selection) for how the runtime resolves the
self-built firmware ELFs.

### Footprint

Measured against `tests/test_matmul_1tile` on Blackhole x86_64:

| Artifact                                        | Size      |
| ----------------------------------------------- | --------- |
| `libtt_foil.a` (static, all of tt-foil's logic) | **6.7 MB**|
| `libtt_foil_hal_local.a` (HAL .cpp, static)     | 4.2 MB    |
| Test binary `test_matmul_1tile` (stripped)      | **633 KB**|
| Test binary (unstripped, with debug)            | 820 KB    |
| `libtt-umd.so.0` (the only runtime dynamic dep) | 4.3 MB    |
| **Total runtime image on disk** (binary + UMD)  | **~5 MB** |

An equivalent tt-metal test binary loads `libtt_metal.so` (22 MB)
**and** `libtt-umd.so` (4.3 MB) at runtime — about **27 MB** of
TT-specific shared objects, ~5–6× the tt-foil footprint. tt-foil's
static lib is 6.7 MB because it bundles HAL + llrt subset; the *image
actually mapped to run a kernel* is a single ~600 KB binary plus
`libtt-umd.so`.

Source-line count: ~700 lines of own runtime code + ~1.2K lines
vendored from tt-metal (`tt_memory.cpp`, `tt_elffile.cpp`).

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

### Running tests with `ctest`

`tests/CMakeLists.txt` wires the kernel-dir + device env vars per
test, so HW integration tests run end-to-end through `ctest`:

```bash
tt-smi -r 3                    # one-shot, ensures clean chip state
ctest --test-dir build         # all tests, serialised on chip
ctest --test-dir build -L unit # host-only unit tests
ctest --test-dir build -L hw   # Blackhole integration tests
```

HW tests share `RESOURCE_LOCK chip` so they never run concurrently
within a ctest invocation. `TT_FOIL_DEVICE` falls back to the env var
if unset on the CMake line, and to `0` otherwise.

### Runtime env vars

| Env var                  | Meaning                                                  |
| ------------------------ | -------------------------------------------------------- |
| `TT_METAL_RUNTIME_ROOT`  | tt-metal source root (firmware ELF auto-discovery)       |
| `TT_FOIL_DEVICE`         | PCIe device index, e.g. `3` (default: `0`)               |
| `TT_FOIL_FIRMWARE_DIR`   | Explicit firmware dir; overrides auto-discovery          |
| `TT_FOIL_KERNEL_DIR`     | Directory containing your kernel ELFs (test convention)  |

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

## Examples

The [`examples/`](examples/) directory contains end-to-end programs
paired with tests under [`tests/`](tests/). Each example has a
`build_kernels.sh` that produces the kernel ELFs and a
`test_<name>.cpp` that drives them.

Pick by what you want to learn:

| If you want to see…                              | Look at                                      |
| ------------------------------------------------ | -------------------------------------------- |
| A single BRISC kernel doing arithmetic           | [`examples/add_two_numbers/`](examples/add_two_numbers/) |
| BRISC ↔ BRISC NOC unicast between two cores      | [`examples/noc_passthrough/`](examples/noc_passthrough/) |
| All 5 RISCs + circular buffers + `copy_tile`     | [`examples/tile_copy/`](examples/tile_copy/) |
| bf16 matmul tiled through DRAM                   | [`examples/matmul_dram/`](examples/matmul_dram/) |
| Multi-core sharded matmul (2×2 / 4×2 grids)      | [`examples/matmul_2x2/`](examples/matmul_2x2/), [`examples/matmul_4x2/`](examples/matmul_4x2/) |
| NOC weight-broadcast (DRAM read once + forward)  | [`examples/matmul_2core_mcast/`](examples/matmul_2core_mcast/) |
| 1×1 / 3×3 convolution via im2col                 | [`examples/conv_1x1/`](examples/conv_1x1/), [`examples/conv_3x3/`](examples/conv_3x3/) |
| 3×3 stride=2 conv (ResNet downsample)            | [`examples/conv_3x3_s2/`](examples/conv_3x3_s2/) |
| Eltwise primitives (ReLU, add, bias broadcast)   | [`examples/relu/`](examples/relu/), [`examples/add_tiles/`](examples/add_tiles/), [`examples/bias_add/`](examples/bias_add/) |

## API

See [`docs/api.md`](docs/api.md) for the full reference. Public
surface lives in `<tt_foil/runtime.hpp>`:

```
open_device / close_device
allocate_buffer / write_buffer / read_buffer
load_kernel / set_runtime_args
register_cbs               // circular buffer descriptors
execute                    // single-kernel + multi-kernel variants
make_noc_unicast_addr      // pack a 64-bit peer-L1 NOC address
make_noc_dram_addr         // pack a 64-bit DRAM-bank NOC address
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
