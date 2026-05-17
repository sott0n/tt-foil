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

| Feature                       | tt-metal                   | tt-foil                                |
| ----------------------------- | -------------------------- | -------------------------------------- |
| Codebase size                 | ~660K lines                | ~2.5K lines (incl. bundled llrt subset)|
| JIT kernel compilation        | Yes                        | No — pre-compiled ELFs only            |
| Dispatch firmware             | Yes                        | No — slow-dispatch via direct mailbox  |
| `libtt_metal.so` link         | Required                   | **Not linked**                         |
| `MetalContext` / IDevice      | Required                   | **Not used** — own UMD + HAL directly  |
| Multi-device / Mesh           | Yes                        | No — single chip                       |
| Multi-Tensix core dispatch    | Yes                        | Yes (v3)                               |
| NOC inter-core (`noc_async_*`)| Yes                        | Yes (v3, unicast)                      |
| TRISC compute + Circular Buffers | Yes                     | Yes (v4)                               |
| Target hardware               | WH, BH, Quasar             | Blackhole only                         |
| Runtime dynamic deps          | `libtt_metal.so`, UMD, ... | UMD only (`libtt-umd.so`)              |
| Runtime image on disk         | ~27 MB shared libs          | **~5 MB** (binary + libtt-umd.so)      |
| Test binary (stripped)        | n/a (linked against .so)    | **~600 KB**                            |

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

Measured against `tests/test_matmul_1tile` (v5-1, the largest test
binary so far — full 5-RISC + 2 input CBs + matmul), on Blackhole x86_64
host:

| Artifact                                        | Size      |
| ----------------------------------------------- | --------- |
| `libtt_foil.a` (static, all of tt-foil's logic) | **6.7 MB**|
| `libtt_foil_hal_local.a` (HAL .cpp, static)     | 4.2 MB    |
| Test binary `test_matmul_1tile` (stripped)      | **633 KB**|
| Test binary (unstripped, with debug)            | 820 KB    |
| `libtt-umd.so.0` (the only runtime dynamic dep) | 4.3 MB    |
| **Total runtime image on disk** (binary + UMD)  | **~5 MB** |

For comparison, an equivalent tt-metal test binary loads
`libtt_metal.so` (22 MB) **and** `libtt-umd.so` (4.3 MB) at runtime —
about **27 MB** of TT-specific shared objects, ~5–6× the tt-foil
footprint. tt-foil's own static lib is 6.7 MB because it bundles HAL +
llrt subset; the *image actually mapped to run a kernel* is a single
~600 KB binary plus libtt-umd.so.

Source-line count: ~700 lines of own runtime code + ~1.2K lines
vendored from tt-metal (`tt_memory.cpp`, `tt_elffile.cpp`).

## Status

| Phase      | Highlight                                                       |
| ---------- | --------------------------------------------------------------- |
| v1         | Single BRISC kernel on one Tensix core, slow-dispatch           |
| v2 (A)     | BRISC + NCRISC concurrent on the same core                      |
| v2 (B1)    | I/O switched to UMD direct (`tt::Cluster` calls dropped)        |
| v2 (B2)    | `CreateDevice` replaced with in-tree cold-boot sequence         |
| v2 (B3)    | `libtt_metal.so` link removed — HAL compiled into tt-foil       |
| v3         | Multi-Tensix boot, multi-kernel dispatch, NOC inter-core L1↔L1  |
| v4         | TRISC compute + Circular Buffers — full 5-RISC pipeline, copy_tile MVP |

Runs on Blackhole (Device=3 verified). End-to-end tests pass with only
`libtt-umd.so` linked.

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

### 1. Build tt-metal once (for headers + firmware artifacts)

```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
cmake -B build_Release -DCMAKE_BUILD_TYPE=Release
cmake --build build_Release -j$(nproc) --target tt_metal
```

This produces:

- `build_Release/lib/libtt-umd.so` (the only dynamic dep tt-foil links)
- `build_Release/include/` (UMD, tt_stl, tt-logger, enchantum, fmt, spdlog headers)
- `build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++`
  (cross-compiler used for both tt-foil's firmware and kernel builds)

tt-foil's own CMake build then produces the 5 firmware ELFs at
`<tt-foil-build>/firmware/{brisc,ncrisc,trisc0,trisc1,trisc2}/*.elf` (plus
`*_weakened.elf` companions for kernel linking). No prior tt-metal run is
required.

### 2. Build tt-foil

```bash
git clone https://github.com/tenstorrent/tt-foil.git
cd tt-foil
cmake -B build -DTT_METAL_BUILD_DIR=/path/to/tt-metal/build_Release \
               -DTT_FOIL_HW_TESTS=ON
cmake --build build -j$(nproc)
```

`TT_METAL_BUILD_DIR` can also be passed via env var. If both `TT_METAL_BUILD_DIR`
and `TT_METAL_SOURCE_DIR` are unset, tt-foil infers the source root from the
parent of the build dir.

### 3. Pre-compile your kernel

Use the SFPI cross-compiler that ships with tt-metal — see
[`examples/tile_copy/build_kernels.sh`](examples/tile_copy/build_kernels.sh)
for the most complete example (BRISC + NCRISC + TRISC0/1/2 with the
chlkc-stub trick for the compute build). The simpler
[`examples/add_two_numbers/build_kernels.sh`](examples/add_two_numbers/build_kernels.sh)
covers just the data-movement RISCs.

### 4. Run

```cpp
#include "tt_foil/runtime.hpp"

// Open chip 3 and cold-boot logical core (0,0).
auto dev = tt::foil::open_device(3, "", {{0, 0}});

tt::foil::CoreCoord core{0, 0};
auto a_buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);
auto r_buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);

uint32_t v = 42;
tt::foil::write_buffer(*dev, *a_buf, &v, 4);

std::array<tt::foil::RiscBinary, 1> bins = {{
    {tt::foil::RiscBinary::RiscId::BRISC, "my_kernel.elf"},
}};
auto kernel = tt::foil::load_kernel(*dev, bins, core);

std::vector<uint32_t> args = {
    static_cast<uint32_t>(a_buf->device_addr),
    static_cast<uint32_t>(r_buf->device_addr),
};
tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::BRISC, args);

tt::foil::execute(*dev, *kernel);

uint32_t result = 0;
tt::foil::read_buffer(*dev, *r_buf, &result, 4);

tt::foil::close_device(std::move(dev));
```

### Required env vars at run time

| Env var                  | Meaning                                                  |
| ------------------------ | -------------------------------------------------------- |
| `TT_METAL_RUNTIME_ROOT`  | tt-metal source root (firmware ELF auto-discovery)       |
| `TT_FOIL_DEVICE`         | PCIe device index, e.g. `3` (default: `0`)               |
| `TT_FOIL_FIRMWARE_DIR`   | Explicit firmware dir; overrides auto-discovery          |
| `TT_FOIL_KERNEL_DIR`     | Directory containing your kernel ELFs (test convention)  |

### Running tests with `ctest`

`tests/CMakeLists.txt` wires the kernel-dir + device env vars per
test, so HW integration tests run end-to-end through `ctest`:

```bash
cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_DEVICE=3
cmake --build build -j$(nproc)
tt-smi -r 3                    # one-shot, ensures clean chip state
ctest --test-dir build         # all tests, serialised on chip
ctest --test-dir build -L unit # host-only unit tests
ctest --test-dir build -L hw   # Blackhole integration tests
```

HW tests share `RESOURCE_LOCK chip` so they never run concurrently
within a ctest invocation. `TT_FOIL_DEVICE` falls back to the env var
if unset on the CMake line, and to `0` otherwise.

### Firmware ELF selection

`tt-foil` resolves firmware ELFs in this order:

1. `$TT_FOIL_FIRMWARE_DIR` if set (must contain
   `brisc/brisc.elf`, `ncrisc/ncrisc.elf`, `trisc{0,1,2}/trisc{0,1,2}.elf`).
2. `$TT_FOIL_BUILD_FIRMWARE_DIR` if set, or the path baked in at CMake
   configure time when `TT_FOIL_BUILD_FIRMWARE=ON` (default) —
   `<tt-foil-build>/firmware/`. **This is the default source**: tt-foil
   compiles each firmware ELF directly from `tt_metal/hw/firmware/src/
   tt-1xx/{brisc,ncrisc,trisc}.cc` via SFPI g++, runs
   `tools/tt_foil_weaken` to produce the `*_weakened.elf` companion, and
   stores the result in this tree.
3. The newest matching dir under `$HOME/.cache/tt-metal-cache/<hash>/firmware/`
   (tt-metal JIT cache, when available).
4. Fallback: the newest matching dir under
   `$TT_METAL_RUNTIME_ROOT/tt_metal/pre-compiled/<hash>/`.

Kernel ↔ firmware ABI consistency matters: a kernel ELF and the firmware
it runs against must come from the same build (their `*_weakened.elf`
must match the firmware that actually loads onto the chip). The
`build_kernels.sh` scripts under `examples/*/` follow the same precedence
order, so by default they link against tt-foil's self-built
`build/firmware/`.

## Examples

| Example                                                    | What it shows                                              |
| ---------------------------------------------------------- | ---------------------------------------------------------- |
| [`examples/add_two_numbers/`](examples/add_two_numbers/)   | Single core, BRISC kernel reads two L1 words and adds them |
| [`examples/noc_passthrough/`](examples/noc_passthrough/)   | Two cores, BRISC producer/consumer via `noc_async_write` + busy-loop sync |
| [`examples/tile_copy/`](examples/tile_copy/)               | One core, 5 RISCs: BRISC reader → CB c_0 → TRISC `copy_tile` → CB c_16 → NCRISC writer |

Each example has a `build_kernels.sh` that produces the ELFs under
`prebuilt/`. Run the corresponding test under `tests/` to exercise it end to
end.

## API Reference

All in `namespace tt::foil`. Include `<tt_foil/runtime.hpp>`.

### Device

```cpp
// Open PCIe chip and cold-boot the requested Tensix cores.
std::shared_ptr<Device> open_device(
    int pcie_device_index = 0,
    const std::string& firmware_dir = "",
    std::vector<CoreCoord> cores = {{0, 0}});

void close_device(std::shared_ptr<Device> device);
```

### Buffer

```cpp
enum class BufferLocation { L1, DRAM };

std::shared_ptr<Buffer> allocate_buffer(
    Device& device, BufferLocation loc,
    std::size_t size_bytes,
    CoreCoord logical_core = {});  // L1: which core's L1; DRAM: ignored

void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes);
void read_buffer (Device& device, Buffer& buf, void* dst,        std::size_t bytes);
```

### Kernel

```cpp
struct RiscBinary {
    enum class RiscId {
        BRISC  = 0,  // data movement processor 0
        NCRISC = 1,  // data movement processor 1
        TRISC0 = 2,  // compute UNPACK
        TRISC1 = 3,  // compute MATH
        TRISC2 = 4,  // compute PACK
    };
    RiscId risc;
    std::string elf_path;
};

std::shared_ptr<Kernel> load_kernel(
    Device& device,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core);

void set_runtime_args(
    Device& device, Kernel& kernel, RiscBinary::RiscId risc,
    std::span<const uint32_t> args);
```

### Circular Buffers (v4)

Pre-compute the descriptor blob on the host; firmware reads it at
launch time and populates each RISC's `cb_interface[]`.

```cpp
// In src/cb_config.hpp (internal — pull in directly for now).
struct CbConfig {
    uint8_t  cb_index;     // 0..31; convention: c_0..c_7 inputs, c_16..c_23 outputs
    uint64_t fifo_addr;    // L1 byte address where the ring lives
    uint32_t fifo_size;    // total ring size in bytes (= num_pages * page_size)
    uint32_t num_pages;    // tile-slot count
    uint32_t page_size;    // bytes per slot
};

void register_cbs(
    Device& device, Kernel& kernel,
    std::span<const CbConfig> cbs);
```

`register_cbs` allocates the blob in the kernel-config region for that
core, writes it to L1, and stores the resulting `CbAllocation` on the
kernel. `execute()` wires `local_cb_offset` / `local_cb_mask` into the
launch_msg from there.

### Execution

```cpp
// Single kernel.
void execute(Device& device, Kernel& kernel);

// Multi-kernel — all GOs are fired before any DONE check, so
// producer/consumer kernels actually meet on the device.
void execute(Device& device, std::initializer_list<Kernel*> kernels);
```

### NOC unicast addressing (multi-core)

```cpp
// Pack a 64-bit NOC unicast destination address for a noc_async_write
// targeting another core's L1. Pass to the producer kernel as two
// 32-bit runtime args (lo, hi).
uint64_t make_noc_unicast_addr(
    Device& device,
    CoreCoord logical_dst,
    uint64_t local_l1_addr);
```

## Architecture

```
tt-foil/
├── include/tt_foil/runtime.hpp     # Public API
├── src/
│   ├── device.{hpp,cpp}            # umd::Cluster + Hal ownership, multi-core cold boot
│   ├── buffer.{hpp,cpp}            # Bump allocators, UMD-direct read/write
│   ├── kernel.{hpp,cpp}            # ELF parsing (XIP), per-RISC RTA staging
│   ├── dispatch.{hpp,cpp}          # Slow-dispatch: reset → setup → fire_go → wait_done
│   ├── cb_config.{hpp,cpp}         # CB descriptor blob layout + L1 write (v4)
│   ├── firmware_load.{hpp,cpp}     # Boot firmware ELF → L1 (per-RISC reloc)
│   ├── firmware_paths.{hpp,cpp}    # Cache > pre-compiled discovery + env override
│   ├── mailbox_init.{hpp,cpp}      # LAUNCH/GO_MSG/rd_ptr/go_idx boot writes
│   ├── core_info_init.{hpp,cpp}    # CORE_INFO mailbox minimal init
│   ├── bank_tables_init.{hpp,cpp}  # BANK_TO_NOC + LOGICAL_TO_VIRTUAL zero-fill
│   ├── reset.{hpp,cpp}             # assert/deassert + INIT mailbox poll
│   ├── noc_addr.{hpp,cpp}          # Host-side NOC unicast addr packing
│   ├── umd_boot.{hpp,cpp}          # UMD Cluster direct open helper
│   ├── runtime_impl.cpp            # Public API thin delegation
│   └── llrt_local/                 # ll_api::memory + tt_elffile + HAL stubs
│       └── (vendored from tt_metal/llrt; ll_api namespace renamed)
└── (no submodule — tt-metal is a sibling directory, located via CMake var)
```

The static library `tt_foil` links a separate `tt_foil_hal_local` static, which
compiles `tt_metal/llrt/hal.cpp` + the Blackhole HAL .cpp files straight from
the tt-metal source tree. `libtt_metal.so` itself is not linked at runtime.

### Cold-boot sequence (`device_open`)

Per requested logical core, in two passes:

```
Pass 1: assert all RISCs reset
        load brisc.elf, ncrisc.elf, trisc0/1/2.elf to fw_base_addr
        zero-fill BANK_TO_NOC_SCRATCH + LOGICAL_TO_VIRTUAL_SCRATCH
        write minimal CORE_INFO (abs_logical_x/y + magic=WORKER)
        write LAUNCH (8 × zero launch_msg_t), GO_MSG.signal=RUN_MSG_INIT, ptrs=0
        deassert BRISC reset

Pass 2: poll GO_MSG.signal until RUN_MSG_DONE per core
        (BRISC firmware brings NCRISC + TRISCs up via subordinate_sync)
```

### Dispatch protocol (`execute`)

```
Stage 0: for each kernel: send RUN_MSG_RESET_READ_PTR_FROM_HOST + zero GO_MSG_INDEX
         (matches tt-metal slow-dispatch send_reset_go_signal)
l1_membar()

Stage 1: for each kernel:
           write kernel ELF (XIP) to kernel_text_addr in KERNEL_CONFIG region
           write runtime args to per-RISC RTA slots
           build + write launch_msg (DISPATCH_MODE_HOST, CB fields from cb_alloc)
sfence()
l1_membar()

Stage 2: for each kernel: write GO_MSG.signal = RUN_MSG_GO
l1_membar()

Stage 3: for each kernel: poll GO_MSG.signal until RUN_MSG_DONE
```

## Out of scope (vs v5 and beyond)

- DRAM interleaved / sharded buffers
- `get_noc_addr_from_logical_xy` (worker_logical_to_virtual table is zero-fill)
- Multicast NOC writes
- Multi-chip / mesh
- Fast dispatch (dispatch firmware)
- Wormhole / Quasar (Blackhole-only by design)
- JIT kernel compilation

## License

Apache-2.0
