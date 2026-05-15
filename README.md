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
| Codebase size                 | ~660K lines                | ~2K lines (incl. bundled llrt subset)  |
| JIT kernel compilation        | Yes                        | No — pre-compiled ELFs only            |
| Dispatch firmware             | Yes                        | No — slow-dispatch via direct mailbox  |
| `libtt_metal.so` link         | Required                   | **Not linked**                         |
| `MetalContext` / IDevice      | Required                   | **Not used** — own UMD + HAL directly  |
| Multi-device / Mesh           | Yes                        | No — single chip                       |
| Multi-Tensix core dispatch    | Yes                        | Yes (v3)                               |
| NOC inter-core (`noc_async_*`)| Yes                        | Yes (v3, unicast)                      |
| TRISC compute + CB            | Yes                        | No                                     |
| Target hardware               | WH, BH, Quasar             | Blackhole only                         |
| Runtime dynamic deps          | `libtt_metal.so`, UMD, ... | UMD only (`libtt-umd.so`)              |

## Status

| Phase      | Highlight                                                     |
| ---------- | ------------------------------------------------------------- |
| v1         | Single BRISC kernel on one Tensix core, slow-dispatch         |
| v2 (A)     | BRISC + NCRISC concurrent on the same core                    |
| v2 (B1)    | I/O switched to UMD direct (`tt::Cluster` calls dropped)      |
| v2 (B2)    | `CreateDevice` replaced with in-tree cold-boot sequence       |
| v2 (B3)    | `libtt_metal.so` link removed — HAL compiled into tt-foil     |
| v3         | Multi-Tensix boot, multi-kernel dispatch, NOC inter-core L1↔L1|

Runs on Blackhole (Device=3 verified). End-to-end tests pass with only
`libtt-umd.so` linked.

## Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+ or Clang 14+)
- Blackhole PCIe device
- A built tt-metal source tree on disk (for headers + pre-compiled firmware
  ELFs + the SFPI cross-compiler — tt-foil bundles the .cpp it needs but the
  tree is still the source of headers and firmware artifacts)

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
- `tt_metal/pre-compiled/<hash>/{brisc,ncrisc,trisc0,trisc1,trisc2}/*.elf`
  (boot firmware)
- `build_Release/libexec/tt-metalium/runtime/sfpi/compiler/bin/riscv-tt-elf-g++`
  (kernel cross-compiler)

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
[`examples/add_two_numbers/build_kernels.sh`](examples/add_two_numbers/build_kernels.sh)
for a working invocation that links against firmware-weakened ELFs and the
appropriate linker script.

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
| `TT_FOIL_FIRMWARE_DIR`   | Explicit `pre-compiled/<hash>/` dir; overrides auto-find |
| `TT_FOIL_KERNEL_DIR`     | Directory containing your kernel ELFs (test convention)  |

## Examples

| Example                                                    | What it shows                                              |
| ---------------------------------------------------------- | ---------------------------------------------------------- |
| [`examples/add_two_numbers/`](examples/add_two_numbers/)   | Single core, BRISC kernel reads two L1 words and adds them |
| [`examples/noc_passthrough/`](examples/noc_passthrough/)   | Two cores, BRISC producer/consumer via `noc_async_write` + busy-loop sync |

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
    enum class RiscId { BRISC = 0, NCRISC = 1 };
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
│   ├── dispatch.{hpp,cpp}          # Slow-dispatch: setup → fire_go → wait_done
│   ├── firmware_load.{hpp,cpp}     # Boot firmware ELF → L1 (per-RISC reloc)
│   ├── firmware_paths.{hpp,cpp}    # pre-compiled/<hash>/ discovery + override
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
Stage 1: for each kernel:
           write kernel ELF (XIP) to kernel_text_addr in KERNEL_CONFIG region
           write runtime args to per-RISC RTA slots
           build + write launch_msg (DISPATCH_MODE_HOST)
sfence()

Stage 2: for each kernel: write GO_MSG.signal = RUN_MSG_GO
l1_membar()

Stage 3: for each kernel: poll GO_MSG.signal until RUN_MSG_DONE
```

## Out of scope (vs v4 and beyond)

- TRISC (compute) kernels and Circular Buffers
- DRAM interleaved / sharded buffers
- `get_noc_addr_from_logical_xy` (worker_logical_to_virtual table is zero-fill)
- Multicast NOC writes
- Multi-chip / mesh
- Fast dispatch (dispatch firmware)
- Wormhole / Quasar (Blackhole-only by design)
- JIT kernel compilation

## License

Apache-2.0
