# tt-foil

A lightweight C++ runtime for running pre-compiled kernels on a single Blackhole Tensix core. Designed for embedding into larger systems where the full tt-metal stack is too heavy.

## Why

The standard tt-metal runtime is ~660K lines and includes JIT compilation, dispatch firmware, profiling, multi-device mesh support, and Python bindings. For embedding use cases, most of this is unnecessary. tt-foil provides only what is needed to load a pre-compiled kernel and run it.

| Feature | tt-metal | tt-foil |
|---|---|---|
| Codebase size | ~660K lines | ~1.5K lines |
| JIT kernel compilation | Yes | No (pre-compiled only) |
| Dispatch firmware | Yes | No (slow-dispatch via direct mailbox write) |
| Multi-device / Mesh | Yes | No (single chip) |
| Python bindings | Yes | No |
| Target hardware | WH, BH, Quasar | Blackhole only (v1) |

## Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+ or Clang 14+)
- Blackhole PCIe device
- Pre-built tt-metal management firmware binaries
- Pre-compiled RISC-V kernel ELF(s)

## Getting Started

### 1. Clone with submodule

```bash
git clone https://github.com/tenstorrent/tt-foil.git
cd tt-foil
git submodule update --init --recursive third_party/tt-metal/tt_metal/third_party/umd
```

> Note: Only the UMD portion of the submodule needs to be initialized for the build.

### 2. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 3. Prepare firmware

tt-foil does not compile firmware. Build tt-metal once to produce the firmware ELFs:

```bash
# In the tt-metal repo
cmake -B build && cmake --build build --target firmware
```

The firmware files needed are: `brisc.elf`, `ncrisc.elf`, `trisc0.elf`, `trisc1.elf`, `trisc2.elf`.

### 4. Pre-compile your kernel

Kernels run on the BRISC or NCRISC RISC-V processors inside Tensix. Compile with the tt-metal device-side toolchain:

```bash
riscv32-unknown-elf-g++ \
  -march=rv32imc -mabi=ilp32 -O2 \
  -fno-exceptions -fno-rtti \
  -I third_party/tt-metal/tt_metal/hw/inc \
  -I third_party/tt-metal/tt_metal/hw/inc/hostdev \
  my_kernel.cpp -o my_kernel.elf
```

See [`examples/add_two_numbers/kernels/add_brisc.cpp`](examples/add_two_numbers/kernels/add_brisc.cpp) for a minimal example.

### 5. Run

```cpp
#include "tt_foil/runtime.hpp"

auto dev = tt::foil::open_device(0, "/path/to/firmware");

tt::foil::CoreCoord core{0, 0};
auto buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);

uint32_t value = 42;
tt::foil::write_buffer(*dev, *buf, &value, 4);

std::array<tt::foil::RiscBinary, 1> bins = {{
    {tt::foil::RiscBinary::RiscId::BRISC, "my_kernel.elf"},
}};
auto kernel = tt::foil::load_kernel(*dev, bins, core);

std::vector<uint32_t> args = {static_cast<uint32_t>(buf->device_addr)};
tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::BRISC, args);

tt::foil::execute(*dev, *kernel);

uint32_t result = 0;
tt::foil::read_buffer(*dev, *buf, &result, 4);

tt::foil::close_device(std::move(dev));
```

## API Reference

All functions are in `namespace tt::foil`. Include `<tt_foil/runtime.hpp>`.

### Device

```cpp
// Open PCIe device. firmware_dir contains brisc.elf, ncrisc.elf, trisc*.elf.
std::unique_ptr<Device> open_device(int pcie_device_index = 0,
                                    const std::string& firmware_dir = "");

void close_device(std::unique_ptr<Device> device);
```

### Buffer

```cpp
enum class BufferLocation { L1, DRAM };

std::unique_ptr<Buffer> allocate_buffer(Device& device, BufferLocation loc,
                                        std::size_t size_bytes,
                                        CoreCoord logical_core = {});
void free_buffer(std::unique_ptr<Buffer> buffer);

void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes);
void read_buffer(Device& device, Buffer& buf, void* dst, std::size_t bytes);
```

### Kernel

```cpp
struct RiscBinary {
    enum class RiscId { BRISC = 0, NCRISC = 1 };
    RiscId risc;
    std::string elf_path;
};

// Load pre-compiled ELFs from disk (all disk I/O happens here).
std::unique_ptr<Kernel> load_kernel(Device& device,
                                    std::span<const RiscBinary> binaries,
                                    CoreCoord logical_core);

// Set 32-bit runtime args for a RISC. Max 64 words per RISC.
void set_runtime_args(Device& device, Kernel& kernel,
                      RiscBinary::RiscId risc,
                      std::span<const uint32_t> args);
```

### Execution

```cpp
// Blocking execution. Throws std::runtime_error on timeout (default 5 s).
void execute(Device& device, Kernel& kernel);
```

## Architecture

```
tt-foil/
├── include/tt_foil/runtime.hpp   # Public API
├── src/
│   ├── device.cpp                # UMD Cluster + HAL init, firmware loading
│   ├── buffer.cpp                # Bump allocator, DMA transfer wrappers
│   ├── kernel.cpp                # ELF loading (XIP), runtime arg storage
│   ├── dispatch.cpp              # Slow-dispatch: ELF write + mailbox + poll
│   └── runtime_impl.cpp          # Public API delegation
└── third_party/tt-metal/         # git submodule (UMD, HAL, llrt subset only)
```

**Dispatch protocol (DISPATCH_MODE_HOST):**
1. Write kernel ELF binary to `fw_base_addr` in core L1
2. Write runtime args to RTA region in L1
3. Write `launch_msg` with `mode = DISPATCH_MODE_HOST`
4. Write `go_msg.signal = RUN_MSG_GO`
5. Poll `go_msg.signal` until `RUN_MSG_DONE`

## Scope (v1)

**Supported:**
- Single Blackhole chip (one PCIe device)
- Single Tensix core target per kernel
- BRISC and NCRISC data movement kernels
- L1 and DRAM buffer allocation
- Up to 64 × 32-bit runtime arguments per RISC

**Not supported (future work):**
- Circular buffers → required for Compute (TRISC) kernels
- Multi-core kernels
- Multi-chip / Mesh
- Subset of Tensix cores (pseudo-embedding mode)
- Fast dispatch (dispatch firmware)
- JIT kernel compilation

## License

Apache-2.0
