# tt-foil API Reference

All public symbols live in `namespace tt::foil`. Single header:

```cpp
#include <tt_foil/runtime.hpp>
```

This file is the canonical reference; the source of truth is
[`include/tt_foil/runtime.hpp`](../include/tt_foil/runtime.hpp).

## Device

```cpp
// Open PCIe chip and cold-boot the requested Tensix cores.
std::shared_ptr<Device> open_device(
    int pcie_device_index = 0,
    const std::string& firmware_dir = "",
    std::vector<CoreCoord> cores = {{0, 0}});

void close_device(std::shared_ptr<Device> device);
```

`firmware_dir` is optional; when empty, the resolver picks
`<build>/firmware/` from CMake, falling back to the JIT cache and then
the pre-compiled tree (see [Firmware ELF
selection](../README.md#firmware-elf-selection)).

## Buffer

```cpp
enum class BufferLocation { L1, DRAM };

std::shared_ptr<Buffer> allocate_buffer(
    Device& device, BufferLocation loc,
    std::size_t size_bytes,
    CoreCoord logical_core = {});  // L1: which core's L1; DRAM: ignored

void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes);
void read_buffer (Device& device, Buffer& buf, void* dst,        std::size_t bytes);
```

Bump-allocated per-core L1 (no free); reset by destroying the `Device`.
DRAM uses channel 0 with a single shared bump allocator.

## Kernel

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

A `Kernel` is the bundle of per-RISC ELFs that run on one Tensix core.
`load_kernel` parses each ELF (XIP, no relocation) and writes the text
to the per-RISC `kernel_text_addr` in the KERNEL_CONFIG L1 region.

## Circular Buffers

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

## Execution

```cpp
// Single kernel.
void execute(Device& device, Kernel& kernel);

// Multi-kernel — all GOs are fired before any DONE check, so
// producer/consumer kernels actually meet on the device.
void execute(Device& device, std::initializer_list<Kernel*> kernels);
```

The slow-dispatch protocol has four stages: reset → setup → fire_go →
wait_done. All kernels' GOs are written before any DONE poll so
multi-core kernels run concurrently on the device.

## NOC unicast addressing (multi-core)

```cpp
// Pack a 64-bit NOC unicast destination address for noc_async_read /
// noc_async_write targeting another core's L1. Pass to the kernel via
// two 32-bit runtime args (lo, hi).
uint64_t make_noc_unicast_addr(
    Device& device,
    CoreCoord logical_dst,
    uint64_t local_l1_addr);

// Pack a 64-bit NOC address for DRAM channel 0 (handles the per-bank
// coord + offset internally).
uint64_t make_noc_dram_addr(Device& device, uint64_t dram_offset);
```

> **BRISC writes:** on Blackhole, `noc_async_write` from BRISC to peer
> Tensix L1 must use `noc=1` explicitly (NOC 0 is the read direction
> and stalls in `noc_async_write_barrier`). Reads stay on NOC 0. See
> the matmul_2core_mcast example for the working pattern.

## Source layout

```
tt-foil/
├── include/tt_foil/runtime.hpp     # Public API (only header users include)
├── src/
│   ├── device.{hpp,cpp}            # umd::Cluster + Hal ownership, multi-core boot
│   ├── buffer.{hpp,cpp}            # Bump allocators, UMD direct read/write
│   ├── kernel.{hpp,cpp}            # ELF parsing, per-RISC RTA staging
│   ├── dispatch.{hpp,cpp}          # Slow-dispatch: reset → setup → fire_go → wait
│   ├── cb_config.{hpp,cpp}         # CB descriptor blob layout + L1 write
│   ├── firmware_load.{hpp,cpp}     # Boot firmware ELF → L1
│   ├── firmware_paths.{hpp,cpp}    # build/firmware > cache > pre-compiled discovery
│   ├── mailbox_init.{hpp,cpp}      # LAUNCH/GO_MSG/rd_ptr/go_idx boot writes
│   ├── core_info_init.{hpp,cpp}    # CORE_INFO mailbox minimal init
│   ├── bank_tables_init.{hpp,cpp}  # BANK_TO_NOC + LOGICAL_TO_VIRTUAL zero-fill
│   ├── reset.{hpp,cpp}             # assert/deassert + INIT mailbox poll
│   ├── noc_addr.{hpp,cpp}          # Host-side NOC address packing
│   ├── umd_boot.{hpp,cpp}          # UMD Cluster direct open helper
│   ├── runtime_impl.cpp            # Public API thin delegation
│   └── llrt_local/                 # Vendored ll_api::memory + tt_elffile
├── tools/tt_foil_weaken.cpp        # Host CLI: firmware ELF → *_weakened.elf
├── scripts/build_firmware.sh       # Builds the 5 RISC firmware ELFs
└── (no submodule — tt-metal is a sibling directory, located via CMake var)
```
