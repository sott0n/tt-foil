# tt-foil Codebase Guide

## What This Is

tt-foil is a lightweight C++ runtime for running pre-compiled RISC-V kernels on a single Blackhole Tensix core. It is a standalone repository that references tt-metal as a git submodule and uses only a small subset of it (UMD, HAL, llrt).

## Repository Layout

```
tt-foil/
├── include/tt_foil/runtime.hpp   # The only public header. All user-facing API lives here.
├── src/
│   ├── device.hpp/.cpp           # Device struct, UMD/HAL init, firmware load, DMA helpers
│   ├── buffer.hpp/.cpp           # Buffer struct, BumpAllocator, write_l1/write_dram
│   ├── kernel.hpp/.cpp           # Kernel struct, ELF loading, runtime arg storage
│   ├── dispatch.hpp/.cpp         # execute() — slow-dispatch protocol
│   └── runtime_impl.cpp          # Thin delegation from public API → internal modules
├── tests/
│   ├── test_bump_alloc.cpp       # No hardware required
│   ├── test_elf_load.cpp         # No hardware required (set TT_FOIL_FW_DIR)
│   └── test_add_kernel.cpp       # Requires Blackhole hardware
└── examples/add_two_numbers/     # End-to-end example with device-side kernel source
```

## Building

```bash
git submodule update --init --recursive third_party/tt-metal/tt_metal/third_party/umd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Hardware integration tests (needs a Blackhole chip):
cmake -B build -DTT_FOIL_HW_TESTS=ON
```

## Key Design Decisions

### No MetalContext
tt-metal uses a global singleton `MetalContext::instance()` to access the `Hal` and `Cluster` objects. tt-foil avoids this entirely — `Device` owns both directly and passes them by reference through every call.

### No JIT
`jit_build/` is never linked. Kernels must be pre-compiled RISC-V ELFs. The `ll_api::memory` class (from `llrt/tt_memory.h`) handles ELF parsing and XIP transformation.

### Slow-dispatch only
No dispatch firmware runs. The host writes the kernel ELF to `fw_base_addr` in L1, populates `launch_msg` with `DISPATCH_MODE_HOST`, fires `go_msg = RUN_MSG_GO`, and polls until `RUN_MSG_DONE`. This is in `src/dispatch.cpp`.

### Bump allocator
`L1Allocator` and `DramAllocator` are simple bump-pointer allocators (no free-list). L1 allocations start at `MEM_MAP_END + 69KB` (above firmware, mailbox, and kernel config buffer). DRAM allocations start at `HalDramMemAddrType::UNRESERVED`.

### Only BRISC and NCRISC (v1)
Circular buffers are required for Compute kernels (TRISC0/1/2). v1 does not support CBs, so only data movement kernels on BRISC and NCRISC are valid targets.

## tt-metal Submodule Usage

`cmake/tt_metal_deps.cmake` selects only these files from the submodule:

| Component | Files used |
|---|---|
| UMD | `tt_metal/third_party/umd/` (full, required for PCIe) |
| HAL | `tt_metal/llrt/hal/tt-1xx/blackhole/` (Blackhole memory maps) |
| llrt subset | `tt_elffile.cpp`, `tt_memory.cpp`, `tt_cluster.cpp`, `hal.cpp`, `rtoptions.cpp`, `tlb_config.cpp`, `metal_soc_descriptor.cpp`, `core_descriptor.cpp` |

Everything else in tt-metal (`impl/`, `jit_build/`, `distributed/`, `fabric/`, Python bindings) is not linked.

## Important Constants (Blackhole)

Defined in `third_party/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/`:

- `MEM_L1_SIZE` = 1536 KB — total Tensix L1 size
- `MEM_MAILBOX_BASE` = 96 — start of the mailbox region
- `MEM_MAILBOX_SIZE` = 12912 bytes
- `MEM_MAP_END` = end of the firmware + mailbox reserved area
- `MaxProcessorsPerCoreType` = 5 (BRISC=0, NCRISC=1, TRISC0=2, TRISC1=3, TRISC2=4)

From `dev_msgs.h`:
- `RUN_MSG_GO` = 0x80 — signal to start kernel
- `RUN_MSG_DONE` = 0x00 — kernel has finished
- `DISPATCH_MODE_HOST` — host controls execution directly (no dispatch firmware)

## How Dispatch Works (dispatch.cpp)

```
execute(dev, kernel):
  1. for each loaded RISC:
       write kernel ELF to fw_base_addr in core L1
         (via mem.process_spans + hal.relocate_dev_addr + cluster.write_core)
  2. write runtime args to rta_base_addr in core L1
  3. build launch_msg:
       kernel_config.mode = DISPATCH_MODE_HOST
       kernel_config.enables = bitmask of active processors
       kernel_config.kernel_config_base[TENSIX] = rta_base_addr
       kernel_config.rta_offset[proc_idx] = proc_idx * 64 * 4
  4. cluster.write_core_immediate(launch_msg → LAUNCH mailbox addr)
  5. sfence()
  6. cluster.write_core_immediate(go_msg{RUN_MSG_GO} → GO_MSG addr)
  7. cluster.l1_barrier(chip)
  8. poll go_msg until signal == RUN_MSG_DONE (timeout 5s)
```

## Firmware Loading (device.cpp)

`device_open()` calls `load_firmware()` which:
1. Iterates 5 RISC types (BRISC, NCRISC, TRISC0/1/2)
2. For each: reads `<firmware_dir>/<risc>.elf` into `ll_api::memory` with CONTIGUOUS_XIP
3. Writes to all Tensix cores via `mem.process_spans + cluster.write_core`
4. Programs the launch address register for non-BRISC RISCs

Then `deassert_risc_reset()` is called and `wait_for_fw_init()` polls `go_msg` until the firmware confirms INIT-done.

## Common Tasks

### Add a new runtime API function
1. Declare in `include/tt_foil/runtime.hpp`
2. Implement in the relevant `src/*.cpp`
3. Add the delegation call in `src/runtime_impl.cpp`

### Add support for TRISC (Compute kernels)
Requires circular buffer (CB) setup in the `launch_msg`. The `local_cb_offset`, `local_cb_mask` fields in `kernel_config_msg_t` must be populated, and the CB layout written to L1 before firing `go_msg`.

### Change the RTA region size
`kMaxRtaWords` in `src/kernel.hpp` controls the per-RISC arg limit (currently 64 × 4 bytes = 256 bytes per RISC). Changing this also changes `rta_region_size` and the offsets written into `launch_msg.kernel_config.rta_offset`.
