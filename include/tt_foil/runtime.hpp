// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace tt::foil {

// ---------------------------------------------------------------------------
// Opaque forward declarations
// ---------------------------------------------------------------------------
struct Device;
struct Kernel;

// ---------------------------------------------------------------------------
// Core coordinate (logical, Tensix grid)
// ---------------------------------------------------------------------------
struct CoreCoord {
    uint32_t x{0};
    uint32_t y{0};
    bool operator==(const CoreCoord& o) const { return x == o.x && y == o.y; }
};

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

enum class BufferLocation {
    L1,    // per-core local SRAM
    DRAM,  // off-chip DRAM channel 0
};

// Buffer: device memory allocation.
// device_addr is the NOC-visible address; pass it as a kernel runtime arg.
struct Buffer {
    BufferLocation location;
    uint64_t       device_addr{0};
    std::size_t    size_bytes{0};
    CoreCoord      core;  // for L1 buffers; unused for DRAM
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

// Open the Nth PCIe Blackhole chip (0-indexed) and cold-boot the requested
// Tensix cores. `cores` defaults to a single core at logical (0,0) for
// backwards compatibility with v1/v2 callers. Pass additional CoreCoord
// entries (e.g. {{0,0}, {0,1}}) to boot multiple cores at open time —
// kernels can only be loaded on cores listed here.
//
// firmware_dir: directory containing pre-built management firmware ELFs
//   (brisc.elf, ncrisc.elf, trisc0.elf, trisc1.elf, trisc2.elf).
// Throws std::runtime_error on failure (chip missing, firmware ELFs not
// found, INIT poll timeout).
std::shared_ptr<Device> open_device(
    int pcie_device_index = 0,
    const std::string& firmware_dir = "",
    std::vector<CoreCoord> cores = {{0, 0}});

// Close the device explicitly before the shared_ptr goes out of scope.
// Asserts resets on all cores and tears down UMD.
// The shared_ptr will call this automatically when the last reference drops.
void close_device(std::shared_ptr<Device> device);

// ---------------------------------------------------------------------------
// Buffer API
// ---------------------------------------------------------------------------

// Allocate a contiguous region of device memory via a bump allocator.
// For L1, logical_core identifies which core's L1 to use.
// Throws std::runtime_error if out of memory.
std::shared_ptr<Buffer> allocate_buffer(
    Device& device,
    BufferLocation loc,
    std::size_t size_bytes,
    CoreCoord logical_core = {});

// Release a buffer (bump allocator; only the most recent alloc is freed).
void free_buffer(std::shared_ptr<Buffer> buffer);

// Blocking host -> device write. Optional `label` is attached to the
// TF_write_buffer Tracy zone as ZoneText so the per-call CSV identifies
// which logical tensor a transfer corresponds to (no effect when Tracy
// is disabled — the label is read only inside the TF_ZONE block).
void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes,
                  const char* label = nullptr);

// Blocking host -> device write, at a byte offset within the buffer. Useful
// for partial updates of large buffers (e.g. a KV cache slot).
void write_buffer(Device& device, Buffer& buf, std::size_t offset_bytes,
                  const void* src, std::size_t bytes,
                  const char* label = nullptr);

// Blocking device -> host read. Optional `label` is attached to the
// TF_read_buffer Tracy zone (see write_buffer for semantics).
void read_buffer(Device& device, Buffer& buf, void* dst, std::size_t bytes,
                 const char* label = nullptr);

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

struct RiscBinary {
    enum class RiscId {
        BRISC  = 0,  // data movement processor 0
        NCRISC = 1,  // data movement processor 1
        TRISC0 = 2,  // compute UNPACK
        TRISC1 = 3,  // compute MATH
        TRISC2 = 4,  // compute PACK
    };
    RiscId risc;
    std::string elf_path;  // path to a pre-compiled RISC-V ELF
};

// Load ELF binaries from disk and prepare them for execution on logical_core.
// v1: single core target only.
std::shared_ptr<Kernel> load_kernel(
    Device& device,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core);

// Release all kernel-config state (RTAs + kernel text) for `logical_core`,
// so subsequent load_kernel() calls can reuse the per-core KERNEL_CONFIG
// region from scratch.
//
// The Blackhole KERNEL_CONFIG region is small (~69 KB on Tensix), and
// load_kernel is a bump allocator with no per-kernel free.  When a chain
// needs more distinct programs than fit at once, the caller is meant to
// drop all outstanding std::shared_ptr<Kernel> for `logical_core` and
// then call release_kernels(device, logical_core); the next load_kernel
// starts at the base of the region again.
//
// Behaviour after release:
//   • Any Kernel object whose shared_ptr is still alive is UB to use —
//     its rta_base_addr / kernel_text_addr now point at memory that the
//     next load_kernel will overwrite.  Drop them first.
//   • Buffers in DRAM and in the user L1 region are unaffected.
//   • The launch_msg slot the firmware polls is untouched.
void release_kernels(Device& device, CoreCoord logical_core);

// Pin a kernel so it survives release_kernels() / reset_l1() cycles.
//
// After building a persistent op (its L1 CBs allocated, kernel ELF
// loaded), call pin_persistent(dev, kernel, core) to:
//   • freeze the current L1 watermark — subsequent reset_l1() rewinds
//     to this point, not to base, so the op's CB-backing L1 stays valid;
//   • freeze the kernel_config watermark — subsequent release_kernels()
//     rewinds to this point, so the op's kernel text + RTA slots stay;
//   • add the kernel to the per-core pinned set — release_kernels()
//     skips it when clearing resident_kernels, so dispatch's ELF NOC
//     skip keeps hitting on the next dispatch.
//
// Caller must keep the std::shared_ptr<Kernel> alive and use the op's
// set_*_args() per call to update DRAM addresses + shape RTAs.
void pin_persistent(Device& device, const Kernel& kernel, CoreCoord logical_core);

// Reset the per-core L1 bump allocator: subsequent allocate_buffer(L1, …)
// calls reclaim the whole user L1 region from the base.
//
// The caller MUST ensure no live shared_ptr<Buffer> still references an L1
// address from this core (otherwise the next op writing to the reused
// region will overwrite live data the holder still expects). Typical
// pattern: run each op in its own scope so the Op handle (and its L1
// shared_ptrs) drops before reset_l1 is called. DRAM buffers and the
// per-core KERNEL_CONFIG region are untouched.
void reset_l1(Device& device, CoreCoord logical_core);

// Write runtime arguments for a specific RISC processor.
void set_runtime_args(
    Device& device,
    Kernel& kernel,
    RiscBinary::RiscId risc,
    std::span<const uint32_t> args);

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// Blocking execution: write ELF + args, fire go_msg, poll until DONE.
// Throws std::runtime_error on timeout.
void execute(Device& device, Kernel& kernel);

// Multi-kernel launch — every kernel's GO is fired before any DONE check,
// so producer/consumer kernels actually observe each other on the device.
// Throws std::runtime_error if any kernel times out.
void execute(Device& device, std::initializer_list<Kernel*> kernels);

// ---------------------------------------------------------------------------
// NOC unicast address helper (multi-core kernels)
// ---------------------------------------------------------------------------

// Pack a 64-bit NOC unicast destination address suitable for a device-side
// noc_async_write_one_packet() call. `logical_dst` is the logical CoreCoord
// (same coord space the host uses with allocate_buffer / load_kernel), and
// `local_l1_addr` is the byte offset into that core's L1.
//
// Blackhole layout: bits [47:42] = NOC y, [41:36] = NOC x, [35:0] = local.
// Pre-computing this on the host means producer/consumer kernels don't need
// to read the worker_logical_to_virtual scratch (which tt-foil zero-fills).
uint64_t make_noc_unicast_addr(
    Device& device,
    CoreCoord logical_dst,
    uint64_t local_l1_addr);

// Pack a 64-bit NOC address for DRAM channel 0. `dram_offset` is the
// offset returned by allocate_buffer(BufferLocation::DRAM). Kernels
// receive this 64-bit value via RTA (split hi/lo) and feed it to
// noc_async_read / noc_async_write.
uint64_t make_noc_dram_addr(Device& device, uint64_t dram_offset);

}  // namespace tt::foil
