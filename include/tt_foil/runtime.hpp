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

// Blocking host -> device write.
void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes);

// Blocking device -> host read.
void read_buffer(Device& device, Buffer& buf, void* dst, std::size_t bytes);

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
