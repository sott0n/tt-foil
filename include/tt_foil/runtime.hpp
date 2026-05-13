// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

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

// Open the Nth PCIe Blackhole chip (0-indexed).
// firmware_dir: directory containing pre-built management firmware ELFs
//   (brisc.elf, ncrisc.elf, trisc0.elf, trisc1.elf, trisc2.elf).
// Throws std::runtime_error on failure.
std::shared_ptr<Device> open_device(
    int pcie_device_index = 0,
    const std::string& firmware_dir = "");

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

}  // namespace tt::foil
