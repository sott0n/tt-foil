// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace tt::foil {

// ---------------------------------------------------------------------------
// Forward declarations (opaque handles — callers never touch internals)
// ---------------------------------------------------------------------------
struct Device;
struct Buffer;
struct Kernel;

// ---------------------------------------------------------------------------
// Core coordinate (logical, Tensix grid)
// ---------------------------------------------------------------------------
struct CoreCoord {
    uint32_t x{0};
    uint32_t y{0};
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

// Open the Nth PCIe Blackhole chip (0-indexed).
// firmware_dir: directory containing pre-built management firmware ELFs
//   (brisc.elf, ncrisc.elf, trisc0.elf, trisc1.elf, trisc2.elf).
//   If empty, searches the default build output path.
// Throws std::runtime_error on failure.
std::unique_ptr<Device> open_device(
    int pcie_device_index = 0,
    const std::string& firmware_dir = "");

// Close the device, assert resets on all cores, tear down UMD.
void close_device(std::unique_ptr<Device> device);

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

enum class BufferLocation {
    L1,    // per-core local SRAM
    DRAM,  // off-chip DRAM channel 0
};

// Allocate a contiguous region of device memory via a bump allocator.
// For L1, logical_core identifies which core's L1 to use.
// For DRAM, logical_core is ignored.
// Throws std::runtime_error if out of memory.
std::unique_ptr<Buffer> allocate_buffer(
    Device& device,
    BufferLocation loc,
    std::size_t size_bytes,
    CoreCoord logical_core = {});

// Release a buffer.
// Simple bump allocator: only the most recent allocation is actually freed.
void free_buffer(std::unique_ptr<Buffer> buffer);

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
// The ELFs are parsed and XIP-transformed at this point; all disk I/O happens
// inside load_kernel().
// v1: single core target only.
std::unique_ptr<Kernel> load_kernel(
    Device& device,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core);

// Write runtime arguments for a specific RISC processor.
// Must be called after load_kernel() and before execute().
// Args are 32-bit words written to the kernel's RTA region in L1.
void set_runtime_args(
    Device& device,
    Kernel& kernel,
    RiscBinary::RiscId risc,
    std::span<const uint32_t> args);

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// Blocking execution:
//   1. Write loaded ELF binaries to core L1
//   2. Write launch_msg (DISPATCH_MODE_HOST)
//   3. Write runtime args to the RTA region
//   4. Fire go_msg (RUN_MSG_GO)
//   5. Poll go_msg until RUN_MSG_DONE
// Throws std::runtime_error on timeout or hardware error.
void execute(Device& device, Kernel& kernel);

}  // namespace tt::foil
