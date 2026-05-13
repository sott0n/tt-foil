// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implements the public API declared in include/tt_foil/runtime.hpp
// by delegating to the internal device/buffer/kernel/dispatch modules.

#include "tt_foil/runtime.hpp"

#include "device.hpp"
#include "buffer.hpp"
#include "kernel.hpp"
#include "dispatch.hpp"

#include <stdexcept>

namespace tt::foil {

// ---- Device ----

std::unique_ptr<Device> open_device(int pcie_device_index, const std::string& firmware_dir) {
    return device_open(pcie_device_index, firmware_dir);
}

void close_device(std::unique_ptr<Device> device) {
    if (device) {
        device_close(*device);
    }
}

// ---- Buffer ----

std::unique_ptr<Buffer> allocate_buffer(
    Device& device, BufferLocation loc, std::size_t size_bytes, CoreCoord logical_core)
{
    Buffer* raw = buffer_alloc(device, loc, size_bytes, logical_core);
    return std::unique_ptr<Buffer>(raw);
}

void free_buffer(std::unique_ptr<Buffer> buffer) {
    // unique_ptr destructor calls buffer_free via custom deleter — but Buffer's
    // destructor is trivial (bump allocator; no actual memory reclaim).
    // Explicitly do nothing: the unique_ptr going out of scope handles cleanup.
    (void)buffer;
}

void write_buffer(Device& device, Buffer& buf, const void* src, std::size_t bytes) {
    if (bytes > buf.size_bytes) {
        throw std::runtime_error("tt-foil: write_buffer size exceeds allocation");
    }
    switch (buf.location) {
        case BufferLocation::L1:
            write_l1(device, buf.core, buf.device_addr, src, bytes);
            break;
        case BufferLocation::DRAM:
            write_dram(device, buf.device_addr, src, bytes);
            break;
    }
}

void read_buffer(Device& device, Buffer& buf, void* dst, std::size_t bytes) {
    if (bytes > buf.size_bytes) {
        throw std::runtime_error("tt-foil: read_buffer size exceeds allocation");
    }
    switch (buf.location) {
        case BufferLocation::L1:
            read_l1(device, buf.core, buf.device_addr, dst, bytes);
            break;
        case BufferLocation::DRAM:
            read_dram(device, buf.device_addr, dst, bytes);
            break;
    }
}

// ---- Kernel ----

std::unique_ptr<Kernel> load_kernel(
    Device& device,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core)
{
    Kernel* raw = kernel_load(device, binaries, logical_core);
    return std::unique_ptr<Kernel>(raw);
}

void set_runtime_args(
    Device& /*device*/,
    Kernel& kernel,
    RiscBinary::RiscId risc,
    std::span<const uint32_t> args)
{
    kernel_set_runtime_args(kernel, risc, args);
}

// ---- Execution ----

void execute(Device& device, Kernel& kernel) {
    dispatch_execute(device, kernel);
}

}  // namespace tt::foil
