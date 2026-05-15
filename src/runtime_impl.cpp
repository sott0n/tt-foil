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

std::shared_ptr<Device> open_device(
    int pcie_device_index,
    const std::string& firmware_dir,
    std::vector<CoreCoord> cores) {
    return std::shared_ptr<Device>(
        device_open(pcie_device_index, firmware_dir, std::move(cores)).release(),
        [](Device* d) { device_close(*d); delete d; });
}

void close_device(std::shared_ptr<Device> device) {
    if (device) {
        device_close(*device);
        device.reset();
    }
}

// ---- Buffer ----

std::shared_ptr<Buffer> allocate_buffer(
    Device& device, BufferLocation loc, std::size_t size_bytes, CoreCoord logical_core)
{
    Buffer* raw = buffer_alloc(device, loc, size_bytes, logical_core);
    return std::shared_ptr<Buffer>(raw, [](Buffer* b) { buffer_free(b); });
}

void free_buffer(std::shared_ptr<Buffer> buffer) {
    (void)buffer;  // shared_ptr destructor calls buffer_free when refcount hits 0
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

std::shared_ptr<Kernel> load_kernel(
    Device& device,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core)
{
    Kernel* raw = kernel_load(device, binaries, logical_core);
    return std::shared_ptr<Kernel>(raw, [](Kernel* k) { delete k; });
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

void execute(Device& device, std::initializer_list<Kernel*> kernels) {
    std::vector<Kernel*> v(kernels.begin(), kernels.end());
    dispatch_execute_multi(device, std::span<Kernel* const>(v.data(), v.size()));
}

}  // namespace tt::foil
