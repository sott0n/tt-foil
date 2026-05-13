// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <stdexcept>

// tt-metal public API — delegates firmware init to tt-metal
#include "tt-metalium/host_api.hpp"   // CreateDevice, CloseDevice

// MetalContext — cluster and HAL accessors (available after CreateDevice)
#include "impl/context/metal_context.hpp"

// Full cluster and HAL headers (needed for member function calls in this file)
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"

// UMD types
#include <umd/device/types/xy_pair.hpp>

#include "tt_foil/runtime.hpp"  // for CoreCoord

namespace tt::foil {

using TtCoreCoord = tt::xy_pair;  // tt::tt_metal::CoreCoord = tt::xy_pair

// ---- Allocator helpers ----

uint64_t L1Allocator::alloc(std::size_t bytes, uint32_t alignment) {
    uint64_t aligned = (current + alignment - 1) & ~static_cast<uint64_t>(alignment - 1);
    if (aligned + bytes > end) {
        throw std::runtime_error("tt-foil: L1 out of memory");
    }
    current = aligned + bytes;
    return aligned;
}

void L1Allocator::reset() { current = base; }

uint64_t DramAllocator::alloc(std::size_t bytes, uint32_t alignment) {
    uint64_t aligned = (current + alignment - 1) & ~static_cast<uint64_t>(alignment - 1);
    if (aligned + bytes > end) {
        throw std::runtime_error("tt-foil: DRAM out of memory");
    }
    current = aligned + bytes;
    return aligned;
}

void DramAllocator::reset() { current = base; }

// ---- Device internals ----

L1Allocator& Device::kernel_config_for_core(const CoreCoord& logical_core) {
    uint64_t key = core_key(logical_core.x, logical_core.y);
    auto it = kernel_config_allocs.find(key);
    if (it == kernel_config_allocs.end()) {
        uint64_t base = hal->get_dev_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
        // Use DEFAULT_UNRESERVED start as the end of kernel config region.
        // (HAL forbids querying KERNEL_CONFIG size directly.)
        uint64_t end = hal->get_dev_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

        L1Allocator alloc{};
        alloc.base    = base;
        alloc.current = base;
        alloc.end     = end;
        kernel_config_allocs.emplace(key, alloc);
        return kernel_config_allocs.at(key);
    }
    return it->second;
}

L1Allocator& Device::l1_for_core(const CoreCoord& logical_core) {
    uint64_t key = core_key(logical_core.x, logical_core.y);
    auto it = l1_allocs.find(key);
    if (it == l1_allocs.end()) {
        uint64_t l1_user_base = hal->get_dev_addr(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint64_t l1_user_size = hal->get_dev_size(
            tt_metal::HalProgrammableCoreType::TENSIX,
            tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

        L1Allocator alloc{};
        alloc.base    = l1_user_base;
        alloc.current = l1_user_base;
        alloc.end     = l1_user_base + l1_user_size;
        l1_allocs.emplace(key, alloc);
        return l1_allocs.at(key);
    }
    return it->second;
}

// ---- Public device API ----

std::unique_ptr<Device> device_open(int pcie_device_index, const std::string& /*firmware_dir*/) {
    auto dev = std::make_unique<Device>();
    dev->chip_id = static_cast<uint32_t>(pcie_device_index);

    // Delegate device init (cluster setup, HAL, firmware load, wait for INIT)
    // entirely to tt-metal's CreateDevice.
    dev->tt_device = tt::tt_metal::CreateDevice(
        static_cast<tt::ChipId>(pcie_device_index));

    // Borrow cluster and HAL from MetalContext (valid while tt_device is alive).
    auto& ctx = tt::tt_metal::MetalContext::instance();
    dev->cluster = &ctx.get_cluster();
    dev->hal     = &ctx.hal();

    // Initialise DRAM bump allocator from HAL.
    uint64_t dram_base = dev->hal->get_dev_addr(tt_metal::HalDramMemAddrType::UNRESERVED);
    uint64_t dram_size = dev->hal->get_dev_size(tt_metal::HalDramMemAddrType::UNRESERVED);
    dev->dram_alloc = DramAllocator{dram_base, dram_base, dram_base + dram_size};

    return dev;
}

void device_close(Device& dev) {
    if (dev.tt_device) {
        tt::tt_metal::CloseDevice(dev.tt_device);
        dev.tt_device = nullptr;
        dev.cluster   = nullptr;
        dev.hal       = nullptr;
    }
}

tt::xy_pair logical_to_virtual(const Device& dev, const CoreCoord& logical) {
    tt::tt_metal::CoreCoord virt =
        dev.tt_device->worker_core_from_logical_core({logical.x, logical.y});
    return TtCoreCoord{virt.x, virt.y};
}

void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size) {
    TtCoreCoord virt = logical_to_virtual(dev, core);
    dev.cluster->write_core(src, static_cast<uint32_t>(size), tt_cxy_pair(dev.chip_id, virt), addr);
}

void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size) {
    TtCoreCoord virt = logical_to_virtual(dev, core);
    dev.cluster->read_core(dst, static_cast<uint32_t>(size), tt_cxy_pair(dev.chip_id, virt), addr);
}

void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size) {
    dev.cluster->write_dram_vec(src, static_cast<uint32_t>(size), dev.chip_id, /*channel=*/0, addr);
}

void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size) {
    dev.cluster->read_dram_vec(dst, static_cast<uint32_t>(size), dev.chip_id, /*channel=*/0, addr);
}

}  // namespace tt::foil
