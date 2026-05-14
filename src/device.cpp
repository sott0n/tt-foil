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
#include "llrt/metal_soc_descriptor.hpp"

// UMD direct API (Phase B1)
#include <umd/device/cluster.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

// NOC enum for soc_desc DRAM lookup
#include "tt-metalium/kernel_types.hpp"

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

    // Cache the underlying UMD driver so host<->core I/O can bypass tt::Cluster.
    dev->umd_driver = dev->cluster->get_driver().get();

    // Pre-translate DRAM channel 0's preferred worker core into UMD's
    // CoordSystem::TRANSLATED, plus the per-view address offset.
    {
        const auto& soc_desc = dev->cluster->get_soc_desc(dev->chip_id);
        ::CoreCoord dram_xy = soc_desc.get_preferred_worker_core_for_dram_view(0, tt_metal::NOC::NOC_0);
        dev->dram0_core = tt::umd::CoreCoord{
            dram_xy.x, dram_xy.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
        dev->dram0_offset = soc_desc.get_address_offset(0);
    }

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
        dev.umd_driver = nullptr;
    }
}

tt::xy_pair logical_to_virtual(const Device& dev, const CoreCoord& logical) {
    tt::tt_metal::CoreCoord virt =
        dev.tt_device->worker_core_from_logical_core({logical.x, logical.y});
    return TtCoreCoord{virt.x, virt.y};
}

static tt::umd::CoreCoord tensix_translated(const Device& dev, const CoreCoord& core) {
    TtCoreCoord virt = logical_to_virtual(dev, core);
    return tt::umd::CoreCoord{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
}

void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size) {
    auto cc = tensix_translated(dev, core);
    dev.umd_driver->write_to_device(src, size, dev.chip_id, cc, addr);
}

void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size) {
    auto cc = tensix_translated(dev, core);
    dev.umd_driver->read_from_device(dst, dev.chip_id, cc, addr, size);
}

void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size) {
    dev.umd_driver->write_to_device(src, size, dev.chip_id, dev.dram0_core, addr + dev.dram0_offset);
}

void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size) {
    dev.umd_driver->read_from_device(dst, dev.chip_id, dev.dram0_core, addr + dev.dram0_offset, size);
}

}  // namespace tt::foil
