// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <stdexcept>

// MetalContext — cluster and HAL accessors. Implicitly initialised on first
// `instance()` call; no need to go through CreateDevice anymore.
#include "impl/context/metal_context.hpp"

// Full cluster and HAL headers (needed for member function calls in this file)
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"

// UMD direct API (Phase B1)
#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

// NOC enum for soc_desc DRAM lookup
#include "tt-metalium/kernel_types.hpp"

#include "tt_foil/runtime.hpp"  // for CoreCoord

// Phase B2: UMD-direct cold boot pieces (replaces tt::tt_metal::CreateDevice)
#include "bank_tables_init.hpp"
#include "core_info_init.hpp"
#include "firmware_load.hpp"
#include "firmware_paths.hpp"
#include "mailbox_init.hpp"
#include "reset.hpp"

#include <cstdlib>

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

// Translate a logical Tensix coord to UMD's TRANSLATED coord system via the
// soc descriptor. This replaces the IDevice::worker_core_from_logical_core
// call we used to make before dropping CreateDevice.
static tt::umd::CoreCoord soc_logical_to_translated(
    const metal_SocDescriptor& soc_desc, uint32_t logical_x, uint32_t logical_y) {
    tt::umd::CoreCoord logical{
        logical_x, logical_y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
    return soc_desc.translate_coord_to(logical, tt::CoordSystem::TRANSLATED);
}

std::unique_ptr<Device> device_open(int pcie_device_index, const std::string& /*firmware_dir*/) {
    auto dev = std::make_unique<Device>();
    dev->chip_id = static_cast<uint32_t>(pcie_device_index);

    // ---------------------------------------------------------------------
    // Phase B2 step 8: cold-boot the chip ourselves, no CreateDevice.
    //
    // MetalContext::instance() implicitly constructs MetalEnv, which in turn
    // builds a tt::Cluster (=> umd::Cluster) and a Hal — without touching
    // device firmware. We borrow the driver pointer and HAL from there.
    // ---------------------------------------------------------------------
    auto& ctx   = tt::tt_metal::MetalContext::instance();
    dev->cluster    = &ctx.get_cluster();
    dev->hal        = &ctx.hal();
    dev->umd_driver = dev->cluster->get_driver().get();

    const auto& soc_desc = dev->cluster->get_soc_desc(dev->chip_id);

    // Pre-translate DRAM channel 0 worker core for write_dram / read_dram.
    {
        ::CoreCoord dram_xy = soc_desc.get_preferred_worker_core_for_dram_view(0, tt_metal::NOC::NOC_0);
        dev->dram0_core = tt::umd::CoreCoord{
            dram_xy.x, dram_xy.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
        dev->dram0_offset = soc_desc.get_address_offset(0);
    }

    // Init DRAM bump allocator from HAL.
    uint64_t dram_base = dev->hal->get_dev_addr(tt_metal::HalDramMemAddrType::UNRESERVED);
    uint64_t dram_size = dev->hal->get_dev_size(tt_metal::HalDramMemAddrType::UNRESERVED);
    dev->dram_alloc = DramAllocator{dram_base, dram_base, dram_base + dram_size};

    // ---------------------------------------------------------------------
    // Cold-boot logical (0,0). Other Tensix cores stay in whatever state
    // they were in (typically reset on a fresh power-up). Multi-core boot
    // can be added later by looping this block over a set of cores.
    // ---------------------------------------------------------------------
    auto fw = resolve_firmware_paths(
        // tt-metal source root is needed for auto-discovery; rtoptions is
        // initialised lazily and may not have set its root_dir at this point,
        // so honour TT_METAL_RUNTIME_ROOT explicitly with a sensible default.
        []() -> std::string {
            if (const char* p = std::getenv("TT_METAL_RUNTIME_ROOT")) return p;
            return "/home/kyamaguchi/tt-metal";
        }());

    constexpr uint32_t kLogX = 0, kLogY = 0;
    auto core = soc_logical_to_translated(soc_desc, kLogX, kLogY);

    assert_tensix_reset(*dev->umd_driver, dev->chip_id, core);

    load_tensix_firmware(*dev->umd_driver, *dev->hal, dev->chip_id, core, fw.brisc,  kBrisc);
    load_tensix_firmware(*dev->umd_driver, *dev->hal, dev->chip_id, core, fw.ncrisc, kNcrisc);
    load_tensix_firmware(*dev->umd_driver, *dev->hal, dev->chip_id, core, fw.trisc0, kTrisc0);
    load_tensix_firmware(*dev->umd_driver, *dev->hal, dev->chip_id, core, fw.trisc1, kTrisc1);
    load_tensix_firmware(*dev->umd_driver, *dev->hal, dev->chip_id, core, fw.trisc2, kTrisc2);

    zero_fill_bank_tables       (*dev->umd_driver, *dev->hal, dev->chip_id, core);
    init_tensix_core_info_minimal(*dev->umd_driver, *dev->hal, dev->chip_id, core, kLogX, kLogY);
    init_tensix_mailboxes        (*dev->umd_driver, *dev->hal, dev->chip_id, core);

    deassert_brisc_reset(*dev->umd_driver, dev->chip_id, core);
    wait_tensix_init_done(*dev->umd_driver, *dev->hal, dev->chip_id, core, /*timeout_ms=*/10000);

    return dev;
}

void device_close(Device& dev) {
    // No CloseDevice/IDevice to tear down — MetalContext stays alive for the
    // process. Clear the borrowed pointers so dangling use is loud.
    dev.cluster    = nullptr;
    dev.hal        = nullptr;
    dev.umd_driver = nullptr;
}

tt::xy_pair logical_to_virtual(const Device& dev, const CoreCoord& logical) {
    const auto& soc_desc = dev.cluster->get_soc_desc(dev.chip_id);
    auto t = soc_logical_to_translated(soc_desc, logical.x, logical.y);
    return TtCoreCoord{t.x, t.y};
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
