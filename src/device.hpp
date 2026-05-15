// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

// UMD xy_pair = tt::tt_metal::CoreCoord (needed as return type below)
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

// Forward declarations — complete types provided in device.cpp
namespace tt {
class Cluster;
namespace umd {
class Cluster;
}
}  // namespace tt
namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::foil {

struct CoreCoord;

// Internal per-core allocator state
struct L1Allocator {
    uint64_t base{0};
    uint64_t current{0};
    uint64_t end{0};

    uint64_t alloc(std::size_t bytes, uint32_t alignment = 16);
    void reset();
};

// Whole-device DRAM allocator (channel 0)
struct DramAllocator {
    uint64_t base{0};
    uint64_t current{0};
    uint64_t end{0};

    uint64_t alloc(std::size_t bytes, uint32_t alignment = 32);
    void reset();
};

// Device handle. As of Phase B3 step 1, HAL is owned by Device directly
// (no longer borrowed from MetalContext::hal()). umd::Cluster is still
// borrowed from MetalContext::instance().get_cluster().get_driver() — owning
// our own umd::Cluster requires also stopping ll_api::memory(XIP) from
// implicitly initialising MetalContext (it does so for an XIP-dump debug
// path), which means bundling tt_memory.cpp locally. That work is B3-2 +
// B3-3 combined; see commit log.
struct Device {
    uint32_t chip_id{0};

    // Borrowed from MetalContext — valid for the lifetime of the process.
    tt::Cluster*                cluster{nullptr};

    // Owned by Device — destructor defined in device.cpp.
    std::unique_ptr<tt::tt_metal::Hal> owned_hal;

    // Convenience alias into owned_hal.
    const tt::tt_metal::Hal*    hal{nullptr};

    // UMD driver borrowed from cluster->get_driver().
    tt::umd::Cluster*           umd_driver{nullptr};

    // Translated coord + offset of DRAM channel 0's preferred worker core.
    // Cached at device_open so write_dram/read_dram don't need tt::Cluster.
    tt::umd::CoreCoord          dram0_core{};
    uint64_t                    dram0_offset{0};

    // Per-core L1 bump allocators, keyed by (x,y)
    std::unordered_map<uint64_t, L1Allocator> l1_allocs;

    // Per-core kernel-config region allocators (KERNEL_CONFIG = MEM_MAP_END).
    // RTA must live here so the uint16_t rta_offset fits.
    std::unordered_map<uint64_t, L1Allocator> kernel_config_allocs;

    // DRAM bump allocator (channel 0)
    DramAllocator dram_alloc;

    static uint64_t core_key(uint32_t x, uint32_t y) { return (static_cast<uint64_t>(x) << 32) | y; }

    L1Allocator& l1_for_core(const CoreCoord& logical_core);
    L1Allocator& kernel_config_for_core(const CoreCoord& logical_core);

    // Defined in device.cpp where tt::tt_metal::Hal is a complete type
    // (unique_ptr destructor requires it).
    Device();
    ~Device();
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

// Open device via tt-metal CreateDevice() (handles FW init).
std::unique_ptr<Device> device_open(int pcie_device_index, const std::string& firmware_dir);

// Close device via tt-metal CloseDevice().
void device_close(Device& dev);

// Resolve a logical Tensix core to a virtual coordinate understood by UMD.
tt::xy_pair logical_to_virtual(const Device& dev, const CoreCoord& logical);

void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size);
void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size);
void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size);
void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size);

}  // namespace tt::foil
