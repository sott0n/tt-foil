// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

// UMD xy_pair = tt::tt_metal::CoreCoord (needed as return type below)
#include <umd/device/types/xy_pair.hpp>

// Forward declarations — complete types provided in device.cpp
namespace tt {
class Cluster;
}
namespace tt::tt_metal {
class Hal;
class IDevice;
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

// Device handle. Firmware init is delegated to tt-metal's CreateDevice().
// cluster and hal are borrowed references from MetalContext (valid while tt_device is alive).
struct Device {
    uint32_t chip_id{0};

    // Owned: call CloseDevice() in device_close().
    tt::tt_metal::IDevice* tt_device{nullptr};

    // Borrowed from MetalContext — valid while tt_device is alive.
    tt::Cluster*                cluster{nullptr};
    const tt::tt_metal::Hal*    hal{nullptr};

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
