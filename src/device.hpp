// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

// Forward declarations to avoid pulling in heavy headers in this file
namespace tt {
class Cluster;
namespace llrt {
class RunTimeOptions;
}
namespace tt_metal {
class Hal;
}
}  // namespace tt

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

// Core of the Device handle.  Owns UMD Cluster + HAL + allocators.
// Not exposed to external callers — only accessible via runtime.hpp APIs.
struct Device {
    uint32_t chip_id{0};
    std::unique_ptr<tt::llrt::RunTimeOptions> rtoptions;
    std::unique_ptr<tt::Cluster>              cluster;
    std::unique_ptr<tt::tt_metal::Hal>        hal;

    // Per-core L1 bump allocators, keyed by (x,y)
    std::unordered_map<uint64_t, L1Allocator> l1_allocs;

    // DRAM bump allocator (channel 0)
    DramAllocator dram_alloc;

    // Returns the key used to index l1_allocs
    static uint64_t core_key(uint32_t x, uint32_t y) { return (static_cast<uint64_t>(x) << 32) | y; }

    // Get or create an L1 allocator for a logical core.
    L1Allocator& l1_for_core(const CoreCoord& logical_core);
};

// Initialize device: UMD cluster, HAL, firmware load, wait for INIT-done.
std::unique_ptr<Device> device_open(int pcie_device_index, const std::string& firmware_dir);

// Tear down device: assert resets, close cluster.
void device_close(Device& dev);

// Resolve a logical Tensix core to a virtual coordinate understood by UMD.
tt::CoreCoord logical_to_virtual(const Device& dev, const CoreCoord& logical);

// Write `size` bytes from `src` to device L1 at `addr` on `core`.
void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size);

// Read `size` bytes from device L1 at `addr` on `core` into `dst`.
void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size);

// Write `size` bytes from `src` to DRAM at `addr`.
void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size);

// Read `size` bytes from DRAM at `addr` into `dst`.
void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size);

}  // namespace tt::foil
