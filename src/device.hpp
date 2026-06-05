// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// UMD xy_pair = tt::tt_metal::CoreCoord (needed as return type below)
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "firmware_paths.hpp"

// Forward declarations — complete types provided in device.cpp
namespace tt {
namespace umd {
class Cluster;
}
}  // namespace tt
namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::foil {

struct CoreCoord;
struct Kernel;

// Internal per-core allocator state
struct L1Allocator {
    uint64_t base{0};
    uint64_t current{0};
    uint64_t end{0};
    // iter21: watermark for persistent allocations. reset() rewinds
    // `current` back to `watermark` (≥ base). Caller can call
    // set_watermark() after staging persistent buffers to keep them
    // alive across reset_l1() / release_kernels() cycles.
    uint64_t watermark{0};

    uint64_t alloc(std::size_t bytes, uint32_t alignment = 16);
    void reset();
    void set_watermark() { watermark = current; }
};

// Whole-device DRAM allocator (channel 0)
struct DramAllocator {
    uint64_t base{0};
    uint64_t current{0};
    uint64_t end{0};

    uint64_t alloc(std::size_t bytes, uint32_t alignment = 32);
    void reset();
};

// Device handle. As of Phase B3 steps 2 + 3, Device owns both the HAL and
// the umd::Cluster. Nothing in tt-foil references tt::Cluster or
// MetalContext anymore; ll_api::memory is bundled locally so kernel ELF
// loading doesn't drag MetalContext in either. libtt_metal.so is still
// linked (HAL impl + a few transitive symbols), but the singleton init that
// used to fight our chip open at kernel_load time no longer happens.
struct Device {
    uint32_t chip_id{0};

    // Owned by Device — destructor defined in device.cpp where the full
    // types are visible.
    std::unique_ptr<tt::tt_metal::Hal> owned_hal;
    std::unique_ptr<tt::umd::Cluster>  owned_cluster;

    // Convenience aliases into the owned objects above. Keep `umd_driver`
    // so the rest of the codebase (dispatch.cpp, buffer.cpp, …) doesn't
    // change shape.
    const tt::tt_metal::Hal*    hal{nullptr};
    tt::umd::Cluster*           umd_driver{nullptr};

    // Translated coord + offset of DRAM channel 0's preferred worker core.
    // Cached at device_open so write_dram/read_dram don't need tt::Cluster.
    tt::umd::CoreCoord          dram0_core{};
    uint64_t                    dram0_offset{0};

    // All DRAM channels' preferred-worker cores (TRANSLATED) + address
    // offsets, resolved at device_open. dram_cores[0] == dram0_core.
    // Blackhole has 8 independent channels; used for multi-channel weight
    // sharding to lift the single-channel decode read-bandwidth ceiling.
    std::vector<tt::umd::CoreCoord> dram_cores;
    std::vector<uint64_t>           dram_offsets;

    // Firmware ELF paths resolved at device_open. Cached so the runtime
    // manifest check in kernel_load() can compare kernel ELFs against the
    // firmware they were linked against. See src/kernel_manifest.cpp.
    FirmwarePaths               firmware{};

    // Per-core L1 bump allocators, keyed by (x,y)
    std::unordered_map<uint64_t, L1Allocator> l1_allocs;

    // Per-core kernel-config region allocators (KERNEL_CONFIG = MEM_MAP_END).
    // RTA must live here so the uint16_t rta_offset fits.
    std::unordered_map<uint64_t, L1Allocator> kernel_config_allocs;

    // Per-core set of Kernels whose ELF is currently resident in L1
    // (already NOC-written and not yet displaced by an arena rewind).
    // dispatch_stage_setup uses this to skip the ELF NOC write when a
    // Kernel is re-dispatched. release_kernels clears the per-core set
    // because rewinding the KERNEL_CONFIG arena lets the next load_kernel
    // reuse those L1 addresses for new binaries.
    std::unordered_map<uint64_t, std::unordered_set<const Kernel*>> resident_kernels;

    // iter21: per-core set of Kernels that are NOT evicted by
    // release_kernels(). Used together with the L1/kernel_config
    // watermark so a persistent op can outlive surrounding
    // reset_l1/release_kernels cycles.
    std::unordered_map<uint64_t, std::unordered_set<const Kernel*>> pinned_kernels;

    // DRAM bump allocator (channel 0)
    DramAllocator dram_alloc;

    // Cores that were cold-booted at device_open time. kernel_load() only
    // accepts cores listed here. Order matches the `cores` argument to
    // open_device() — preserved so error messages can refer back to it.
    std::vector<CoreCoord> booted_cores;

    // R5 G2: optional fast-dispatch instance. When set, dispatch_execute_multi
    // routes through it for single-kernel launches (host writes ELF/RTA/launch_msg
    // via PCIe as usual, then FD fires GO + polls DONE over NOC).
    // Owned externally (caller is responsible for lifetime + start/terminate).
    struct FastDispatch* fast_dispatch{nullptr};

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

// Open UMD + cold-boot the requested Tensix cores. `cores` must contain at
// least one entry. Throws std::runtime_error on any per-core boot failure.
std::unique_ptr<Device> device_open(
    int pcie_device_index,
    const std::string& firmware_dir,
    std::vector<CoreCoord> cores);

// Tear down: drop owned UMD cluster + HAL, clear pointers.
void device_close(Device& dev);

// Resolve a logical Tensix core to a virtual coordinate understood by UMD.
tt::xy_pair logical_to_virtual(const Device& dev, const CoreCoord& logical);

void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size);
void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size);
void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size);
void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size);

}  // namespace tt::foil
