// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

// tt-metal llrt headers (from submodule)
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "llrt/hal_types.hpp"
#include "llrt/tt_memory.h"
#include "llrt/tt_elffile.hpp"

// Blackhole memory map constants
#include "dev_mem_map.h"         // MEM_BRISC_FIRMWARE_BASE etc.
#include "dev_msgs.h"            // launch_msg_t, go_msg_t, RUN_MSG_*

// UMD types
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "tt_foil/runtime.hpp"  // for CoreCoord

namespace tt::foil {

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

L1Allocator& Device::l1_for_core(const CoreCoord& logical_core) {
    uint64_t key = core_key(logical_core.x, logical_core.y);
    auto it = l1_allocs.find(key);
    if (it == l1_allocs.end()) {
        // First access: initialise from HAL constants.
        // USER area starts at UNRESERVED (above the firmware + mailbox + kernel config).
        // HAL does not expose UNRESERVED for TENSIX directly; we use the kernel config
        // region end: BASE + KERNEL_CONFIG_SIZE rounds up to user space start.
        uint64_t user_base =
            hal->get_dev_noc_addr(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::BASE) +
            hal->get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::BASE) -
            /* subtract kernel_config size to locate unreserved region */ 0;

        // Simpler: use known Blackhole constants directly.
        // MEM_MAP_END is the end of the reserved firmware/mailbox area.
        // Kernel config buffer follows immediately.  The user area starts after that.
        // Default kernel config buffer = 69 KB per the HAL.
        constexpr uint64_t KERNEL_CONFIG_SIZE = 69 * 1024;
        uint64_t l1_user_base = MEM_MAP_END + KERNEL_CONFIG_SIZE;
        uint64_t l1_end       = static_cast<uint64_t>(MEM_L1_SIZE);

        L1Allocator alloc{};
        alloc.base    = l1_user_base;
        alloc.current = l1_user_base;
        alloc.end     = l1_end;
        l1_allocs.emplace(key, alloc);
        return l1_allocs.at(key);
    }
    return it->second;
}

// Write an ELF binary to a single core's L1, applying the HAL relocation.
// Equivalent to the relevant part of llrt::test_load_write_read_risc_binary
// but without going through MetalContext.
static void load_elf_to_core(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const tt::CoreCoord& virt_core,
    uint64_t local_init_addr)
{
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr = hal.relocate_dev_addr(addr, local_init_addr, /*shared_local=*/false);
        cluster.write_core(
            &*mem_ptr,
            len_words * sizeof(uint32_t),
            tt_cxy_pair(chip_id, virt_core),
            relo_addr);
    });
}

// ---- Firmware loading ----

// Expected filenames under firmware_dir for each RISC
static constexpr struct { const char* filename; } kFwFiles[] = {
    {"brisc.elf"},    // BRISC  - processor_class DM, type 0
    {"ncrisc.elf"},   // NCRISC - processor_class DM, type 1
    {"trisc0.elf"},   // TRISC0 - processor_class COMPUTE, type 0
    {"trisc1.elf"},   // TRISC1 - processor_class COMPUTE, type 1
    {"trisc2.elf"},   // TRISC2 - processor_class COMPUTE, type 2
};

static void load_firmware(Device& dev, const std::string& firmware_dir) {
    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::Cluster& cluster         = *dev.cluster;

    // The Tensix HAL has two processor classes: DM (0) and COMPUTE (1).
    // DM has 2 types (BRISC=0, NCRISC=1); COMPUTE has 3 types (TRISC0/1/2).
    // kFwFiles[i] index maps:  0=BRISC, 1=NCRISC, 2=TRISC0, 3=TRISC1, 4=TRISC2.
    struct FwEntry { uint32_t proc_class; uint32_t proc_type; std::string filename; };
    std::vector<FwEntry> entries = {
        {0, 0, "brisc.elf"},
        {0, 1, "ncrisc.elf"},
        {1, 0, "trisc0.elf"},
        {1, 1, "trisc1.elf"},
        {1, 2, "trisc2.elf"},
    };

    // Get all Tensix worker cores from the cluster (virtual coordinates).
    auto tensix_cores = cluster.get_soc_desc(dev.chip_id).physical_workers;

    for (const auto& entry : entries) {
        std::string elf_path = firmware_dir + "/" + entry.filename;
        if (!std::filesystem::exists(elf_path)) {
            throw std::runtime_error("tt-foil: firmware not found: " + elf_path);
        }

        const auto& jit_cfg = hal.get_jit_build_config(
            static_cast<uint32_t>(tt_metal::HalProgrammableCoreType::TENSIX),
            entry.proc_class,
            entry.proc_type);

        ll_api::memory mem(elf_path, ll_api::memory::Loading::CONTIGUOUS_XIP);

        for (const auto& phys_core : tensix_cores) {
            tt::CoreCoord virt_core{phys_core.x, phys_core.y};
            load_elf_to_core(cluster, hal, mem, dev.chip_id, virt_core, jit_cfg.local_init_addr);

            // Program the launch address register for non-BRISC RISCs.
            if (jit_cfg.fw_launch_addr != 0) {
                uint32_t launch_val = static_cast<uint32_t>(jit_cfg.fw_launch_addr_value);
                cluster.write_core_immediate(
                    &launch_val, sizeof(launch_val),
                    tt_cxy_pair(dev.chip_id, virt_core),
                    jit_cfg.fw_launch_addr);
            }
        }
    }
}

// Poll go_msg.signal == RUN_MSG_INIT on all Tensix cores (firmware boot confirmation).
static void wait_for_fw_init(Device& dev, int timeout_ms = 10000) {
    const tt::tt_metal::Hal& hal = *dev.hal;
    tt::Cluster& cluster         = *dev.cluster;

    auto tensix_cores = cluster.get_soc_desc(dev.chip_id).physical_workers;
    uint64_t go_addr  = hal.get_dev_noc_addr(
        tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::GO_MSG);

    auto start = std::chrono::steady_clock::now();
    for (const auto& phys_core : tensix_cores) {
        tt::CoreCoord virt{phys_core.x, phys_core.y};
        while (true) {
            uint32_t signal = 0;
            cluster.read_core(
                &signal, sizeof(signal),
                tt_cxy_pair(dev.chip_id, virt),
                go_addr & ~0x3ULL);
            uint8_t sig_byte = static_cast<uint8_t>(signal);
            if (sig_byte == tt_metal::dev_msgs::RUN_MSG_DONE ||
                sig_byte == tt_metal::dev_msgs::RUN_MSG_INIT) {
                break;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (elapsed > timeout_ms) {
                throw std::runtime_error("tt-foil: timeout waiting for firmware INIT on core ("
                    + std::to_string(virt.x) + "," + std::to_string(virt.y) + ")");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ---- Public device API ----

std::unique_ptr<Device> device_open(int pcie_device_index, const std::string& firmware_dir) {
    auto dev = std::make_unique<Device>();
    dev->chip_id = static_cast<uint32_t>(pcie_device_index);

    // Build RunTimeOptions — the minimum required for Cluster construction.
    dev->rtoptions = std::make_unique<tt::llrt::RunTimeOptions>();

    // Construct the UMD cluster (PCIe enumeration, TLB setup, reset assertion).
    dev->cluster = std::make_unique<tt::Cluster>(*dev->rtoptions);

    // Construct the HAL for Blackhole.
    // All optional features disabled for the minimal embedding runtime.
    dev->hal = std::make_unique<tt::tt_metal::Hal>(
        tt::ARCH::BLACKHOLE,
        /*is_base_routing_fw_enabled=*/false,
        /*enable_2_erisc_mode=*/false,
        /*profiler_dram_bank_size_per_risc_bytes=*/0,
        /*enable_dram_backed_cq=*/false,
        /*is_simulator=*/false,
        /*enable_blackhole_dram_programmable_cores=*/false);

    // Initialise DRAM bump allocator from HAL.
    uint64_t dram_base = dev->hal->get_dev_addr(tt_metal::HalDramMemAddrType::UNRESERVED);
    uint64_t dram_size = dev->hal->get_dev_size(tt_metal::HalDramMemAddrType::UNRESERVED);
    dev->dram_alloc = DramAllocator{dram_base, dram_base, dram_base + dram_size};

    // Load management firmware to all Tensix cores.
    std::string fw_dir = firmware_dir;
    if (fw_dir.empty()) {
        // Default: look for firmware relative to the tt-foil binary location.
        fw_dir = "firmware";
    }
    load_firmware(*dev, fw_dir);

    // Deassert RISC reset so firmware can start executing.
    dev->cluster->deassert_risc_reset();

    // Wait for firmware to reach INIT state on all cores.
    wait_for_fw_init(*dev);

    return dev;
}

void device_close(Device& dev) {
    // Assert resets on all Tensix cores before teardown.
    dev.cluster->assert_risc_reset();
    // Cluster destructor handles UMD teardown.
}

tt::CoreCoord logical_to_virtual(const Device& dev, const CoreCoord& logical) {
    // For a single-chip Blackhole setup, logical == virtual for Tensix workers.
    // A full implementation would use the cluster's core-mapping API.
    return tt::CoreCoord{logical.x, logical.y};
}

void write_l1(Device& dev, const CoreCoord& core, uint64_t addr, const void* src, std::size_t size) {
    tt::CoreCoord virt = logical_to_virtual(dev, core);
    dev.cluster->write_core(src, static_cast<uint32_t>(size), tt_cxy_pair(dev.chip_id, virt), addr);
}

void read_l1(Device& dev, const CoreCoord& core, uint64_t addr, void* dst, std::size_t size) {
    tt::CoreCoord virt = logical_to_virtual(dev, core);
    dev.cluster->read_core(dst, static_cast<uint32_t>(size), tt_cxy_pair(dev.chip_id, virt), addr);
}

void write_dram(Device& dev, uint64_t addr, const void* src, std::size_t size) {
    dev.cluster->write_dram_vec(src, static_cast<uint32_t>(size), dev.chip_id, /*channel=*/0, addr);
}

void read_dram(Device& dev, uint64_t addr, void* dst, std::size_t size) {
    dev.cluster->read_dram_vec(dst, static_cast<uint32_t>(size), dev.chip_id, /*channel=*/0, addr);
}

}  // namespace tt::foil
