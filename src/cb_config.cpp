// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cb_config.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "device.hpp"
#include "kernel.hpp"

#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

namespace {

constexpr uint32_t kUint32WordsPerCb = 4;        // matches UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG
constexpr uint32_t kCbBlobAlignment  = 16;       // CB descriptors are 16 B aligned

}  // namespace

void register_cbs(
    Device& dev,
    Kernel& kernel,
    std::span<const CbConfig> cbs) {

    if (cbs.empty()) {
        kernel.cb_alloc = CbAllocation{};  // invalid (valid=false)
        return;
    }

    // Find min/max indices and validate uniqueness.
    uint8_t min_idx = 255, max_idx = 0;
    uint64_t seen_mask = 0;
    for (const auto& c : cbs) {
        if (c.cb_index >= 32) {
            // Blackhole's firmware uses 32-bit local_cb_mask. Upper CBs (32..63)
            // exist on BH but aren't reachable through this mask path. Reject
            // here rather than silently truncate.
            throw std::runtime_error(
                "tt-foil: register_cbs: cb_index must be < 32 (got " +
                std::to_string(c.cb_index) + ")");
        }
        if (seen_mask & (uint64_t(1) << c.cb_index)) {
            throw std::runtime_error(
                "tt-foil: register_cbs: cb_index " +
                std::to_string(c.cb_index) + " listed twice");
        }
        seen_mask |= uint64_t(1) << c.cb_index;
        min_idx = std::min(min_idx, c.cb_index);
        max_idx = std::max(max_idx, c.cb_index);
    }

    // Blob covers cb indices [min_idx, max_idx]; gaps are zero descriptors so
    // the firmware's mask-driven loop skips them.
    const uint32_t num_descriptors = static_cast<uint32_t>(max_idx - min_idx + 1);
    const uint32_t blob_words      = num_descriptors * kUint32WordsPerCb;
    const uint32_t blob_bytes      = blob_words * sizeof(uint32_t);

    std::vector<uint32_t> blob(blob_words, 0);
    for (const auto& c : cbs) {
        const uint32_t rel_idx = static_cast<uint32_t>(c.cb_index - min_idx);
        uint32_t* slot = blob.data() + rel_idx * kUint32WordsPerCb;
        slot[0] = static_cast<uint32_t>(c.fifo_addr);  // raw bytes; TRISC >>4 at read
        slot[1] = c.fifo_size;
        slot[2] = c.num_pages;
        slot[3] = c.page_size;
    }

    // Allocate inside KERNEL_CONFIG region for this core (the same bump
    // allocator that the per-RISC kernel ELF + RTA slots come from). Aligned
    // to 16 B because firmware does pointer arithmetic in 16 B units.
    auto& kcfg = dev.kernel_config_for_core(kernel.core);
    const uint64_t blob_addr = kcfg.alloc(blob_bytes, kCbBlobAlignment);

    // kernel_config_base = HAL TENSIX KERNEL_CONFIG addr — relative offset
    // for the launch_msg field.
    const uint64_t kcfg_base = dev.hal->get_dev_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
    if (blob_addr < kcfg_base) {
        throw std::runtime_error("tt-foil: register_cbs: CB blob allocated below kernel_config_base");
    }
    const uint32_t local_cb_offset = static_cast<uint32_t>(blob_addr - kcfg_base);

    // Write the blob to L1.
    tt::umd::CoreCoord cc{
        kernel.virt_x, kernel.virt_y,
        tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    dev.umd_driver->write_to_device(
        blob.data(), blob_bytes, dev.chip_id, cc, blob_addr);

    // Record the allocation; dispatch_execute will pick it up next launch.
    CbAllocation a;
    a.blob_l1_addr             = blob_addr;
    a.local_cb_offset          = local_cb_offset;
    a.local_cb_mask            = static_cast<uint32_t>(seen_mask);
    a.min_local_cb_start_index = min_idx;
    a.valid                    = true;
    kernel.cb_alloc = a;
}

}  // namespace tt::foil
