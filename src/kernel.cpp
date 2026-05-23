// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kernel.hpp"
#include "device.hpp"

#include <filesystem>
#include <stdexcept>

#include "llrt_local/tt_memory.h"    // tt::foil::ll_api::memory
#include "llrt/hal.hpp"        // Hal, HalJitBuildConfig

namespace tt::foil {


Kernel* kernel_load(
    Device& dev,
    std::span<const RiscBinary> binaries,
    CoreCoord logical_core)
{
    if (binaries.empty()) {
        throw std::runtime_error("tt-foil: no binaries provided to load_kernel");
    }

    auto* kernel       = new Kernel{};
    kernel->core       = logical_core;
    {
        auto virt = logical_to_virtual(dev, logical_core);
        kernel->virt_x = virt.x;
        kernel->virt_y = virt.y;
    }

    // Allocate a contiguous RTA region inside the KERNEL_CONFIG region (MEM_MAP_END).
    // rta_offset in launch_msg is uint16_t (relative to kernel_config_base), so
    // the RTA must live within the KERNEL_CONFIG region, not DEFAULT_UNRESERVED.
    // Layout: [BRISC args (kMaxRtaWords * 4 bytes)] [NCRISC args (kMaxRtaWords * 4 bytes)]
    uint32_t rta_region_bytes = kMaxRiscs * kMaxRtaWords * sizeof(uint32_t);
    kernel->rta_base_addr  = dev.kernel_config_for_core(logical_core).alloc(rta_region_bytes, /*alignment=*/16);
    kernel->rta_region_size = rta_region_bytes;

    for (const auto& rb : binaries) {
        if (!std::filesystem::exists(rb.elf_path)) {
            delete kernel;
            throw std::runtime_error("tt-foil: ELF not found: " + rb.elf_path);
        }

        LoadedRisc lr;
        risc_to_hal_indices(rb.risc, lr.proc_class, lr.proc_type, lr.processor_index);

        // Load ELF from disk and apply XIP transformation.
        lr.mem = std::make_unique<tt::foil::ll_api::memory>(rb.elf_path, tt::foil::ll_api::memory::Loading::CONTIGUOUS_XIP);

        // Allocate space for kernel text in the KERNEL_CONFIG region.
        // The binary will be written here; firmware calls kernel_config_base + text_offset.
        std::size_t text_bytes = lr.mem->size() * sizeof(uint32_t);
        lr.kernel_text_addr = dev.kernel_config_for_core(logical_core).alloc(text_bytes, /*alignment=*/16);

        kernel->riscs.push_back(std::move(lr));
    }

    return kernel;
}

void kernel_set_runtime_args(
    Kernel& kernel,
    RiscBinary::RiscId risc,
    std::span<const uint32_t> args)
{
    uint32_t proc_class, proc_type, proc_idx;
    risc_to_hal_indices(risc, proc_class, proc_type, proc_idx);

    for (auto& lr : kernel.riscs) {
        if (lr.proc_class == proc_class && lr.proc_type == proc_type) {
            if (args.size() > kMaxRtaWords) {
                throw std::runtime_error("tt-foil: too many runtime args (max " +
                    std::to_string(kMaxRtaWords) + ")");
            }
            lr.runtime_args.assign(args.begin(), args.end());
            return;
        }
    }
    throw std::runtime_error("tt-foil: set_runtime_args called for a RISC not in this kernel");
}

void pin_persistent(Device& device, const Kernel& kernel, CoreCoord logical_core) {
    uint64_t key = Device::core_key(logical_core.x, logical_core.y);
    // Freeze current high-water marks so reset/release rewinds only
    // back to here, not all the way to base.
    auto kc_it = device.kernel_config_allocs.find(key);
    if (kc_it != device.kernel_config_allocs.end()) kc_it->second.set_watermark();
    auto l1_it = device.l1_allocs.find(key);
    if (l1_it != device.l1_allocs.end()) l1_it->second.set_watermark();
    // Mark the kernel as pinned so release_kernels won't evict it
    // from resident_kernels once its ELF has actually been NOC-written
    // (which happens on the first dispatch — pin_persistent itself
    // is too early; make_*() only *allocates* kernel_text_addr without
    // writing to L1).
    device.pinned_kernels[key].insert(&kernel);
}

void release_kernels(Device& device, CoreCoord logical_core) {
    // The kernel_config_for_core() entry is lazily created on first use.
    // Erase it; the next load_kernel() will lazily reconstruct it with
    // current == base, reclaiming the whole KERNEL_CONFIG region for new
    // kernel text + RTAs. Any std::shared_ptr<Kernel> still held by the
    // caller now has stale pointers into the (about-to-be-overwritten)
    // region — see runtime.hpp for the contract.
    uint64_t key = Device::core_key(logical_core.x, logical_core.y);
    // iter21: rewind to watermark instead of dropping the allocator —
    // anything below the watermark (a pinned persistent op's text +
    // RTAs) stays addressable. If no pin was ever set, watermark==0
    // and L1Allocator::reset() falls back to base, matching the
    // pre-iter21 behaviour.
    auto kc_it = device.kernel_config_allocs.find(key);
    if (kc_it != device.kernel_config_allocs.end()) {
        kc_it->second.reset();
    }
    // Evict resident kernels except those explicitly pinned. Pinned
    // kernels live below the kernel_config watermark; their text is
    // intact across this reset.
    auto& resident = device.resident_kernels[key];
    auto pinned_it = device.pinned_kernels.find(key);
    if (pinned_it == device.pinned_kernels.end() || pinned_it->second.empty()) {
        resident.clear();
    } else {
        const auto& pinned = pinned_it->second;
        for (auto it = resident.begin(); it != resident.end();) {
            if (pinned.contains(*it)) ++it;
            else it = resident.erase(it);
        }
    }
}

void reset_l1(Device& device, CoreCoord logical_core) {
    // iter21: L1Allocator::reset() now rewinds `current` to
    // max(watermark, base). If pin_persistent set the watermark on
    // this core, that op's L1 CB-backing buffers stay valid across
    // this call; otherwise the behaviour matches the original
    // (rewind to base). Any L1 Buffer the caller still references
    // ABOVE the watermark becomes stale (see runtime.hpp contract).
    uint64_t key = Device::core_key(logical_core.x, logical_core.y);
    auto it = device.l1_allocs.find(key);
    if (it != device.l1_allocs.end()) it->second.reset();
}

}  // namespace tt::foil
