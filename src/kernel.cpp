// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kernel.hpp"
#include "device.hpp"

#include <filesystem>
#include <stdexcept>

#include "llrt/tt_memory.h"    // ll_api::memory
#include "llrt/hal.hpp"        // Hal, HalJitBuildConfig

namespace tt::foil {

// Reserve a fixed number of 32-bit words for RTA per risc.
// This caps the maximum number of runtime args per kernel.
static constexpr uint32_t kMaxRtaWords = 64;

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

    // Allocate a contiguous RTA region in L1 for all active riscs.
    // Layout: [BRISC args (kMaxRtaWords * 4 bytes)] [NCRISC args (kMaxRtaWords * 4 bytes)]
    // Indices: rta_base_addr + proc_idx * kMaxRtaWords * 4
    uint32_t rta_region_bytes = kMaxRiscs * kMaxRtaWords * sizeof(uint32_t);
    kernel->rta_base_addr  = dev.l1_for_core(logical_core).alloc(rta_region_bytes, /*alignment=*/16);
    kernel->rta_region_size = rta_region_bytes;

    for (const auto& rb : binaries) {
        if (!std::filesystem::exists(rb.elf_path)) {
            delete kernel;
            throw std::runtime_error("tt-foil: ELF not found: " + rb.elf_path);
        }

        LoadedRisc lr;
        risc_to_hal_indices(rb.risc, lr.proc_class, lr.proc_type, lr.processor_index);

        // Load ELF from disk and apply XIP transformation.
        lr.mem = std::make_unique<ll_api::memory>(rb.elf_path, ll_api::memory::Loading::CONTIGUOUS_XIP);

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

}  // namespace tt::foil
