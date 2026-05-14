// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 2): UMD Cluster direct open.
//
// CreateDevice() does a great many things — initializes MetalContext,
// dispatch_core_manager, build envs, kernel JIT, firmware load, fast-dispatch
// kernel binaries, etc. For an embedding-style runtime we want the bare
// minimum: open the UMD driver against one PCIe-attached chip and stop there.
//
// This header provides that bare open. Firmware boot, mailbox init and
// reset deassert are intentionally not done here — those land in later steps.
// At this stage the chip's RISCs remain in reset (their initial state after
// driver open), but host-side NOC reads/writes to L1 / DRAM via UMD already
// work because they go straight to the device through PCIe BARs.

#pragma once

#include <cstdint>
#include <memory>

namespace tt::umd {
class Cluster;
}

namespace tt::foil {

struct UmdHandle {
    std::unique_ptr<tt::umd::Cluster> cluster;  // owned: drops chip on destruction
    uint32_t chip_id{0};                        // logical chip id (== pcie index for single MMIO chip)
};

// Open a UMD Cluster scoped to a single PCIe-attached chip.
// Implemented as ClusterOptions{ chip_type=SILICON, target_devices={pcie_index} }.
// Throws std::runtime_error on failure (chip not present, driver mismatch, ...).
UmdHandle umd_open(int pcie_device_index);

}  // namespace tt::foil
