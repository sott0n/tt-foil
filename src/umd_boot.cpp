// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "umd_boot.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::foil {

UmdHandle umd_open(int pcie_device_index) {
    if (pcie_device_index < 0) {
        throw std::runtime_error(
            "tt-foil: umd_open: invalid pcie_device_index " +
            std::to_string(pcie_device_index));
    }

    tt::umd::ClusterOptions opts;
    opts.chip_type = tt::umd::ChipType::SILICON;
    // Restrict the cluster to just the requested chip. Other MMIO chips in the
    // system are left alone (UMD can co-exist with other processes that have
    // their own chips open, but not with another open of the same chip).
    opts.target_devices = {static_cast<tt::ChipId>(pcie_device_index)};

    UmdHandle h;
    h.cluster = std::make_unique<tt::umd::Cluster>(std::move(opts));
    h.chip_id = static_cast<uint32_t>(pcie_device_index);
    return h;
}

}  // namespace tt::foil
