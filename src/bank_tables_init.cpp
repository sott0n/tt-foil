// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bank_tables_init.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

namespace {

void zero_region(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    uint64_t addr,
    uint32_t size_bytes) {
    if (size_bytes == 0) return;
    std::vector<std::byte> zeros(size_bytes, std::byte{0});
    driver.write_to_device(zeros.data(), zeros.size(), chip_id, core, addr);
}

}  // namespace

void zero_fill_bank_tables(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {

    using namespace tt::tt_metal;

    const uint64_t bank_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t bank_size = hal.get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    zero_region(driver, chip_id, core, bank_addr, bank_size);

    const uint64_t l2v_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t l2v_size = hal.get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    zero_region(driver, chip_id, core, l2v_addr, l2v_size);
}

}  // namespace tt::foil
