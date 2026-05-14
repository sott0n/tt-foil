// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "core_info_init.hpp"

#include <cstdint>

#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

void init_tensix_core_info_minimal(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    uint32_t logical_x,
    uint32_t logical_y) {

    using namespace tt::tt_metal;

    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto core_info = factory.create<dev_msgs::core_info_msg_t>();
    // Buffer is zero-initialized by Factory::create<>(); only set the fields
    // the boot firmware actually reads.
    auto v = core_info.view();
    v.absolute_logical_x() = logical_x;
    v.absolute_logical_y() = logical_y;
    v.core_magic_number()  = dev_msgs::CoreMagicNumber::WORKER;

    const uint64_t core_info_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::CORE_INFO);

    driver.write_to_device(core_info.data(), core_info.size(),
                           chip_id, core, core_info_addr);
}

}  // namespace tt::foil
