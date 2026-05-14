// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mailbox_init.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

void init_tensix_mailboxes(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {

    using namespace tt::tt_metal;

    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);

    // ---- launch_msg buffer: N copies of all-zero launch_msg_t ----
    auto zero_launch_msg = factory.create<dev_msgs::launch_msg_t>();
    const std::size_t lm_size = zero_launch_msg.size();
    std::vector<std::byte> launch_buf(lm_size * dev_msgs::launch_msg_buffer_num_entries, std::byte{0});
    // (No copy needed — zero_launch_msg is already all-zero, and we want all
    //  buffer entries zero, which is the default-constructed std::byte{0}.)

    const uint64_t launch_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH);
    driver.write_to_device(launch_buf.data(), launch_buf.size(),
                           chip_id, core, launch_addr);

    // ---- go_msg: signal = RUN_MSG_INIT ----
    auto go = factory.create<dev_msgs::go_msg_t>();
    go.view().signal() = dev_msgs::RUN_MSG_INIT;
    const uint64_t go_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    driver.write_to_device(go.data(), go.size(), chip_id, core, go_addr);

    // ---- launch_msg buffer read pointer: 0 ----
    const uint32_t zero32 = 0;
    const uint64_t rd_ptr_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    driver.write_to_device(&zero32, sizeof(zero32), chip_id, core, rd_ptr_addr);

    // ---- go_message index: 0 ----
    const uint64_t go_idx_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG_INDEX);
    driver.write_to_device(&zero32, sizeof(zero32), chip_id, core, go_idx_addr);
}

}  // namespace tt::foil
