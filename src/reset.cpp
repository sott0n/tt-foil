// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reset.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/risc_type.hpp>

namespace tt::foil {

void assert_tensix_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {
    // Default mask asserts ALL Tensix RISCs (BRISC + NCRISC + TRISC0/1/2).
    driver.assert_risc_reset_at_core(chip_id, core);
}

void deassert_brisc_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {
    // umd::Cluster::deassert_risc_reset takes a RiscType bitmask. BRISC alone
    // is the right value on Blackhole; brisc firmware brings up NCRISC and
    // TRISCs from there.
    driver.deassert_risc_reset(chip_id, core, tt::umd::RiscType::BRISC, /*staggered_start=*/true);
}

void wait_tensix_init_done(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    int timeout_ms) {

    using namespace tt::tt_metal;

    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    const uint64_t go_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);

    auto go = factory.create<dev_msgs::go_msg_t>();
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        driver.read_from_device(go.data(), chip_id, core, go_addr, go.size());
        const uint8_t sig = go.view().signal();
        if (sig == dev_msgs::RUN_MSG_DONE) {
            return;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (timeout_ms > 0 && elapsed > timeout_ms) {
            throw std::runtime_error(
                "tt-foil: timeout waiting for firmware init to complete on core "
                "(" + std::to_string(core.x) + "," + std::to_string(core.y) + "); "
                "last go_msg.signal = 0x" + std::to_string(sig));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

}  // namespace tt::foil
