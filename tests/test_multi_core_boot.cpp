// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v3-1: confirm device_open boots multiple Tensix cores cold.
//
// Boots logical (0,0) and (0,1) in one call, then sanity-checks each by
// reading back GO_MSG.signal directly via UMD — both should be in
// RUN_MSG_DONE after BRISC firmware finished its init sequence.
//
// Usage: TT_FOIL_DEVICE=3 ./test_multi_core_boot

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tt_foil/runtime.hpp"

// For peeking at GO_MSG via UMD — keep this test honest about what
// "booted" means without relying on subsequent kernel launches.
#include "device.hpp"
#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(
        pcie_index,
        /*firmware_dir=*/"",
        /*cores=*/{{0, 0}, {0, 1}});
    std::puts("test_multi_core_boot: device_open returned for 2 cores");

    const auto& hal = *dev->hal;
    auto& driver   = *dev->umd_driver;
    const auto& soc_desc = driver.get_soc_descriptor(dev->chip_id);

    const uint64_t go_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        tt::tt_metal::HalL1MemAddrType::GO_MSG);

    bool ok = true;
    for (const auto& logical : dev->booted_cores) {
        tt::umd::CoreCoord lc{logical.x, logical.y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
        auto core = soc_desc.translate_coord_to(lc, tt::CoordSystem::TRANSLATED);

        auto go = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::TENSIX)
                     .create<tt::tt_metal::dev_msgs::go_msg_t>();
        driver.read_from_device(go.data(), dev->chip_id, core, go_addr, go.size());
        uint8_t sig = go.view().signal();
        std::printf("  logical (%u,%u) translated (%zu,%zu) GO_MSG.signal=0x%02x\n",
            logical.x, logical.y, core.x, core.y, sig);
        if (sig != tt::tt_metal::dev_msgs::RUN_MSG_DONE) {
            std::fprintf(stderr, "  expected RUN_MSG_DONE (0x%02x), got 0x%02x\n",
                tt::tt_metal::dev_msgs::RUN_MSG_DONE, sig);
            ok = false;
        }
    }

    tt::foil::close_device(std::move(dev));

    if (!ok) { std::puts("test_multi_core_boot: FAIL"); return 1; }
    std::puts("test_multi_core_boot: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_multi_core_boot: FAIL — %s\n", e.what());
    return 1;
}
