// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 4b) integration test: init_tensix_core_info_minimal()
// writes absolute_logical_x/y and core_magic_number correctly.
//
// tt-metal's populate_core_info_msg fills in many more fields than we do;
// they're all only used by optional watcher/DPRINT paths. We verify our
// minimal write puts the firmware-critical fields where they belong, then
// restore tt-metal's full snapshot before exit so any other path that
// expects those extras (watcher, dprint) keeps working on this chip.
//
// Usage: TT_FOIL_DEVICE=3 ./test_core_info_init

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "core_info_init.hpp"

int main() try {
    using namespace tt::tt_metal;

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto* dev = CreateDevice(static_cast<tt::ChipId>(pcie_index));
    auto& ctx = MetalContext::instance();
    const auto& hal = ctx.hal();
    auto* driver = ctx.get_cluster().get_driver().get();
    std::puts("test_core_info_init: CreateDevice done");

    constexpr uint32_t kLogX = 0, kLogY = 0;
    auto virt = dev->worker_core_from_logical_core({kLogX, kLogY});
    tt::umd::CoreCoord core{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    std::printf("test_core_info_init: target translated coord (%zu,%zu)\n",
                core.x, core.y);

    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto sample = factory.create<dev_msgs::core_info_msg_t>();
    const std::size_t ci_size = sample.size();
    const uint64_t ci_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::CORE_INFO);
    std::printf("test_core_info_init: CORE_INFO @ 0x%lx, size %zu bytes\n", ci_addr, ci_size);

    // Snapshot tt-metal's fully-populated core_info to restore at end.
    std::vector<std::byte> snapshot(ci_size);
    driver->read_from_device(snapshot.data(), pcie_index, core, ci_addr, ci_size);

    // Call our minimal init.
    tt::foil::init_tensix_core_info_minimal(*driver, hal, pcie_index, core, kLogX, kLogY);
    std::puts("test_core_info_init: init_tensix_core_info_minimal called");

    // Read back; parse via the factory's View.
    auto rb = factory.create<dev_msgs::core_info_msg_t>();
    driver->read_from_device(rb.data(), pcie_index, core, ci_addr, ci_size);
    auto v = rb.view();
    const uint32_t got_x = v.absolute_logical_x();
    const uint32_t got_y = v.absolute_logical_y();
    const auto got_magic = static_cast<uint32_t>(v.core_magic_number());

    bool ok = true;
    if (got_x != kLogX) { std::fprintf(stderr, "x=%u expected %u\n", got_x, kLogX); ok = false; }
    if (got_y != kLogY) { std::fprintf(stderr, "y=%u expected %u\n", got_y, kLogY); ok = false; }
    const auto expected_magic = static_cast<uint32_t>(dev_msgs::CoreMagicNumber::WORKER);
    if (got_magic != expected_magic) {
        std::fprintf(stderr, "magic=0x%08x expected 0x%08x (WORKER)\n", got_magic, expected_magic);
        ok = false;
    }

    if (ok) {
        std::printf("test_core_info_init: abs_logical=(%u,%u) magic=0x%08x ✓\n",
                    got_x, got_y, got_magic);
    }

    // Restore tt-metal's snapshot.
    driver->write_to_device(snapshot.data(), ci_size, pcie_index, core, ci_addr);
    std::puts("test_core_info_init: tt-metal snapshot restored");

    CloseDevice(dev);
    if (!ok) {
        std::puts("test_core_info_init: FAIL");
        return 1;
    }
    std::puts("test_core_info_init: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_core_info_init: FAIL — %s\n", e.what());
    return 1;
}
