// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 4) integration test: byte-level verification that
// init_tensix_mailboxes() writes the expected boot-time mailbox state.
//
// We can't simply re-init while firmware is running and let it run — RUN_MSG_INIT
// in GO_MSG is benign on an idle core (BRISC polls for RUN_MSG_GO, not INIT,
// so writing INIT is ignored), but we still restore the snapshot at the end
// so the chip is left in tt-metal's idle state. No kernel is launched.
//
// Verifications after init:
//   - launch_msg buffer: all zero (launch_msg_buffer_num_entries * sizeof)
//   - GO_MSG.signal byte == RUN_MSG_INIT (0x40)
//   - LAUNCH_MSG_BUFFER_RD_PTR == 0
//   - GO_MSG_INDEX == 0
//
// Usage: TT_FOIL_DEVICE=3 ./test_mailbox_init

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

#include "mailbox_init.hpp"

int main() try {
    using namespace tt::tt_metal;

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto* dev = CreateDevice(static_cast<tt::ChipId>(pcie_index));
    auto& ctx = MetalContext::instance();
    const auto& hal = ctx.hal();
    auto* driver = ctx.get_cluster().get_driver().get();
    std::puts("test_mailbox_init: CreateDevice done");

    auto virt = dev->worker_core_from_logical_core({0, 0});
    tt::umd::CoreCoord core{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    std::printf("test_mailbox_init: target translated coord (%zu,%zu)\n",
                core.x, core.y);

    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);

    const uint64_t launch_addr  = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH);
    const uint64_t go_addr      = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    const uint64_t rd_ptr_addr  = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    const uint64_t go_idx_addr  = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG_INDEX);

    auto sample_launch  = factory.create<dev_msgs::launch_msg_t>();
    auto sample_go      = factory.create<dev_msgs::go_msg_t>();
    const std::size_t lm_size      = sample_launch.size();
    const std::size_t go_size      = sample_go.size();
    const std::size_t launch_total = lm_size * dev_msgs::launch_msg_buffer_num_entries;

    std::printf("test_mailbox_init: launch=0x%lx (%zu bytes total) go=0x%lx (%zu) rd_ptr=0x%lx go_idx=0x%lx\n",
        launch_addr, launch_total, go_addr, go_size, rd_ptr_addr, go_idx_addr);

    // ---- Snapshot the GO_MSG region so we can restore the firmware's idle state. ----
    std::vector<std::byte> go_snapshot(go_size);
    driver->read_from_device(go_snapshot.data(), pcie_index, core, go_addr, go_size);

    // ---- Call init_tensix_mailboxes ----
    tt::foil::init_tensix_mailboxes(*driver, hal, pcie_index, core);
    std::puts("test_mailbox_init: init_tensix_mailboxes called");

    // ---- Read back launch_msg buffer; expect all zero ----
    std::vector<std::byte> launch_rb(launch_total);
    driver->read_from_device(launch_rb.data(), pcie_index, core, launch_addr, launch_total);
    for (std::size_t i = 0; i < launch_rb.size(); ++i) {
        if (launch_rb[i] != std::byte{0}) {
            std::fprintf(stderr, "launch_msg buf nonzero byte at offset %zu = 0x%02x\n",
                i, static_cast<unsigned>(launch_rb[i]));
            // Restore go_msg before exit to keep chip happy.
            driver->write_to_device(go_snapshot.data(), go_size, pcie_index, core, go_addr);
            return 1;
        }
    }
    std::puts("test_mailbox_init: launch_msg buffer = all zero ✓");

    // ---- Read back GO_MSG, verify signal byte ----
    auto go_rb = factory.create<dev_msgs::go_msg_t>();
    driver->read_from_device(go_rb.data(), pcie_index, core, go_addr, go_size);
    const uint8_t got_signal = go_rb.view().signal();
    if (got_signal != dev_msgs::RUN_MSG_INIT) {
        std::fprintf(stderr, "GO_MSG.signal = 0x%02x, expected 0x%02x (RUN_MSG_INIT)\n",
            got_signal, dev_msgs::RUN_MSG_INIT);
        driver->write_to_device(go_snapshot.data(), go_size, pcie_index, core, go_addr);
        return 1;
    }
    std::printf("test_mailbox_init: GO_MSG.signal = 0x%02x (RUN_MSG_INIT) ✓\n", got_signal);

    // ---- Read back rd_ptr and go_msg_index ----
    uint32_t rd_ptr = 0xFFFFFFFFu, go_idx = 0xFFFFFFFFu;
    driver->read_from_device(&rd_ptr, pcie_index, core, rd_ptr_addr, sizeof(rd_ptr));
    driver->read_from_device(&go_idx, pcie_index, core, go_idx_addr, sizeof(go_idx));
    if (rd_ptr != 0 || go_idx != 0) {
        std::fprintf(stderr, "rd_ptr=%u go_idx=%u, expected 0/0\n", rd_ptr, go_idx);
        driver->write_to_device(go_snapshot.data(), go_size, pcie_index, core, go_addr);
        return 1;
    }
    std::puts("test_mailbox_init: rd_ptr = 0, go_msg_index = 0 ✓");

    // ---- Restore go_msg so the firmware sees RUN_MSG_DONE again ----
    driver->write_to_device(go_snapshot.data(), go_size, pcie_index, core, go_addr);
    std::puts("test_mailbox_init: GO_MSG restored");

    CloseDevice(dev);
    std::puts("test_mailbox_init: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_mailbox_init: FAIL — %s\n", e.what());
    return 1;
}
