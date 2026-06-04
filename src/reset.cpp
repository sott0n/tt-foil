// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reset.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "llrt/hal.hpp"
#include "hal/generated/dev_msgs.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/risc_type.hpp>

namespace tt::foil {

namespace {
// Absolute soft-reset register value that parks every Tensix RISC in reset.
// Bit positions come from umd's (now-removed) TensixSoftResetOptions:
//   BRISC=11, TRISC0=12, TRISC1=13, TRISC2=14, NCRISC=18.
// This equals the old TENSIX_ASSERT_SOFT_RESET (no STAGGERED_START bit 31).
constexpr uint32_t kAllTensixAssert =
    (1u << 11) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 18);
}  // namespace

void assert_tensix_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {
    // Do an ABSOLUTE soft-reset write, not Cluster::assert_risc_reset().
    //
    // umd v0.9.6 removed assert_risc_reset_at_core() (which mapped to
    // send_tensix_risc_reset → a single absolute `set_risc_reset_state(core,
    // mask)` with no read). Its replacement Cluster::assert_risc_reset() is
    // read-modify-write: it first READS the soft-reset reg from the core, then
    // ORs in the bits. That read is over NOC and is unreliable for exactly the
    // mid-NOC-transaction cores this per-core unicast assert exists to force
    // into reset — producing intermittent incomplete resets and flaky
    // cross-test hangs. Restore the deterministic absolute write via TTDevice.
    driver.get_tt_device(chip_id)->set_risc_reset_state(core, kAllTensixAssert);
}

void deassert_brisc_reset(
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {
    // umd::Cluster::deassert_risc_reset takes a RiscType bitmask. BRISC alone
    // is the right value on Blackhole; brisc firmware brings up NCRISC and
    // TRISCs from there.
    //
    // tt-metal performs an L1 membar across the chip *before* deasserting,
    // to ensure every Tensix tile sees the firmware writes. Skip the barrier
    // and BRISC on the second-and-later rows can fetch garbage and lock up
    // before reaching its first mailbox write — observed symptom: go.signal
    // stuck at RUN_MSG_INIT (0x40) at translated rows >= y=3.
    driver.l1_membar(chip_id);
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
            // Format sig as a real hex byte. std::to_string(uint8_t) prints in
            // decimal and silently turns 0x40 into "64", which collides with
            // 0x64 to the eye. Annotate the well-known RUN_MSG_* values too so
            // the failure mode is obvious from one log line.
            char buf[16];
            std::snprintf(buf, sizeof(buf), "0x%02x", sig);
            const char* meaning = "unknown";
            switch (sig) {
                case dev_msgs::RUN_MSG_DONE:                  meaning = "RUN_MSG_DONE"; break;
                case dev_msgs::RUN_MSG_INIT:                  meaning = "RUN_MSG_INIT (firmware not yet up)"; break;
                case dev_msgs::RUN_MSG_GO:                    meaning = "RUN_MSG_GO";   break;
                case dev_msgs::RUN_MSG_RESET_READ_PTR:        meaning = "RUN_MSG_RESET_READ_PTR"; break;
                case dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST: meaning = "RUN_MSG_RESET_READ_PTR_FROM_HOST"; break;
                default: break;
            }
            throw std::runtime_error(
                "tt-foil: timeout waiting for firmware init to complete on core "
                "(" + std::to_string(core.x) + "," + std::to_string(core.y) + "); "
                "last go_msg.signal = " + buf + " (" + meaning + ")");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

}  // namespace tt::foil
