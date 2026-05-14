// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 5) integration test: zero_fill_bank_tables() writes all
// zeros to BANK_TO_NOC_SCRATCH and LOGICAL_TO_VIRTUAL_SCRATCH.
//
// CreateDevice populates real values into these regions; we snapshot,
// overwrite with our zero-fill, verify all-zero, restore the snapshot.
// Restoring matters because any kernel using bank-interleaved addressing
// or get_noc_addr_from_logical_xy() on this chip needs the proper tables.
//
// Usage: TT_FOIL_DEVICE=3 ./test_bank_tables_init

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

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "bank_tables_init.hpp"

namespace {

bool verify_zero(
    const char* name,
    tt::umd::Cluster& driver,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core,
    uint64_t addr,
    uint32_t size) {
    std::vector<std::byte> rb(size);
    driver.read_from_device(rb.data(), chip_id, core, addr, size);
    for (std::size_t i = 0; i < rb.size(); ++i) {
        if (rb[i] != std::byte{0}) {
            std::fprintf(stderr, "%s: nonzero byte at offset %zu = 0x%02x\n",
                name, i, static_cast<unsigned>(rb[i]));
            return false;
        }
    }
    std::printf("%s: %u bytes all zero ✓\n", name, size);
    return true;
}

}  // namespace

int main() try {
    using namespace tt::tt_metal;

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto* dev = CreateDevice(static_cast<tt::ChipId>(pcie_index));
    auto& ctx = MetalContext::instance();
    const auto& hal = ctx.hal();
    auto* driver = ctx.get_cluster().get_driver().get();
    std::puts("test_bank_tables_init: CreateDevice done");

    auto virt = dev->worker_core_from_logical_core({0, 0});
    tt::umd::CoreCoord core{
        virt.x, virt.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};

    const uint64_t bank_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t bank_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint64_t l2v_addr  = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t l2v_size  = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    std::printf("test_bank_tables_init: BANK_TO_NOC_SCRATCH=0x%lx (%u B)\n", bank_addr, bank_size);
    std::printf("test_bank_tables_init: LOGICAL_TO_VIRTUAL_SCRATCH=0x%lx (%u B)\n", l2v_addr, l2v_size);

    // Snapshot tt-metal's populated tables so we can restore them.
    std::vector<std::byte> bank_snap(bank_size), l2v_snap(l2v_size);
    driver->read_from_device(bank_snap.data(), pcie_index, core, bank_addr, bank_size);
    driver->read_from_device(l2v_snap.data(),  pcie_index, core, l2v_addr,  l2v_size);

    tt::foil::zero_fill_bank_tables(*driver, hal, pcie_index, core);
    std::puts("test_bank_tables_init: zero_fill_bank_tables called");

    bool ok = true;
    ok &= verify_zero("BANK_TO_NOC_SCRATCH",        *driver, pcie_index, core, bank_addr, bank_size);
    ok &= verify_zero("LOGICAL_TO_VIRTUAL_SCRATCH", *driver, pcie_index, core, l2v_addr,  l2v_size);

    // Restore tt-metal's tables.
    driver->write_to_device(bank_snap.data(), bank_size, pcie_index, core, bank_addr);
    driver->write_to_device(l2v_snap.data(),  l2v_size,  pcie_index, core, l2v_addr);
    std::puts("test_bank_tables_init: tt-metal snapshots restored");

    CloseDevice(dev);
    if (!ok) { std::puts("test_bank_tables_init: FAIL"); return 1; }
    std::puts("test_bank_tables_init: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_bank_tables_init: FAIL — %s\n", e.what());
    return 1;
}
