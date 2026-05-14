// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B2 (step 2) integration test: open a UMD Cluster directly — no
// tt-metal CreateDevice anywhere — and exercise a host<->L1 round trip.
//
// Cores remain in reset (no firmware is loaded yet), but UMD reads/writes go
// directly via PCIe NOC and don't need a running RISC. We pick a Tensix
// worker core's L1 (logical (0,0) -> translated coord from soc_desc) and
// confirm a write+read round trip preserves data.
//
// Usage:
//   TT_FOIL_DEVICE=3 ./test_umd_open
//
// Must NOT run alongside another process that has the same chip open.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "umd_boot.hpp"

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto h = tt::foil::umd_open(pcie_index);
    std::printf("test_umd_open: UMD cluster opened (chip_id=%u)\n", h.chip_id);

    const auto& soc = h.cluster->get_soc_descriptor(h.chip_id);
    auto tensix_cores = soc.get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED);
    if (tensix_cores.empty()) {
        throw std::runtime_error("no Tensix cores reported by soc descriptor");
    }
    auto target = tensix_cores.front();
    std::printf("test_umd_open: Tensix translated coord (%zu,%zu)\n", target.x, target.y);

    // Pick a scratch address well above any boot mailbox area. L1 base + 256KB
    // is comfortably inside Tensix L1 (1.5MB) and below any firmware ranges.
    constexpr uint64_t kScratchAddr = 256 * 1024;

    std::vector<uint32_t> tx(8), rx(8, 0);
    for (size_t i = 0; i < tx.size(); ++i) tx[i] = 0xCAFE0000u + static_cast<uint32_t>(i);

    h.cluster->write_to_device(tx.data(), tx.size() * sizeof(uint32_t),
                               h.chip_id, target, kScratchAddr);
    h.cluster->read_from_device(rx.data(), h.chip_id, target, kScratchAddr,
                                rx.size() * sizeof(uint32_t));

    for (size_t i = 0; i < tx.size(); ++i) {
        if (rx[i] != tx[i]) {
            std::fprintf(stderr,
                "L1 round-trip mismatch at word %zu: wrote 0x%08x read 0x%08x\n",
                i, tx[i], rx[i]);
            return 1;
        }
    }
    std::puts("test_umd_open: host<->L1 round trip OK");
    std::puts("test_umd_open: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_umd_open: FAIL — %s\n", e.what());
    return 1;
}
