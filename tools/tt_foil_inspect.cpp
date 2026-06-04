// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tt_foil_inspect: open UMD on TT_FOIL_DEVICE (chip id), read soft-reset
// state + L1[0]/L1[0x38E0] of every Tensix translated core, print one
// line per core. Does NOT run init / firmware load / deassert. Used to
// inspect chip state after a degrade event without triggering further
// state changes.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <umd/device/cluster.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/risc_type.hpp>

int main() try {
    const char* env = std::getenv("TT_FOIL_DEVICE");
    int pci_idx = env ? std::stoi(env) : 0;

    tt::umd::ClusterOptions opts;
    opts.target_devices = std::unordered_set<int>{pci_idx};
    tt::umd::Cluster cluster(opts);

    // Translate "PCI index" to UMD chip id (assume same here).
    uint32_t chip = static_cast<uint32_t>(pci_idx);

    const auto& soc = cluster.get_soc_descriptor(chip);
    auto grid = soc.get_grid_size(tt::CoreType::TENSIX);
    std::printf("chip=%u tensix grid=%zux%zu\n", chip, grid.x, grid.y);

    // Optional: TT_FOIL_INSPECT_PARK=1 issues a per-core assert_risc_reset_at_core
    // on every Tensix core before reading state. Lets us see whether a directed
    // per-core assert can park a core that broadcast-assert couldn't.
    bool park = std::getenv("TT_FOIL_INSPECT_PARK") != nullptr;
    if (park) {
        std::printf("[park] issuing per-core assert_risc_reset_at_core on all Tensix\n");
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                try {
                    tt::umd::CoreCoord lc{x, y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
                    auto t = soc.translate_coord_to(lc, tt::CoordSystem::TRANSLATED);
                    // Absolute soft-reset write (BRISC|NCRISC|TRISC0/1/2),
                    // not the RMW Cluster::assert_risc_reset — see reset.cpp.
                    cluster.get_tt_device(chip)->set_risc_reset_state(
                        t, (1u << 11) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 18));
                } catch (...) {}
            }
        }
        cluster.l1_membar(chip);
    }

    // TT_FOIL_INSPECT_BCAST=1: issue broadcast assert_risc_reset() and probe
    // whether (1,2) (= logical (0,0)) parks. Mimics what our open_device
    // does first; if (1,2) ends 0xf7 after this, the broadcast is being
    // ignored by that core — that's the "stuck" state we are hunting.
    // TT_FOIL_INSPECT_NOC=1: read NOC NIU outstanding counters at logical
    // (0,0) and (0,1). Smoking gun for symptom-B-style "kernel hangs in
    // noc_async_write_barrier" because previous test left HW counters in
    // a non-resettable state.
    if (std::getenv("TT_FOIL_INSPECT_NOC") != nullptr) {
        struct R { const char* name; uint64_t addr; };
        for (uint32_t y = 0; y < 2; ++y) {
            tt::umd::CoreCoord lc{0u, y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
            auto t = soc.translate_coord_to(lc, tt::CoordSystem::TRANSLATED);
            std::printf("--- NOC counters at logical (0,%u) translated (%zu,%zu) ---\n",
                        y, t.x, t.y);
            R regs[] = {
                {"NOC0 NONPOSTED_WR_SENT  ", 0xFFB20228ull},
                {"NOC0 WR_ACK_RECEIVED    ", 0xFFB20204ull},
                {"NOC0 OUTSTANDING_ID(0)  ", 0xFFB20240ull},
                {"NOC0 RD_RESP_RECEIVED   ", 0xFFB20208ull},
                {"NOC1 NONPOSTED_WR_SENT  ", 0xFFB30228ull},
                {"NOC1 WR_ACK_RECEIVED    ", 0xFFB30204ull},
                {"NOC1 OUTSTANDING_ID(0)  ", 0xFFB30240ull},
                {"NOC1 RD_RESP_RECEIVED   ", 0xFFB30208ull},
            };
            for (const auto& r : regs) {
                uint32_t v = 0;
                try {
                    cluster.read_from_device(&v, chip, t, r.addr, sizeof(v));
                    std::printf("  %s @ 0x%lx = 0x%08x  (%u)\n", r.name, r.addr, v, v);
                } catch (...) {
                    std::printf("  %s @ 0x%lx = read failed\n", r.name, r.addr);
                }
            }
        }
    }
    bool bcast = std::getenv("TT_FOIL_INSPECT_BCAST") != nullptr;
    if (bcast) {
        std::printf("[bcast] issuing broadcast assert_risc_reset()\n");
        cluster.assert_risc_reset();
        cluster.l1_membar(chip);
    }

    std::printf("%-10s %-12s %-12s %-16s %-12s\n",
                "logical", "translated", "soft_reset", "L1[0]", "L1[0x38E0]");

    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            try {
                tt::umd::CoreCoord lc{x, y, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL};
                auto t = soc.translate_coord_to(lc, tt::CoordSystem::TRANSLATED);
                auto rs = cluster.get_risc_reset_state(chip, t);
                uint32_t l1_0   = 0xdeadbeef;
                uint32_t l1_fw  = 0xdeadbeef;
                cluster.read_from_device(&l1_0,  chip, t, 0x0,    sizeof(l1_0));
                cluster.read_from_device(&l1_fw, chip, t, 0x38E0, sizeof(l1_fw));
                char lbuf[16], tbuf[16];
                std::snprintf(lbuf, sizeof(lbuf), "(%u,%u)", x, y);
                std::snprintf(tbuf, sizeof(tbuf), "(%zu,%zu)", t.x, t.y);
                std::printf("%-10s %-12s 0x%02x         0x%08x      0x%08x\n",
                            lbuf, tbuf,
                            static_cast<uint32_t>(rs), l1_0, l1_fw);
            } catch (const std::exception& e) {
                std::printf("(%u,%u) THROW: %s\n", x, y, e.what());
            }
        }
    }
    return 0;
} catch (const std::exception& ex) {
    std::fprintf(stderr, "tt_foil_inspect FAIL: %s\n", ex.what());
    return 1;
}
