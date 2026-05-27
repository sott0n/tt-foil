// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bank_tables_init.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

namespace {

// Blackhole DRAM YAML (`blackhole_140_arch.yaml`) declares
// `worker_endpoint: [2, 1]` — the per-DRAM-channel subchannel index
// used as the preferred worker-side NOC endpoint, indexed by NOC id.
// Mirrors device.cpp's existing `kBhDramCh0NocZeroSubchannel = 2`.
constexpr int kBhDramSubchannelForNoc[2] = {2, 1};

// Blackhole bank counts. Match the -D flags in scripts/build_firmware.sh
// (NUM_DRAM_BANKS=8, NUM_L1_BANKS=140). The latter is unused in tt-foil
// today (no L1-interleaved buffers) but the firmware still copies the
// full blob, so we must reserve the slot.
constexpr uint32_t kNumDramBanks = 8;
constexpr uint32_t kNumL1Banks   = 140;

}  // namespace

void init_bank_tables(
    tt::umd::Cluster& driver,
    const tt::tt_metal::Hal& hal,
    uint32_t chip_id,
    const tt::umd::CoreCoord& core) {

    using namespace tt::tt_metal;

    const uint64_t bank_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t bank_size = hal.get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    const uint32_t num_nocs = hal.get_num_nocs();
    const uint32_t node_id_bits = hal.get_noc_addr_node_id_bits();
    const uint32_t reg_offset = hal.get_noc_coord_reg_offset();

    // Layout (matches firmware_common.h::noc_bank_table_init copy order):
    //   [0]                                   dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS]   uint16
    //   [dram_xy_bytes]                       l1_bank_to_noc_xy [NUM_NOCS][NUM_L1_BANKS]      uint16
    //   [+l1_xy_bytes]                        bank_to_dram_offset[NUM_DRAM_BANKS]             int32
    //   [+dram_off_bytes]                     bank_to_l1_offset [NUM_L1_BANKS]                int32
    //
    // We populate the DRAM xy table; everything else stays zero.
    //  - L1 xy table: tt-foil has no L1-interleaved buffer path.
    //  - bank_to_{dram,l1}_offset: per-bank shift. Our DRAM buffers
    //    use the same logical base across all 8 channels (each bank
    //    has its own address space starting at 0), so a zero offset
    //    matches what the user passes as the buffer base address.

    const uint32_t dram_xy_bytes = num_nocs * kNumDramBanks * sizeof(uint16_t);
    const uint32_t l1_xy_bytes   = num_nocs * kNumL1Banks   * sizeof(uint16_t);
    const uint32_t dram_off_bytes = kNumDramBanks * sizeof(int32_t);
    const uint32_t l1_off_bytes   = kNumL1Banks   * sizeof(int32_t);
    const uint32_t total_bytes = dram_xy_bytes + l1_xy_bytes + dram_off_bytes + l1_off_bytes;
    if (total_bytes > bank_size) {
        throw std::runtime_error(
            "init_bank_tables: blob exceeds BANK_TO_NOC_SCRATCH capacity");
    }

    std::vector<std::byte> blob(bank_size, std::byte{0});
    auto* dram_xy = reinterpret_cast<uint16_t*>(blob.data());

    const auto& soc_desc = driver.get_soc_descriptor(chip_id);
    for (uint32_t noc = 0; noc < num_nocs; ++noc) {
        const int subchannel = kBhDramSubchannelForNoc[noc];
        for (uint32_t bank = 0; bank < kNumDramBanks; ++bank) {
            const tt::umd::CoreCoord c = soc_desc.get_dram_core_for_channel(
                static_cast<int>(bank), subchannel, tt::CoordSystem::TRANSLATED);
            const uint16_t x = static_cast<uint16_t>(c.x);
            const uint16_t y = static_cast<uint16_t>(c.y);
            const uint16_t xy =
                static_cast<uint16_t>(((y << node_id_bits) | x) << reg_offset);
            dram_xy[noc * kNumDramBanks + bank] = xy;
        }
    }

    driver.write_to_device(blob.data(), blob.size(), chip_id, core, bank_addr);

    // LOGICAL_TO_VIRTUAL scratch: kernels in tt-foil get TRANSLATED
    // coords from the host via RTA (see make_noc_unicast_addr) and
    // never call get_noc_addr_from_logical_xy(), so zero-fill is still
    // correctness-safe here.
    const uint64_t l2v_addr = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t l2v_size = hal.get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    if (l2v_size > 0) {
        std::vector<std::byte> zeros(l2v_size, std::byte{0});
        driver.write_to_device(zeros.data(), zeros.size(), chip_id, core, l2v_addr);
    }
}

}  // namespace tt::foil
