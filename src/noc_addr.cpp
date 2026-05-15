// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "noc_addr.hpp"

#include <umd/device/types/core_coordinates.hpp>

namespace tt::foil {

uint64_t make_noc_unicast_addr(
    const tt::umd::CoreCoord& translated_dst,
    uint64_t local_l1_addr) {
    // Constants from blackhole/noc_parameters.h. Hardcoded for Blackhole;
    // tt-foil's plan is BH-only for v2/v3.
    constexpr unsigned kNodeIdBits = 6;
    constexpr unsigned kLocalBits  = 36;
    const uint64_t x = translated_dst.x & ((1ull << kNodeIdBits) - 1);
    const uint64_t y = translated_dst.y & ((1ull << kNodeIdBits) - 1);
    return (y << (kLocalBits + kNodeIdBits)) |
           (x <<  kLocalBits) |
           (local_l1_addr & ((1ull << kLocalBits) - 1));
}

}  // namespace tt::foil
