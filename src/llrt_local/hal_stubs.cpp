// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B3-4 stubs for non-Blackhole HAL initialisers.
//
// tt_metal/llrt/hal.cpp's Hal::Hal() switches on tt::ARCH and calls one of
// initialize_wh / initialize_qa / initialize_bh. tt-foil only targets
// Blackhole, but the references to the other two are still emitted by the
// compiler, so the linker needs a definition. We provide a trivial throw —
// these paths are unreachable for our supported arches.

#include <stdexcept>

#include "llrt/hal.hpp"

namespace tt::tt_metal {

void Hal::initialize_wh(
    bool /*is_base_routing_fw_enabled*/,
    uint32_t /*profiler_dram_bank_size_per_risc_bytes*/,
    bool /*enable_dram_backed_cq*/) {
    throw std::runtime_error(
        "tt-foil: Wormhole HAL is not built in. Set ARCH_BLACKHOLE only.");
}

void Hal::initialize_qa(
    uint32_t /*profiler_dram_bank_size_per_risc_bytes*/,
    bool /*enable_dram_backed_cq*/) {
    throw std::runtime_error(
        "tt-foil: Quasar HAL is not built in. Set ARCH_BLACKHOLE only.");
}

}  // namespace tt::tt_metal
