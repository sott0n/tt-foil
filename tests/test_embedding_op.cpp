// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Embedding via op_lib.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"
#include "tile_utils.hpp"

namespace {
using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;
constexpr uint32_t kV = 256, kD = 32, kN = 32;
}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    std::vector<uint16_t> table(kV * kD);
    for (uint32_t t = 0; t < kV; ++t)
        for (uint32_t c = 0; c < kD; ++c)
            table[t * kD + c] = f32_to_bf16(0.01f * t + 0.001f * c - 1.0f);

    std::vector<uint32_t> token_ids(kN);
    for (uint32_t i = 0; i < kN; ++i) token_ids[i] = (i * 7 + 13) % kV;

    std::vector<uint16_t> ref(kN * kD);
    for (uint32_t r = 0; r < kN; ++r)
        for (uint32_t c = 0; c < kD; ++c)
            ref[r * kD + c] = table[token_ids[r] * kD + c];

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    ol::TensorDesc table_td;
    table_td.buf = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kV * kD * 2);
    table_td.num_tiles = 0;  // row-major table, not tile-format
    tt::foil::write_buffer(*dev, *table_td.buf, table.data(), kV * kD * 2);

    ol::TensorDesc out;
    auto op = ol::make_embedding(*dev, table_td, token_ids, kD, out);
    ol::execute(*dev, op);

    std::vector<uint16_t> got(kN * kD, 0);
    tt::foil::read_buffer(*dev, *out.buf, got.data(), kN * kD * 2);

    uint32_t bad = 0;
    for (uint32_t i = 0; i < kN * kD; ++i) if (got[i] != ref[i]) ++bad;
    if (bad != 0) {
        std::fprintf(stderr, "test_embedding_op: %u bad\n", bad);
        std::fprintf(stderr, "test_embedding_op: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }
    std::printf("test_embedding_op: PASS  (V=%u D=%u N=%u)\n", kV, kD, kN);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_embedding_op: FAIL — %s\n", e.what());
    return 1;
}
