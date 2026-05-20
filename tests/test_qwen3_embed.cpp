// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Qwen3-VL-2B token-embedding lookup on the real (622 MB)
// embed_tokens table. Verifies the embedding op gathers the right rows
// (bit-exact, no compute drift expected from a pure DRAM→L1 copy path).
//
// Data:
//   data/qwen3_vl_2b/model/embed_tokens.bin   [151936, 2048] bf16
// Produced by:
//   python3 tools/export_qwen3_layer.py --model Qwen/Qwen3-VL-2B-Instruct \
//       --layer none --model-tensors --out-dir data/qwen3_vl_2b

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/ops.hpp"

namespace {

// Qwen3-VL-2B: vocab × hidden.
constexpr uint32_t kV = 151936;
constexpr uint32_t kD = 2048;
// A handful of deterministic token ids spanning the vocab; this matches
// the size we want to chain into a downstream test_qwen3_inference run
// later, but for now we only verify the gather is bit-exact.
constexpr uint32_t kN = 32;

std::vector<uint16_t> load_bin(const std::string& path, std::size_t nelem) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open: " + path);
    std::vector<uint16_t> v(nelem);
    f.read(reinterpret_cast<char*>(v.data()), nelem * 2);
    if (f.gcount() != static_cast<std::streamsize>(nelem * 2))
        throw std::runtime_error("short read: " + path);
    return v;
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    const char* data_env = std::getenv("TT_FOIL_QWEN3_DATA");
    const std::string base = data_env ? data_env : "data/qwen3_vl_2b";

    // -----------------------------------------------------------------
    // 1. Load the 622 MB embed table.
    // -----------------------------------------------------------------
    std::printf("loading embed_tokens [%u, %u] (%lu MB)...\n",
                kV, kD, (std::size_t)kV * kD * 2 / (1024 * 1024));
    auto table = load_bin(base + "/model/embed_tokens.bin",
                          static_cast<std::size_t>(kV) * kD);

    // -----------------------------------------------------------------
    // 2. Deterministic token ids covering low / mid / high parts of the
    //    vocabulary (catches off-by-one slicing in either direction).
    // -----------------------------------------------------------------
    std::vector<uint32_t> token_ids(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        // Hash-ish spread: low half then high half, plus a couple of edges.
        uint32_t h = (i * 9973u + 17u) % kV;
        token_ids[i] = h;
    }
    token_ids[0]      = 0;          // first row
    token_ids[1]      = kV - 1;     // last row
    token_ids[2]      = 151643;     // a common Qwen3 special token (<|endoftext|>)

    std::vector<uint16_t> ref(static_cast<std::size_t>(kN) * kD);
    for (uint32_t r = 0; r < kN; ++r) {
        const uint32_t tok = token_ids[r];
        std::copy(table.data() + static_cast<std::size_t>(tok) * kD,
                  table.data() + static_cast<std::size_t>(tok + 1) * kD,
                  ref.data() + static_cast<std::size_t>(r) * kD);
    }

    // -----------------------------------------------------------------
    // 3. Device run.
    // -----------------------------------------------------------------
    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    ol::TensorDesc T_table;
    T_table.buf = tt::foil::allocate_buffer(
        *dev, tt::foil::BufferLocation::DRAM,
        static_cast<std::size_t>(kV) * kD * 2);
    T_table.num_tiles = 0;  // row-major table, not tile-format
    tt::foil::write_buffer(*dev, *T_table.buf, table.data(),
                           static_cast<std::size_t>(kV) * kD * 2);

    ol::TensorDesc T_out;
    auto op = ol::make_embedding(*dev, T_table, token_ids, kD, T_out);
    ol::execute(*dev, op);

    std::vector<uint16_t> got(static_cast<std::size_t>(kN) * kD, 0);
    tt::foil::read_buffer(*dev, *T_out.buf, got.data(),
                          static_cast<std::size_t>(kN) * kD * 2);

    // -----------------------------------------------------------------
    // 4. Bit-exact compare.
    // -----------------------------------------------------------------
    std::size_t bad = 0;
    std::size_t first_bad = static_cast<std::size_t>(kN) * kD;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            if (first_bad == got.size()) first_bad = i;
            ++bad;
        }
    }

    tt::foil::close_device(std::move(dev));

    if (bad != 0) {
        const uint32_t row = first_bad / kD;
        const uint32_t col = first_bad % kD;
        std::fprintf(stderr,
            "test_qwen3_embed: %zu mismatched bf16 cells "
            "(first at row=%u (token=%u) col=%u: got=0x%04x ref=0x%04x)\n",
            bad, row, token_ids[row], col,
            got[first_bad], ref[first_bad]);
        return 1;
    }
    std::printf("test_qwen3_embed: PASS  (V=%u D=%u N=%u, %zu cells bit-exact)\n",
                kV, kD, kN, got.size());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_qwen3_embed: FAIL — %s\n", e.what());
    return 1;
}
