// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: Embedding lookup (BRISC + NCRISC only, no TRISC).
//
// Gathers N rows from a [V, D] BF16 embedding table in DRAM by token id,
// writes [N, D] row-major BF16 to a DRAM destination.
//
// First cut: V=256, D=32, N=32 → 1 tile worth of output (row-major, not
// tile-format).

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "cb_config.hpp"
#include "tile_utils.hpp"

namespace {

using tt::foil::test::f32_to_bf16;
using tt::foil::test::bf16_to_f32;

constexpr uint32_t kV = 256;   // vocab size
constexpr uint32_t kD = 32;    // embedding dim
constexpr uint32_t kN = 32;    // tokens per batch
constexpr uint32_t kDBytes     = kD * 2;          // BF16 row size
constexpr uint32_t kTableBytes = kV * kDBytes;
constexpr uint32_t kOutBytes   = kN * kDBytes;

std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

}  // namespace

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    // Build embedding table — deterministic per-(token, col) values.
    std::vector<uint16_t> table(kV * kD);
    for (uint32_t t = 0; t < kV; ++t) {
        for (uint32_t c = 0; c < kD; ++c) {
            float v = 0.01f * static_cast<float>(t) + 0.001f * static_cast<float>(c) - 1.0f;
            table[t * kD + c] = f32_to_bf16(v);
        }
    }

    // Pick N deterministic token ids (a mix to exercise non-trivial ordering).
    std::vector<uint32_t> token_ids(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        token_ids[i] = (i * 7 + 13) % kV;
    }

    // Host reference: gather rows row-major.
    std::vector<uint16_t> ref(kN * kD);
    for (uint32_t r = 0; r < kN; ++r) {
        for (uint32_t c = 0; c < kD; ++c) {
            ref[r * kD + c] = table[token_ids[r] * kD + c];
        }
    }

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    auto buf_table = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kTableBytes, core);
    auto buf_out   = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::DRAM, kOutBytes,   core);
    auto l1_out    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1,   kOutBytes,   core);

    tt::foil::write_buffer(*dev, *buf_table, table.data(), kTableBytes);
    {
        std::vector<uint8_t> zero(kOutBytes, 0);
        tt::foil::write_buffer(*dev, *buf_out, zero.data(), kOutBytes);
    }

    uint64_t emb_noc = tt::foil::make_noc_dram_addr(*dev, buf_table->device_addr);
    uint64_t dst_noc = tt::foil::make_noc_dram_addr(*dev, buf_out->device_addr);

    using R = tt::foil::RiscBinary;
    std::array<R, 2> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // CB 16 holds the gathered [N, D] block (one slot, page = full payload).
    std::array<tt::foil::CbConfig, 1> cbs = {{
        {16, l1_out->device_addr, kOutBytes, 1, kOutBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    // Reader args: emb_noc(2), N, D_bytes, token_ids[N]
    std::vector<uint32_t> ra_brisc;
    ra_brisc.reserve(4 + kN);
    ra_brisc.push_back(static_cast<uint32_t>(emb_noc & 0xffffffffu));
    ra_brisc.push_back(static_cast<uint32_t>(emb_noc >> 32));
    ra_brisc.push_back(kN);
    ra_brisc.push_back(kDBytes);
    for (uint32_t i = 0; i < kN; ++i) ra_brisc.push_back(token_ids[i]);

    std::array<uint32_t, 3> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc & 0xffffffffu),
        static_cast<uint32_t>(dst_noc >> 32),
        kOutBytes,
    };

    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

    std::vector<uint16_t> out(kN * kD, 0);
    tt::foil::read_buffer(*dev, *buf_out, out.data(), kOutBytes);

    uint32_t bad = 0, first_bad = kN * kD;
    for (uint32_t i = 0; i < kN * kD; ++i) {
        if (out[i] != ref[i]) {
            if (first_bad == kN * kD) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        uint32_t r = first_bad / kD;
        uint32_t c = first_bad % kD;
        std::fprintf(stderr,
            "test_embedding: %u/%u mismatches; first at row=%u col=%u (token=%u): "
            "got=0x%04x (%.4f) expected=0x%04x (%.4f)\n",
            bad, kN * kD, r, c, token_ids[r],
            out[first_bad], bf16_to_f32(out[first_bad]),
            ref[first_bad], bf16_to_f32(ref[first_bad]));
        tt::foil::close_device(std::move(dev));
        std::fprintf(stderr, "test_embedding: FAIL\n");
        return 1;
    }

    std::printf("test_embedding: PASS  (V=%u D=%u N=%u; row-major output)\n", kV, kD, kN);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_embedding: FAIL — %s\n", e.what());
    return 1;
}
