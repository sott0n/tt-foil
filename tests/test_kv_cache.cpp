// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HW test: KV cache infrastructure (Phase 1-C minimum scope).
// Verifies:
//   - allocate_kv_cache sizes k_buf / v_buf for the requested geometry
//   - tile_offset_bytes is monotone and covers exactly the allocated range
//   - Host can write a deterministic pattern into K and V independently,
//     read it back, and recover each tile at the right offset.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"
#include "tt_foil/kv_cache.hpp"

namespace {

constexpr uint32_t kNumLayers     = 2;
constexpr uint32_t kNumKvHeads    = 2;
constexpr uint32_t kMaxSeqTiles   = 4;   // = 128 token positions
constexpr uint32_t kHeadDimTiles  = 2;   // head_dim = 64
constexpr uint32_t kTileBytes     = 32 * 32 * 2;
constexpr uint32_t kTileWords     = kTileBytes / 2;

// 16-bit pattern: bit 15 = kv_select, bits 14..8 = tile_index (7 bits, ≤127),
// bits 7..0 = word index within the tile. Distinct (kv_select, tile_index,
// word_idx) triples → distinct words.
uint16_t pattern_word(uint32_t kv_select, std::size_t tile_index, uint32_t word_idx) {
    uint32_t v = ((kv_select & 1u) << 15)
               | ((static_cast<uint32_t>(tile_index) & 0x7fu) << 8)
               | (word_idx & 0xffu);
    return static_cast<uint16_t>(v);
}

void fill_tile(std::vector<uint16_t>& dst, std::size_t tile_index,
               uint32_t kv_select, std::size_t base_word) {
    for (uint32_t i = 0; i < kTileWords; ++i)
        dst[base_word + i] = pattern_word(kv_select, tile_index, i);
}

}  // namespace

int main() try {
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    namespace ol = tt::foil::op_lib;

    auto cache = ol::allocate_kv_cache(*dev,
                                       kNumLayers, kNumKvHeads,
                                       kMaxSeqTiles, kHeadDimTiles);

    // --- Offset sanity ---
    if (cache.tile_offset_bytes(0, 0, 0, 0) != 0)
        throw std::runtime_error("tile_offset_bytes(0,0,0,0) != 0");
    const std::size_t expected_total =
        static_cast<std::size_t>(kNumLayers) * kNumKvHeads *
        kMaxSeqTiles * kHeadDimTiles * kTileBytes;
    if (cache.k_buf->size_bytes != expected_total ||
        cache.v_buf->size_bytes != expected_total)
        throw std::runtime_error("KV buffer size mismatch");
    const std::size_t last = cache.tile_offset_bytes(
        kNumLayers - 1, kNumKvHeads - 1, kMaxSeqTiles - 1, kHeadDimTiles - 1);
    if (last + kTileBytes != expected_total)
        throw std::runtime_error("last tile offset != total - kTileBytes");

    // --- Round-trip pattern test ---
    const std::size_t total_words = expected_total / 2;
    std::vector<uint16_t> k_host(total_words, 0);
    std::vector<uint16_t> v_host(total_words, 0);

    for (uint32_t L = 0; L < kNumLayers; ++L)
        for (uint32_t H = 0; H < kNumKvHeads; ++H)
            for (uint32_t S = 0; S < kMaxSeqTiles; ++S)
                for (uint32_t D = 0; D < kHeadDimTiles; ++D) {
                    const std::size_t byte_off  = cache.tile_offset_bytes(L, H, S, D);
                    const std::size_t word_off  = byte_off / 2;
                    const std::size_t tile_idx  =
                        ((L * kNumKvHeads + H) * kMaxSeqTiles + S) * kHeadDimTiles + D;
                    fill_tile(k_host, tile_idx, /*kv*/0, word_off);
                    fill_tile(v_host, tile_idx, /*kv*/1, word_off);
                }

    tt::foil::write_buffer(*dev, *cache.k_buf, k_host.data(), expected_total);
    tt::foil::write_buffer(*dev, *cache.v_buf, v_host.data(), expected_total);

    std::vector<uint16_t> k_got(total_words, 0xDEAD);
    std::vector<uint16_t> v_got(total_words, 0xDEAD);
    tt::foil::read_buffer(*dev, *cache.k_buf, k_got.data(), expected_total);
    tt::foil::read_buffer(*dev, *cache.v_buf, v_got.data(), expected_total);

    uint32_t k_bad = 0, v_bad = 0;
    for (std::size_t i = 0; i < total_words; ++i) {
        if (k_got[i] != k_host[i]) ++k_bad;
        if (v_got[i] != v_host[i]) ++v_bad;
    }
    if (k_bad != 0 || v_bad != 0) {
        std::fprintf(stderr, "test_kv_cache: k_bad=%u v_bad=%u (out of %zu)\n",
                     k_bad, v_bad, total_words);
        std::fprintf(stderr, "test_kv_cache: FAIL\n");
        tt::foil::close_device(std::move(dev));
        return 1;
    }

    // --- Sanity: K and V do not alias (different first-word pattern) ---
    if (k_got[0] == v_got[0])
        throw std::runtime_error("K and V appear to alias (pattern collision)");

    std::printf("test_kv_cache: PASS  (L=%u H=%u S=%u D=%u → %zu B per buf)\n",
                kNumLayers, kNumKvHeads, kMaxSeqTiles, kHeadDimTiles, expected_total);
    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_kv_cache: FAIL — %s\n", e.what());
    return 1;
}
