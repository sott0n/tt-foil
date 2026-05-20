// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_foil/kv_cache.hpp"

#include <stdexcept>

#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;

std::size_t KvCache::tiles_per_layer_head() const {
    return static_cast<std::size_t>(max_seq_tiles) * head_dim_tiles;
}

std::size_t KvCache::total_tiles() const {
    return static_cast<std::size_t>(num_layers) * num_kv_heads * tiles_per_layer_head();
}

std::size_t KvCache::tile_offset_bytes(uint32_t layer, uint32_t head,
                                       uint32_t seq_tile, uint32_t dim_tile) const {
    // Layout: [layer][head][seq_tile][dim_tile]
    const std::size_t tile_index =
        ((static_cast<std::size_t>(layer) * num_kv_heads + head) * max_seq_tiles + seq_tile)
            * head_dim_tiles + dim_tile;
    return tile_index * kTileBytes;
}

KvCache allocate_kv_cache(tt::foil::Device& dev,
                          uint32_t num_layers,
                          uint32_t num_kv_heads,
                          uint32_t max_seq_tiles,
                          uint32_t head_dim_tiles) {
    if (num_layers == 0 || num_kv_heads == 0 || max_seq_tiles == 0 || head_dim_tiles == 0)
        throw std::runtime_error("allocate_kv_cache: zero dimension");

    KvCache c;
    c.num_layers      = num_layers;
    c.num_kv_heads    = num_kv_heads;
    c.max_seq_tiles   = max_seq_tiles;
    c.head_dim_tiles  = head_dim_tiles;
    c.current_len_tiles = 0;

    const std::size_t per_buf_bytes = c.total_tiles() * kTileBytes;
    c.k_buf = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, per_buf_bytes);
    c.v_buf = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, per_buf_bytes);
    return c;
}

}  // namespace tt::foil::op_lib
