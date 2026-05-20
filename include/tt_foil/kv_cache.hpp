// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// KV-Cache infrastructure for autoregressive Transformer decode.
//
// Storage layout (per K or V buffer):
//   [num_layers][num_kv_heads][max_seq_tiles][head_dim_tiles] tiles of BF16
//
// `seq_tile` indexes blocks of 32 contiguous token positions (one tile-row),
// and `dim_tile` indexes 32-column slices of the head dim.  Tiles are stored
// in tt-metal 4-face format (same as the rest of op_lib).
//
// First cut: storage + offset arithmetic + DRAM buffer ownership. Writing /
// reading K and V tiles is the caller's responsibility via the standard
// tt::foil::write_buffer / read_buffer using the offsets returned here. The
// attention kernel <-> cache wiring lands in a follow-up commit.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tt_foil/runtime.hpp"

namespace tt::foil::op_lib {

struct KvCache {
    std::shared_ptr<tt::foil::Buffer> k_buf;
    std::shared_ptr<tt::foil::Buffer> v_buf;

    uint32_t num_layers{0};
    uint32_t num_kv_heads{0};
    uint32_t max_seq_tiles{0};      // tile-rows; each holds 32 tokens
    uint32_t head_dim_tiles{0};     // tile-cols of head_dim

    // Soft state — number of tile-rows currently populated (global across all
    // layers/heads since they advance together during generation).  Callers
    // bump this themselves after writes.
    uint32_t current_len_tiles{0};

    // Byte offset of tile [layer, head, seq_tile, dim_tile] within k_buf
    // (same arithmetic for v_buf). Caller adds this to the Buffer's logical
    // base when issuing write_buffer / read_buffer.
    std::size_t tile_offset_bytes(uint32_t layer, uint32_t head,
                                  uint32_t seq_tile, uint32_t dim_tile) const;

    // Tiles per (layer, head) — i.e. max_seq_tiles * head_dim_tiles.
    std::size_t tiles_per_layer_head() const;

    // Total tiles in each of k_buf / v_buf.
    std::size_t total_tiles() const;
};

// Allocate K and V DRAM buffers sized for the given configuration.
// Buffers are not zero-initialized.
KvCache allocate_kv_cache(tt::foil::Device& dev,
                          uint32_t num_layers,
                          uint32_t num_kv_heads,
                          uint32_t max_seq_tiles,
                          uint32_t head_dim_tiles);

}  // namespace tt::foil::op_lib
