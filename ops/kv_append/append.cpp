// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BRISC kernel for KvAppend. Replaces qwen3_run's host-side
// `dec:kv_slot1_rebuild` step with an entirely-device-side append.
//
// Decode KV-cache state we're updating, per layer per step:
//   T_Kt_cache  — K^T tile-format, shape (Mt=Nk, Nt=StKv) so
//                 tile(c, slot) at offset (c*StKv + slot)*2048. Each
//                 tile spans NK rows c*32..c*32+31 by seq cols
//                 slot*32..slot*32+31.
//   T_V_cache   — V tile-format, shape (Mt=StKv, Nt=Nk) so
//                 tile(slot, c) at offset (slot*Nk + c)*2048.
//
// Append plan for a new position pos with slot1_r = pos - kS (slot=1):
//
//   V:  per NK-block c, dest tile (1, c). We write *row slot1_r* of
//       this tile. That row spans 32 cols = the 32 NK values for the
//       new pos within block c. In face layout the row crosses two
//       faces (left-cols and right-cols). Each face's row is 32 bytes
//       contiguous → 2 plain noc_async_write per NK-block. No
//       read-modify-write needed (other rows hold earlier slot1_r's
//       data and we don't touch them).
//
//   K:  per NK-block c, dest tile (c, 1). We write *col slot1_r* of
//       this tile. That col spans 32 rows = the 32 NK values for the
//       new pos within block c, but laid out at a 32-byte stride
//       inside each face (one bf16 per row). 2-byte NOC writes don't
//       work, so we read the WHOLE face (512 bytes contiguous),
//       patch one bf16 per row in L1, write it back. 2 RMW per tile.
//
// RTAs:
//   arg[ 0.. 1] = T_Kr NOC (lo, hi)
//   arg[ 2.. 3] = T_V  NOC (lo, hi)
//   arg[ 4.. 5] = T_Kt_cache base NOC (lo, hi)
//   arg[ 6.. 7] = T_V_cache  base NOC (lo, hi)
//   arg[    8 ] = slot1_r  (0..StKv*16-1)
//   arg[    9 ] = Nk       (column-tile count, = kNkDt)
//   arg[   10 ] = StKv     (row-tile count for V / col-tile count for K, = kStKvDec)

#include <cstdint>
#include "dataflow_api.h"

static inline uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

constexpr uint32_t cb_io          = 16;
constexpr uint32_t kTileBytes     = 2048;
constexpr uint32_t kFaceBytes     = 512;
constexpr uint32_t kFaceRowBytes  = 32;  // 16 bf16 cols per face row
constexpr uint8_t  kWriteNoc      = 1;   // BRISC peer/DRAM writes go through NOC 1.

void kernel_main() {
    const uint64_t kr_noc  = join64(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint64_t v_noc   = join64(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const uint64_t kt_base = join64(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const uint64_t v_base  = join64(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    const uint32_t slot1_r = get_arg_val<uint32_t>(8);
    const uint32_t Nk      = get_arg_val<uint32_t>(9);
    const uint32_t StKv    = get_arg_val<uint32_t>(10);

    cb_reserve_back(cb_io, 1);
    const uint32_t l1_base   = get_write_ptr(cb_io);
    const uint32_t scratch_K = l1_base;
    const uint32_t scratch_V = l1_base + Nk * kTileBytes;
    const uint32_t face_buf  = l1_base + 2 * Nk * kTileBytes;

    // Slurp both source tensors into L1.
    noc_async_read(kr_noc, scratch_K, Nk * kTileBytes);
    noc_async_read(v_noc,  scratch_V, Nk * kTileBytes);
    noc_async_read_barrier();

    // V faces (row determines top vs bottom pair).
    const bool     v_bot_half  = (slot1_r >= 16);
    const uint32_t v_face_left  = v_bot_half ? 2 : 0;  // cols 0-15
    const uint32_t v_face_right = v_bot_half ? 3 : 1;  // cols 16-31
    const uint32_t v_row_in_face = slot1_r & 15;

    // ---------------------------- V append -----------------------------
    // Slot1 is mt=1 in T_V_cache. Tile (1, c) at offset (1*Nk + c)*2048.
    // Write face-row at v_row_in_face (32 bytes per face) for both face A
    // and face B per NK-block.
    for (uint32_t c = 0; c < Nk; ++c) {
        const uint32_t src_f0 = scratch_V + c * kTileBytes;                  // face 0 row 0
        const uint32_t src_f1 = scratch_V + c * kTileBytes + kFaceBytes;     // face 1 row 0
        const uint64_t tile_base = v_base + (uint64_t)(1 * Nk + c) * kTileBytes;
        const uint64_t dst_left  = tile_base + (uint64_t)v_face_left  * kFaceBytes
                                              + (uint64_t)v_row_in_face * kFaceRowBytes;
        const uint64_t dst_right = tile_base + (uint64_t)v_face_right * kFaceBytes
                                              + (uint64_t)v_row_in_face * kFaceRowBytes;
        noc_async_write(src_f0, dst_left,  kFaceRowBytes);
        noc_async_write(src_f1, dst_right, kFaceRowBytes);
    }
    noc_async_write_barrier();

    // ---------------------------- K append -----------------------------
    // K^T cache tile (c, 1) at offset (c*StKv + 1)*2048. We write col
    // slot1_r of this tile, which sits in face (top, bot) determined by
    // slot1_r/16 along the column axis. col_right means cols 16-31 of
    // the tile → faces (1, 3); else cols 0-15 → faces (0, 2).
    const bool     k_col_right = (slot1_r >= 16);
    const uint32_t k_face_top = k_col_right ? 1 : 0;
    const uint32_t k_face_bot = k_col_right ? 3 : 2;
    const uint32_t k_col_in_face = slot1_r & 15;

    volatile uint16_t* face_words = reinterpret_cast<volatile uint16_t*>(face_buf);

    for (uint32_t c = 0; c < Nk; ++c) {
        // K[pos][c*32..c*32+15] in T_Kr tile c face 0 row 0 (bytes 0..31)
        // K[pos][c*32+16..c*32+31] in T_Kr tile c face 1 row 0 (bytes 512..543)
        volatile uint16_t* src_top = reinterpret_cast<volatile uint16_t*>(
            scratch_K + c * kTileBytes);
        volatile uint16_t* src_bot = reinterpret_cast<volatile uint16_t*>(
            scratch_K + c * kTileBytes + kFaceBytes);

        // K^T cache is slot-major (see ops/gqa_decode/reader.cpp): slot 1
        // tile c lives at offset (1 * Nk + c) * 2048, same shape as V cache.
        const uint64_t tile_base = kt_base + (uint64_t)(1 * Nk + c) * kTileBytes;

        // ---- top face (NK rows c*32..c*32+15) ----
        const uint64_t face_top_addr = tile_base + (uint64_t)k_face_top * kFaceBytes;
        noc_async_read(face_top_addr, face_buf, kFaceBytes);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 16; ++r) {
            face_words[r * 16 + k_col_in_face] = src_top[r];
        }
        noc_async_write(face_buf, face_top_addr, kFaceBytes);
        noc_async_write_barrier();

        // ---- bot face (NK rows c*32+16..c*32+31) ----
        const uint64_t face_bot_addr = tile_base + (uint64_t)k_face_bot * kFaceBytes;
        noc_async_read(face_bot_addr, face_buf, kFaceBytes);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 16; ++r) {
            face_words[r * 16 + k_col_in_face] = src_bot[r];
        }
        noc_async_write(face_buf, face_bot_addr, kFaceBytes);
        noc_async_write_barrier();
    }

    cb_push_back(cb_io, 1);
}
