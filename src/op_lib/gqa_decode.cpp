// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: GqaDecodeOp — multi-head GQA decode attention.
//
// Identical math to make_gqa_fused, but St_q (Q tile-rows) and St_kv
// (K / V cache tile-rows) are decoupled — typical decode usage has
// St_q = 1 (one query token padded to a 32-row tile) and St_kv = the
// current cache depth in tile-rows.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;
using detail::make_const_tile;

GqaDecodeOp make_gqa_decode(tt::foil::Device& dev,
                            const TensorDesc& q, const TensorDesc& kt,
                            const TensorDesc& v, const TensorDesc& mask,
                            TensorDesc& out,
                            uint32_t St_q, uint32_t St_kv, uint32_t Dt,
                            uint32_t num_q, uint32_t num_kv,
                            tt::foil::CoreCoord core,
                            const std::string& kernel_dir) {
    if (num_kv == 0 || num_q % num_kv != 0)
        throw std::runtime_error("op_lib::make_gqa_decode: num_q must be a multiple of num_kv");

    const uint32_t q_tiles    = St_q  * Dt;
    const uint32_t kt_tiles   = Dt    * St_kv;
    const uint32_t v_tiles    = St_kv * Dt;
    const uint32_t mask_tiles = St_q  * St_kv;
    const uint32_t total_Nq   = num_q  * Dt;
    const uint32_t total_Nk   = num_kv * Dt;

    if (q.num_tiles != St_q * total_Nq)
        throw std::runtime_error("op_lib::make_gqa_decode: q.num_tiles != St_q * num_q * Dt");
    if (kt.num_tiles != total_Nk * St_kv)
        throw std::runtime_error("op_lib::make_gqa_decode: kt.num_tiles != num_kv*Dt * St_kv");
    if (v.num_tiles != St_kv * total_Nk)
        throw std::runtime_error("op_lib::make_gqa_decode: v.num_tiles != St_kv * num_kv * Dt");
    if (mask.num_tiles != mask_tiles)
        throw std::runtime_error("op_lib::make_gqa_decode: mask.num_tiles != St_q * St_kv");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, St_q * total_Nq);
    else if (out.num_tiles != St_q * total_Nq)
        throw std::runtime_error("op_lib::make_gqa_decode: out.num_tiles != St_q * num_q * Dt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "gqa_decode");

    GqaDecodeOp op;
    op.dram_scaler = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    {
        auto s = make_const_tile(1.0f);
        tt::foil::write_buffer(dev, *op.dram_scaler, s.data(), kTileBytes);
    }

    const uint32_t q_bytes    = q_tiles    * kTileBytes;
    const uint32_t kt_bytes   = kt_tiles   * kTileBytes;
    const uint32_t v_bytes    = v_tiles    * kTileBytes;
    const uint32_t mask_bytes = mask_tiles * kTileBytes;
    const uint32_t row_bytes  = St_kv      * kTileBytes;   // scores / exp / exp_m / softmaxed

    auto alloc_l1 = [&](uint32_t bytes) {
        return tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, bytes, core);
    };
    auto l1_q         = alloc_l1(q_bytes);
    auto l1_kt        = alloc_l1(kt_bytes);
    auto l1_v         = alloc_l1(v_bytes);
    auto l1_reduce    = alloc_l1(kTileBytes);
    auto l1_scores    = alloc_l1(row_bytes);
    auto l1_exp       = alloc_l1(row_bytes);
    auto l1_sum       = alloc_l1(kTileBytes);
    auto l1_recip     = alloc_l1(kTileBytes);
    auto l1_softmaxed = alloc_l1(row_bytes);
    auto l1_exp_m     = alloc_l1(row_bytes);
    auto l1_mask      = alloc_l1(mask_bytes);
    auto l1_out       = alloc_l1(kTileBytes);
    op.l1_cbs = {l1_q, l1_kt, l1_v, l1_reduce, l1_scores, l1_exp,
                 l1_sum, l1_recip, l1_softmaxed, l1_exp_m, l1_mask, l1_out};

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/gqa_decode.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/gqa_decode.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/gqa_decode.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 12> cbs = {{
        {0,  l1_q->device_addr,         q_bytes,    q_tiles,    kTileBytes},
        {1,  l1_kt->device_addr,        kt_bytes,   kt_tiles,   kTileBytes},
        {2,  l1_v->device_addr,         v_bytes,    v_tiles,    kTileBytes},
        {3,  l1_reduce->device_addr,    kTileBytes, 1,          kTileBytes},
        {4,  l1_scores->device_addr,    row_bytes,  St_kv,      kTileBytes},
        {5,  l1_exp->device_addr,       row_bytes,  St_kv,      kTileBytes},
        {6,  l1_sum->device_addr,       kTileBytes, 1,          kTileBytes},
        {7,  l1_recip->device_addr,     kTileBytes, 1,          kTileBytes},
        {8,  l1_softmaxed->device_addr, row_bytes,  St_kv,      kTileBytes},
        {9,  l1_exp_m->device_addr,     row_bytes,  St_kv,      kTileBytes},
        {10, l1_mask->device_addr,      mask_bytes, mask_tiles, kTileBytes},
        {16, l1_out->device_addr,       kTileBytes, 1,          kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    const uint64_t q_noc   = tt::foil::make_noc_dram_addr(dev, q.buf->device_addr);
    const uint64_t kt_noc  = tt::foil::make_noc_dram_addr(dev, kt.buf->device_addr);
    const uint64_t v_noc   = tt::foil::make_noc_dram_addr(dev, v.buf->device_addr);
    const uint64_t sc_noc  = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t m_noc   = tt::foil::make_noc_dram_addr(dev, mask.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 15> ra_brisc = {
        (uint32_t)q_noc,   (uint32_t)(q_noc >> 32),
        (uint32_t)kt_noc,  (uint32_t)(kt_noc >> 32),
        (uint32_t)v_noc,   (uint32_t)(v_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)m_noc,   (uint32_t)(m_noc >> 32),
        St_q, St_kv, Dt, num_q, num_kv,
    };
    std::array<uint32_t, 5> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
        St_q, Dt, num_q,
    };
    std::array<uint32_t, 4> ra_trisc = {St_q, St_kv, Dt, num_q};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);

    return op;
}

void execute(tt::foil::Device& dev, GqaDecodeOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
