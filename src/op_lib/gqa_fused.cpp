// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: GqaFusedOp — single-kernel multi-head GQA causal attention.

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

GqaFusedOp make_gqa_fused(tt::foil::Device& dev,
                          const TensorDesc& q, const TensorDesc& kt,
                          const TensorDesc& v, const TensorDesc& mask,
                          TensorDesc& out,
                          uint32_t St, uint32_t Dt,
                          uint32_t num_q, uint32_t num_kv,
                          tt::foil::CoreCoord core,
                          const std::string& kernel_dir) {
    if (num_kv == 0 || num_q % num_kv != 0)
        throw std::runtime_error("op_lib::make_gqa_fused: num_q must be a multiple of num_kv");

    const uint32_t total_Nq   = num_q  * Dt;
    const uint32_t total_Nk   = num_kv * Dt;

    if (q.num_tiles != St * total_Nq)
        throw std::runtime_error("op_lib::make_gqa_fused: q.num_tiles != St * num_q * Dt");
    if (kt.num_tiles != total_Nk * St)
        throw std::runtime_error("op_lib::make_gqa_fused: kt.num_tiles != num_kv*Dt * St");
    if (v.num_tiles != St * total_Nk)
        throw std::runtime_error("op_lib::make_gqa_fused: v.num_tiles != St * num_kv * Dt");
    // Flash form: the causal mask is a single 32x32 lower-triangular tile.
    if (mask.num_tiles != 1)
        throw std::runtime_error("op_lib::make_gqa_fused: mask.num_tiles != 1 (flash tri tile)");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, St * total_Nq);
    else if (out.num_tiles != St * total_Nq)
        throw std::runtime_error("op_lib::make_gqa_fused: out.num_tiles != St * num_q * Dt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "gqa_fused");

    GqaFusedOp op;
    op.dram_scaler = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    {
        auto s = make_const_tile(1.0f);
        tt::foil::write_buffer(dev, *op.dram_scaler, s.data(), kTileBytes);
    }

    // Flash CB sizing — all O(Dt), independent of St. Each CB's fifo_size is
    // exactly depth*page_size (the num_pages*page_size invariant). cb_q/kt/v are
    // double-buffered (2*Dt) for reader/compute overlap; O and l use TWO
    // ping/pong CBs (cb_o_a/b, cb_l_a/b) since a compute kernel must not
    // wait_front + reserve_back the same CB in one iteration.
    auto alloc_l1 = [&](uint32_t tiles) {
        return tt::foil::allocate_buffer(
            dev, tt::foil::BufferLocation::L1, tiles * kTileBytes, core);
    };
    const uint32_t dq = 2 * Dt;   // double-buffered stream depth
    auto l1_q      = alloc_l1(dq);
    auto l1_kt     = alloc_l1(dq);
    auto l1_v      = alloc_l1(dq);
    auto l1_reduce = alloc_l1(1);
    auto l1_s      = alloc_l1(2);
    auto l1_p      = alloc_l1(2);
    auto l1_pm     = alloc_l1(2);
    auto l1_rs     = alloc_l1(2);
    auto l1_pv     = alloc_l1(Dt);
    auto l1_o_a    = alloc_l1(Dt);
    auto l1_o_b    = alloc_l1(Dt);
    auto l1_l_a    = alloc_l1(1);
    auto l1_l_b    = alloc_l1(1);
    auto l1_recip  = alloc_l1(1);
    auto l1_tri    = alloc_l1(1);
    auto l1_out    = alloc_l1(2);
    op.l1_cbs = {l1_q, l1_kt, l1_v, l1_reduce, l1_s, l1_p, l1_pm, l1_rs,
                 l1_pv, l1_o_a, l1_o_b, l1_l_a, l1_l_b, l1_recip, l1_tri, l1_out};

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/gqa_fused.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/gqa_fused.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/gqa_fused.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    const uint32_t tb = kTileBytes;
    std::array<tt::foil::CbConfig, 16> cbs = {{
        {0,  l1_q->device_addr,      dq * tb, dq, tb},
        {1,  l1_kt->device_addr,     dq * tb, dq, tb},
        {2,  l1_v->device_addr,      dq * tb, dq, tb},
        {3,  l1_reduce->device_addr, tb,      1,  tb},
        {4,  l1_s->device_addr,      2 * tb,  2,  tb},
        {5,  l1_p->device_addr,      2 * tb,  2,  tb},
        {6,  l1_pm->device_addr,     2 * tb,  2,  tb},
        {7,  l1_rs->device_addr,     2 * tb,  2,  tb},
        {8,  l1_pv->device_addr,     Dt * tb, Dt, tb},
        {9,  l1_o_a->device_addr,    Dt * tb, Dt, tb},
        {10, l1_o_b->device_addr,    Dt * tb, Dt, tb},
        {11, l1_l_a->device_addr,    tb,      1,  tb},
        {12, l1_l_b->device_addr,    tb,      1,  tb},
        {13, l1_recip->device_addr,  tb,      1,  tb},
        {14, l1_tri->device_addr,    tb,      1,  tb},
        {16, l1_out->device_addr,    2 * tb,  2,  tb},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_gqa_fused_args(dev, op, q, kt, v, mask, out, St, Dt, num_q, num_kv);
    return op;
}

void set_gqa_fused_args(tt::foil::Device& dev, GqaFusedOp& op,
                        const TensorDesc& q, const TensorDesc& kt,
                        const TensorDesc& v, const TensorDesc& mask,
                        const TensorDesc& out,
                        uint32_t St, uint32_t Dt,
                        uint32_t num_q, uint32_t num_kv) {
    using R = tt::foil::RiscBinary;
    const uint64_t q_noc   = tt::foil::make_noc_dram_addr(dev, q.buf->device_addr);
    const uint64_t kt_noc  = tt::foil::make_noc_dram_addr(dev, kt.buf->device_addr);
    const uint64_t v_noc   = tt::foil::make_noc_dram_addr(dev, v.buf->device_addr);
    const uint64_t sc_noc  = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t m_noc   = tt::foil::make_noc_dram_addr(dev, mask.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 14> ra_brisc = {
        (uint32_t)q_noc,   (uint32_t)(q_noc >> 32),
        (uint32_t)kt_noc,  (uint32_t)(kt_noc >> 32),
        (uint32_t)v_noc,   (uint32_t)(v_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)m_noc,   (uint32_t)(m_noc >> 32),
        St, Dt, num_q, num_kv,
    };
    std::array<uint32_t, 5> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
        St, Dt, num_q,
    };
    std::array<uint32_t, 3> ra_trisc = {St, Dt, num_q};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
}

void execute(tt::foil::Device& dev, GqaFusedOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
