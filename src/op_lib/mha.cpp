// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: MhaOp implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;
using detail::make_const_tile;

MhaOp make_mha(tt::foil::Device& dev,
               const TensorDesc& q, const TensorDesc& kt,
               const TensorDesc& v, const TensorDesc& mask,
               TensorDesc& out,
               uint32_t St, uint32_t Dt,
               tt::foil::CoreCoord core,
               const std::string& kernel_dir) {
    const uint32_t qkv_tiles  = St * Dt;
    const uint32_t kt_tiles   = Dt * St;
    const uint32_t mask_tiles = St * St;
    if (q.num_tiles != qkv_tiles || v.num_tiles != qkv_tiles)
        throw std::runtime_error("op_lib::make_mha: q/v num_tiles must be St*Dt");
    if (kt.num_tiles != kt_tiles)
        throw std::runtime_error("op_lib::make_mha: kt.num_tiles must be Dt*St");
    if (mask.num_tiles != mask_tiles)
        throw std::runtime_error("op_lib::make_mha: mask.num_tiles must be St*St");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, qkv_tiles);
    else if (out.num_tiles != qkv_tiles)
        throw std::runtime_error("op_lib::make_mha: out.num_tiles != St*Dt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "mha");

    MhaOp op;
    op.dram_scaler = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    {
        auto s = make_const_tile(1.0f);
        tt::foil::write_buffer(dev, *op.dram_scaler, s.data(), kTileBytes);
    }

    const uint32_t qkv_bytes  = qkv_tiles  * kTileBytes;
    const uint32_t kt_bytes   = kt_tiles   * kTileBytes;
    const uint32_t mask_bytes = mask_tiles * kTileBytes;
    const uint32_t row_bytes  = St         * kTileBytes;

    auto alloc_l1 = [&](uint32_t bytes) {
        return tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, bytes, core);
    };
    auto l1_q         = alloc_l1(qkv_bytes);
    auto l1_kt        = alloc_l1(kt_bytes);
    auto l1_v         = alloc_l1(qkv_bytes);
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
        {R::RiscId::TRISC0, dir + "/mha.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/mha.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/mha.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 12> cbs = {{
        {0,  l1_q->device_addr,         qkv_bytes,  qkv_tiles,  kTileBytes},
        {1,  l1_kt->device_addr,        kt_bytes,   kt_tiles,   kTileBytes},
        {2,  l1_v->device_addr,         qkv_bytes,  qkv_tiles,  kTileBytes},
        {3,  l1_reduce->device_addr,    kTileBytes, 1,          kTileBytes},
        {4,  l1_scores->device_addr,    row_bytes,  St,         kTileBytes},
        {5,  l1_exp->device_addr,       row_bytes,  St,         kTileBytes},
        {6,  l1_sum->device_addr,       kTileBytes, 1,          kTileBytes},
        {7,  l1_recip->device_addr,     kTileBytes, 1,          kTileBytes},
        {8,  l1_softmaxed->device_addr, row_bytes,  St,         kTileBytes},
        {9,  l1_exp_m->device_addr,     row_bytes,  St,         kTileBytes},
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

    std::array<uint32_t, 12> ra_brisc = {
        (uint32_t)q_noc,   (uint32_t)(q_noc >> 32),
        (uint32_t)kt_noc,  (uint32_t)(kt_noc >> 32),
        (uint32_t)v_noc,   (uint32_t)(v_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)m_noc,   (uint32_t)(m_noc >> 32),
        St, Dt,
    };
    std::array<uint32_t, 2> ra_trisc  = {St, Dt};
    std::array<uint32_t, 4> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
        St, Dt,
    };

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);

    return op;
}

void execute(tt::foil::Device& dev, MhaOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
