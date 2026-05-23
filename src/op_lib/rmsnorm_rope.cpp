// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: RmsNormRopeOp — fused RMSNorm(x, gamma) followed by per-head
// RoPE rotation using cos/sin tables. Replaces the rmsnorm_qk → rope
// dispatch pair (2 → 1 dispatch per Q/K projection).

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::kTileW;
using detail::resolve_kernel_dir;
using detail::make_const_tile;

RmsNormRopeOp make_rmsnorm_rope(tt::foil::Device& dev,
                                const TensorDesc& x, const TensorDesc& gamma,
                                const TensorDesc& cos, const TensorDesc& sin,
                                TensorDesc& out,
                                uint32_t St, uint32_t num_heads, uint32_t Dt_half,
                                float eps,
                                tt::foil::CoreCoord core,
                                const std::string& kernel_dir) {
    const uint32_t Wt   = 2 * Dt_half;
    const uint32_t NCHt = St * num_heads;
    if (x.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_rmsnorm_rope: x.num_tiles != NCHt*Wt");
    if (gamma.num_tiles != Wt)
        throw std::runtime_error("op_lib::make_rmsnorm_rope: gamma.num_tiles != Wt");
    if (cos.num_tiles != St * Dt_half || sin.num_tiles != St * Dt_half)
        throw std::runtime_error("op_lib::make_rmsnorm_rope: cos/sin tile count mismatch");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, NCHt * Wt);
    else if (out.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_rmsnorm_rope: out tile count mismatch");

    const std::string dir = resolve_kernel_dir(kernel_dir, "rmsnorm_rope");

    RmsNormRopeOp op;
    const uint32_t H = Wt * kTileW;
    op.dram_scaler = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    op.dram_eps    = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    {
        auto s = make_const_tile(1.0f / static_cast<float>(H));
        tt::foil::write_buffer(dev, *op.dram_scaler, s.data(), kTileBytes);
        auto e = make_const_tile(eps);
        tt::foil::write_buffer(dev, *op.dram_eps, e.data(), kTileBytes);
    }

    const uint32_t l1_wt = Wt * kTileBytes;
    const uint32_t l1_cos = St * Dt_half * kTileBytes;
    op.l1_x          = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_reduce     = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_gamma      = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_eps        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x2         = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_var        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_recip_sqrt = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x_normed   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_cos        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_cos,     core);
    op.l1_sin        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_cos,     core);
    op.l1_normed     = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_tmp0       = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_tmp1       = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/rmsnorm_rope.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/rmsnorm_rope.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/rmsnorm_rope.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    const uint32_t cos_tiles = St * Dt_half;
    std::array<tt::foil::CbConfig, 14> cbs = {{
        {0,  op.l1_x->device_addr,          l1_wt,      Wt,        kTileBytes},
        {1,  op.l1_reduce->device_addr,     kTileBytes, 1,         kTileBytes},
        {2,  op.l1_gamma->device_addr,      l1_wt,      Wt,        kTileBytes},
        {3,  op.l1_eps->device_addr,        kTileBytes, 1,         kTileBytes},
        {4,  op.l1_x2->device_addr,         l1_wt,      Wt,        kTileBytes},
        {5,  op.l1_var->device_addr,        kTileBytes, 1,         kTileBytes},
        {6,  op.l1_recip_sqrt->device_addr, kTileBytes, 1,         kTileBytes},
        {7,  op.l1_x_normed->device_addr,   l1_wt,      Wt,        kTileBytes},
        {8,  op.l1_cos->device_addr,        l1_cos,     cos_tiles, kTileBytes},
        {9,  op.l1_sin->device_addr,        l1_cos,     cos_tiles, kTileBytes},
        {10, op.l1_normed->device_addr,     l1_wt,      Wt,        kTileBytes},
        {11, op.l1_tmp0->device_addr,       kTileBytes, 1,         kTileBytes},
        {12, op.l1_tmp1->device_addr,       kTileBytes, 1,         kTileBytes},
        {16, op.l1_out->device_addr,        kTileBytes, 1,         kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_rmsnorm_rope_args(dev, op, x, gamma, cos, sin, out, St, num_heads, Dt_half);
    return op;
}

void set_rmsnorm_rope_args(tt::foil::Device& dev, RmsNormRopeOp& op,
                           const TensorDesc& x, const TensorDesc& gamma,
                           const TensorDesc& cos, const TensorDesc& sin,
                           const TensorDesc& out,
                           uint32_t St, uint32_t num_heads, uint32_t Dt_half) {
    using R = tt::foil::RiscBinary;
    const uint32_t Wt   = 2 * Dt_half;
    const uint32_t NCHt = St * num_heads;
    const uint64_t x_noc      = tt::foil::make_noc_dram_addr(dev, x.buf->device_addr);
    const uint64_t g_noc      = tt::foil::make_noc_dram_addr(dev, gamma.buf->device_addr);
    const uint64_t sc_noc     = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t eps_noc    = tt::foil::make_noc_dram_addr(dev, op.dram_eps->device_addr);
    const uint64_t cos_noc    = tt::foil::make_noc_dram_addr(dev, cos.buf->device_addr);
    const uint64_t sin_noc    = tt::foil::make_noc_dram_addr(dev, sin.buf->device_addr);
    const uint64_t out_noc    = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 16> ra_brisc = {
        (uint32_t)x_noc,   (uint32_t)(x_noc >> 32),
        (uint32_t)g_noc,   (uint32_t)(g_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)eps_noc, (uint32_t)(eps_noc >> 32),
        (uint32_t)cos_noc, (uint32_t)(cos_noc >> 32),
        (uint32_t)sin_noc, (uint32_t)(sin_noc >> 32),
        NCHt, Wt, St, Dt_half,
    };
    std::array<uint32_t, 4> ra_trisc  = {NCHt, Wt, num_heads, Dt_half};
    std::array<uint32_t, 5> ra_ncrisc = {
        (uint32_t)out_noc, (uint32_t)(out_noc >> 32),
        St, num_heads, Dt_half,
    };

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
}

void execute(tt::foil::Device& dev, RmsNormRopeOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
