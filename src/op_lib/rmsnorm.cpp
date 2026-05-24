// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: RmsNormOp implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include <bit>

#include "cb_config.hpp"
#include "op_cache.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::kTileH;
using detail::kTileW;
using detail::resolve_kernel_dir;
using detail::make_const_tile;

static RmsNormOp make_rmsnorm_impl(tt::foil::Device& dev,
                       const TensorDesc& x, const TensorDesc& gamma,
                       TensorDesc& out,
                       uint32_t NCHt, uint32_t Wt,
                       float eps,
                       tt::foil::CoreCoord core,
                       const std::string& kernel_dir) {
    if (x.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_rmsnorm: x.num_tiles != NCHt*Wt");
    if (gamma.num_tiles != Wt)
        throw std::runtime_error("op_lib::make_rmsnorm: gamma.num_tiles != Wt");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, NCHt * Wt);
    else if (out.num_tiles != x.num_tiles)
        throw std::runtime_error("op_lib::make_rmsnorm: out.num_tiles != x.num_tiles");

    const std::string dir = resolve_kernel_dir(kernel_dir, "rmsnorm");

    RmsNormOp op;

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
    op.l1_inp        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_reduce     = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_gamma      = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_eps        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x2         = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_var        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_recip_sqrt = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x_normed   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_out        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/rmsnorm.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/rmsnorm.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/rmsnorm.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 9> cbs = {{
        {0,  op.l1_inp->device_addr,        l1_wt,      Wt, kTileBytes},
        {1,  op.l1_reduce->device_addr,     kTileBytes, 1,  kTileBytes},
        {2,  op.l1_gamma->device_addr,      l1_wt,      Wt, kTileBytes},
        {3,  op.l1_eps->device_addr,        kTileBytes, 1,  kTileBytes},
        {4,  op.l1_x2->device_addr,         l1_wt,      Wt, kTileBytes},
        {5,  op.l1_var->device_addr,        kTileBytes, 1,  kTileBytes},
        {6,  op.l1_recip_sqrt->device_addr, kTileBytes, 1,  kTileBytes},
        {7,  op.l1_x_normed->device_addr,   l1_wt,      Wt, kTileBytes},
        {16, op.l1_out->device_addr,        kTileBytes, 1,  kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_rmsnorm_args(dev, op, x, gamma, out, NCHt, Wt);
    return op;
}

RmsNormOp make_rmsnorm(tt::foil::Device& dev,
                       const TensorDesc& x, const TensorDesc& gamma,
                       TensorDesc& out,
                       uint32_t NCHt, uint32_t Wt,
                       float eps,
                       tt::foil::CoreCoord core,
                       const std::string& kernel_dir) {
    // Allocate the output here if the caller didn't, so subsequent
    // cache-hit refreshes don't run with num_tiles==0 expecting the
    // factory to allocate. The first-time factory call below will see
    // num_tiles!=0 and just validate.
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, NCHt * Wt);

    static thread_local OpCache<RmsNormOp> g_cache;
    ShapeKey key{
        .op_name  = "rmsnorm",
        .params   = {NCHt, Wt, std::bit_cast<uint32_t>(eps), 0, 0, 0, 0, 0},
        .core_key = core_key_of(core),
    };
    RmsNormOp& op = g_cache.get_or_create(dev, core, key, [&] {
        return make_rmsnorm_impl(dev, x, gamma, out, NCHt, Wt, eps, core, kernel_dir);
    });
    // Cache hit path: refresh RTAs for this call's tensors. (On the
    // first-create path, make_rmsnorm_impl already called this — the
    // extra call here is idempotent and cheap.)
    set_rmsnorm_args(dev, op, x, gamma, out, NCHt, Wt);
    return op;
}

void set_rmsnorm_args(tt::foil::Device& dev, RmsNormOp& op,
                      const TensorDesc& x, const TensorDesc& gamma,
                      const TensorDesc& out,
                      uint32_t NCHt, uint32_t Wt) {
    using R = tt::foil::RiscBinary;
    const uint64_t x_noc     = tt::foil::make_noc_dram_addr(dev, x.buf->device_addr);
    const uint64_t g_noc     = tt::foil::make_noc_dram_addr(dev, gamma.buf->device_addr);
    const uint64_t sc_noc    = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t eps_noc   = tt::foil::make_noc_dram_addr(dev, op.dram_eps->device_addr);
    const uint64_t dst_noc   = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 10> ra_brisc = {
        (uint32_t)x_noc,   (uint32_t)(x_noc >> 32),
        (uint32_t)g_noc,   (uint32_t)(g_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)eps_noc, (uint32_t)(eps_noc >> 32),
        NCHt, Wt,
    };
    std::array<uint32_t, 2> ra_trisc = {NCHt, Wt};
    std::array<uint32_t, 4> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
        NCHt, Wt,
    };

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
}

void execute(tt::foil::Device& dev, RmsNormOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
