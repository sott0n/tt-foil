// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: AddRmsNormOp — fused S = A+B; Y = RMSNorm(S, gamma).
// Writes both S and Y to separate DRAM destinations.

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
using detail::kTileW;
using detail::resolve_kernel_dir;
using detail::make_const_tile;

static AddRmsNormOp make_add_rmsnorm_impl(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              const TensorDesc& gamma,
                              TensorDesc& sum_out, TensorDesc& normed_out,
                              uint32_t NCHt, uint32_t Wt,
                              float eps,
                              tt::foil::CoreCoord core,
                              const std::string& kernel_dir) {
    if (a.num_tiles != NCHt * Wt || b.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_add_rmsnorm: a/b num_tiles != NCHt*Wt");
    if (gamma.num_tiles != Wt)
        throw std::runtime_error("op_lib::make_add_rmsnorm: gamma.num_tiles != Wt");
    if (sum_out.num_tiles == 0)
        sum_out = allocate_tensor_dram(dev, NCHt * Wt);
    else if (sum_out.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_add_rmsnorm: sum_out tile count mismatch");
    if (normed_out.num_tiles == 0)
        normed_out = allocate_tensor_dram(dev, NCHt * Wt);
    else if (normed_out.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_add_rmsnorm: normed_out tile count mismatch");

    const std::string dir = resolve_kernel_dir(kernel_dir, "add_rmsnorm");

    AddRmsNormOp op;
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
    // depth-1 cb_a / cb_b (stream tile-by-tile from reader)
    op.l1_a          = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_b          = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_reduce     = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_gamma      = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_eps        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x2         = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_var        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_recip_sqrt = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_x_normed   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_sum        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_s_out      = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_out        = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/add_rmsnorm.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/add_rmsnorm.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/add_rmsnorm.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    // CB layout (matches add_rmsnorm/compute.cpp + reader.cpp + writer.cpp)
    std::array<tt::foil::CbConfig, 12> cbs = {{
        {0,  op.l1_a->device_addr,          kTileBytes, 1,  kTileBytes},
        {1,  op.l1_reduce->device_addr,     kTileBytes, 1,  kTileBytes},
        {2,  op.l1_gamma->device_addr,      l1_wt,      Wt, kTileBytes},
        {3,  op.l1_eps->device_addr,        kTileBytes, 1,  kTileBytes},
        {4,  op.l1_x2->device_addr,         l1_wt,      Wt, kTileBytes},
        {5,  op.l1_var->device_addr,        kTileBytes, 1,  kTileBytes},
        {6,  op.l1_recip_sqrt->device_addr, kTileBytes, 1,  kTileBytes},
        {7,  op.l1_x_normed->device_addr,   l1_wt,      Wt, kTileBytes},
        {8,  op.l1_b->device_addr,          kTileBytes, 1,  kTileBytes},
        {9,  op.l1_sum->device_addr,        l1_wt,      Wt, kTileBytes},
        {10, op.l1_s_out->device_addr,      l1_wt,      Wt, kTileBytes},
        {16, op.l1_out->device_addr,        kTileBytes, 1,  kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_add_rmsnorm_args(dev, op, a, b, gamma, sum_out, normed_out, NCHt, Wt);
    return op;
}

AddRmsNormOp make_add_rmsnorm(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              const TensorDesc& gamma,
                              TensorDesc& sum_out, TensorDesc& normed_out,
                              uint32_t NCHt, uint32_t Wt,
                              float eps,
                              tt::foil::CoreCoord core,
                              const std::string& kernel_dir) {
    if (sum_out.num_tiles == 0)
        sum_out = allocate_tensor_dram(dev, NCHt * Wt);
    if (normed_out.num_tiles == 0)
        normed_out = allocate_tensor_dram(dev, NCHt * Wt);

    static thread_local OpCache<AddRmsNormOp> g_cache;
    ShapeKey key{
        .op_name  = "add_rmsnorm",
        .params   = {NCHt, Wt, std::bit_cast<uint32_t>(eps), 0, 0, 0, 0, 0},
        .core_key = core_key_of(core),
    };
    AddRmsNormOp& op = g_cache.get_or_create(dev, core, key, [&] {
        return make_add_rmsnorm_impl(dev, a, b, gamma, sum_out, normed_out, NCHt, Wt, eps, core, kernel_dir);
    });
    set_add_rmsnorm_args(dev, op, a, b, gamma, sum_out, normed_out, NCHt, Wt);
    return op;
}

void set_add_rmsnorm_args(tt::foil::Device& dev, AddRmsNormOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& gamma,
                          const TensorDesc& sum_out, const TensorDesc& normed_out,
                          uint32_t NCHt, uint32_t Wt) {
    using R = tt::foil::RiscBinary;
    const uint64_t a_noc   = tt::foil::make_noc_dram_addr(dev, a.buf->device_addr);
    const uint64_t b_noc   = tt::foil::make_noc_dram_addr(dev, b.buf->device_addr);
    const uint64_t g_noc   = tt::foil::make_noc_dram_addr(dev, gamma.buf->device_addr);
    const uint64_t sc_noc  = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t eps_noc = tt::foil::make_noc_dram_addr(dev, op.dram_eps->device_addr);
    const uint64_t s_noc   = tt::foil::make_noc_dram_addr(dev, sum_out.buf->device_addr);
    const uint64_t y_noc   = tt::foil::make_noc_dram_addr(dev, normed_out.buf->device_addr);

    std::array<uint32_t, 12> ra_brisc = {
        (uint32_t)a_noc,   (uint32_t)(a_noc >> 32),
        (uint32_t)b_noc,   (uint32_t)(b_noc >> 32),
        (uint32_t)g_noc,   (uint32_t)(g_noc >> 32),
        (uint32_t)sc_noc,  (uint32_t)(sc_noc >> 32),
        (uint32_t)eps_noc, (uint32_t)(eps_noc >> 32),
        NCHt, Wt,
    };
    std::array<uint32_t, 2> ra_trisc = {NCHt, Wt};
    std::array<uint32_t, 6> ra_ncrisc = {
        (uint32_t)s_noc, (uint32_t)(s_noc >> 32),
        (uint32_t)y_noc, (uint32_t)(y_noc >> 32),
        NCHt, Wt,
    };

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
}

void execute(tt::foil::Device& dev, AddRmsNormOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
