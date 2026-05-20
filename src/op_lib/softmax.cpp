// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: SoftmaxOp implementation.

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

SoftmaxOp make_softmax(tt::foil::Device& dev,
                       const TensorDesc& x, TensorDesc& out,
                       uint32_t NCHt, uint32_t Wt,
                       tt::foil::CoreCoord core,
                       const std::string& kernel_dir) {
    if (x.num_tiles != NCHt * Wt)
        throw std::runtime_error("op_lib::make_softmax: x.num_tiles != NCHt*Wt");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, NCHt * Wt);
    else if (out.num_tiles != x.num_tiles)
        throw std::runtime_error("op_lib::make_softmax: out.num_tiles != x.num_tiles");

    const std::string dir = resolve_kernel_dir(kernel_dir, "softmax");

    SoftmaxOp op;

    // Scaler tile (=1.0) lives in DRAM and is read by the kernel reader.
    op.dram_scaler = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, kTileBytes);
    {
        auto t = make_const_tile(1.0f);
        tt::foil::write_buffer(dev, *op.dram_scaler, t.data(), kTileBytes);
    }

    const uint32_t l1_wt = Wt * kTileBytes;
    op.l1_inp    = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_reduce = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_exp    = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, l1_wt,      core);
    op.l1_sum    = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_recip  = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out    = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/softmax.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/softmax.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/softmax.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 6> cbs = {{
        {0,  op.l1_inp->device_addr,    l1_wt,      Wt, kTileBytes},
        {1,  op.l1_reduce->device_addr, kTileBytes, 1,  kTileBytes},
        {2,  op.l1_exp->device_addr,    l1_wt,      Wt, kTileBytes},
        {3,  op.l1_sum->device_addr,    kTileBytes, 1,  kTileBytes},
        {4,  op.l1_recip->device_addr,  kTileBytes, 1,  kTileBytes},
        {16, op.l1_out->device_addr,    kTileBytes, 1,  kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    const uint64_t x_noc   = tt::foil::make_noc_dram_addr(dev, x.buf->device_addr);
    const uint64_t sc_noc  = tt::foil::make_noc_dram_addr(dev, op.dram_scaler->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 6> ra_brisc = {
        (uint32_t)x_noc,  (uint32_t)(x_noc >> 32),
        (uint32_t)sc_noc, (uint32_t)(sc_noc >> 32),
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

    return op;
}

void execute(tt::foil::Device& dev, SoftmaxOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
