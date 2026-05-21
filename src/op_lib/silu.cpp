// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: SiluOp + allocate_tensor_dram implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

TensorDesc allocate_tensor_dram(tt::foil::Device& dev, uint32_t num_tiles) {
    TensorDesc t;
    t.num_tiles = num_tiles;
    t.buf = tt::foil::allocate_buffer(
        dev, tt::foil::BufferLocation::DRAM,
        static_cast<std::size_t>(num_tiles) * kTileBytes);
    return t;
}

SiluOp make_silu(tt::foil::Device& dev,
                 const TensorDesc& x, TensorDesc& out,
                 tt::foil::CoreCoord core,
                 const std::string& kernel_dir) {
    if (x.num_tiles == 0)
        throw std::runtime_error("op_lib::make_silu: input has zero tiles");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, x.num_tiles);
    else if (out.num_tiles != x.num_tiles)
        throw std::runtime_error("op_lib::make_silu: out.num_tiles != x.num_tiles");

    const std::string dir = resolve_kernel_dir(kernel_dir, "eltwise_unary");

    SiluOp op;
    op.l1_in  = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/silu.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/silu.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/silu.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 2> cbs = {{
        {0,  op.l1_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, op.l1_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_silu_args(dev, op, x, out);
    return op;
}

void set_silu_args(tt::foil::Device& dev, SiluOp& op,
                   const TensorDesc& x, const TensorDesc& out) {
    using R = tt::foil::RiscBinary;
    const uint64_t src_noc = tt::foil::make_noc_dram_addr(dev, x.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);
    std::array<uint32_t, 3> ra_brisc  = {(uint32_t)src_noc, (uint32_t)(src_noc >> 32), x.num_tiles};
    std::array<uint32_t, 3> ra_ncrisc = {(uint32_t)dst_noc, (uint32_t)(dst_noc >> 32), x.num_tiles};
    std::array<uint32_t, 1> ra_trisc  = {x.num_tiles};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
}

void execute(tt::foil::Device& dev, SiluOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
