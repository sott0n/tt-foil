// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: Transpose2dOp implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

Transpose2dOp make_transpose_2d(tt::foil::Device& dev,
                                const TensorDesc& in, TensorDesc& out,
                                uint32_t Rt, uint32_t Ct,
                                tt::foil::CoreCoord core,
                                const std::string& kernel_dir) {
    const uint32_t total_tiles = Rt * Ct;
    if (in.num_tiles != total_tiles)
        throw std::runtime_error("op_lib::make_transpose_2d: in.num_tiles != Rt*Ct");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, total_tiles);
    else if (out.num_tiles != total_tiles)
        throw std::runtime_error("op_lib::make_transpose_2d: out.num_tiles != Rt*Ct");

    const std::string dir = resolve_kernel_dir(kernel_dir, "transpose_2d");

    Transpose2dOp op;
    op.l1_in  = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/transpose_2d.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/transpose_2d.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/transpose_2d.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 2> cbs = {{
        {0,  op.l1_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, op.l1_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_transpose_2d_args(dev, op, in, out, Rt, Ct);
    return op;
}

void set_transpose_2d_args(tt::foil::Device& dev, Transpose2dOp& op,
                           const TensorDesc& in, const TensorDesc& out,
                           uint32_t Rt, uint32_t Ct) {
    using R = tt::foil::RiscBinary;
    const uint32_t total_tiles = Rt * Ct;
    const uint64_t in_noc  = tt::foil::make_noc_dram_addr(dev, in.buf->device_addr);
    const uint64_t out_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 4> ra_brisc  = {(uint32_t)in_noc,  (uint32_t)(in_noc >> 32),  Rt, Ct};
    std::array<uint32_t, 3> ra_ncrisc = {(uint32_t)out_noc, (uint32_t)(out_noc >> 32), total_tiles};
    std::array<uint32_t, 1> ra_trisc  = {total_tiles};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
}

void execute(tt::foil::Device& dev, Transpose2dOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
