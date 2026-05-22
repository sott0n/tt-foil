// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: EltwiseMulOp implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

EltwiseMulOp make_eltwise_mul(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              TensorDesc& out,
                              tt::foil::CoreCoord core,
                              const std::string& kernel_dir) {
    if (a.num_tiles == 0 || a.num_tiles != b.num_tiles)
        throw std::runtime_error("op_lib::make_eltwise_mul: a/b num_tiles mismatch");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, a.num_tiles);
    else if (out.num_tiles != a.num_tiles)
        throw std::runtime_error("op_lib::make_eltwise_mul: out.num_tiles != a.num_tiles");

    const std::string dir = resolve_kernel_dir(kernel_dir, "eltwise_binary");

    EltwiseMulOp op;
    op.l1_a   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_b   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/mul.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/mul.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/mul.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  op.l1_a->device_addr,   kTileBytes, 1, kTileBytes},
        {1,  op.l1_b->device_addr,   kTileBytes, 1, kTileBytes},
        {16, op.l1_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_eltwise_mul_args(dev, op, a, b, out);
    return op;
}

void set_eltwise_mul_args(tt::foil::Device& dev, EltwiseMulOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& out) {
    using R = tt::foil::RiscBinary;
    const uint64_t a_noc = tt::foil::make_noc_dram_addr(dev, a.buf->device_addr);
    const uint64_t b_noc = tt::foil::make_noc_dram_addr(dev, b.buf->device_addr);
    const uint64_t d_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 5> ra_brisc  = {(uint32_t)a_noc, (uint32_t)(a_noc >> 32),
                                         (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
                                         a.num_tiles};
    std::array<uint32_t, 3> ra_ncrisc = {(uint32_t)d_noc, (uint32_t)(d_noc >> 32), a.num_tiles};
    std::array<uint32_t, 1> ra_trisc  = {a.num_tiles};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
}

void execute(tt::foil::Device& dev, EltwiseMulOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
