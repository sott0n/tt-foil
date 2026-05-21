// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: MatMulOp — C = A · B with A [Mt × Kt], B [Kt × Nt], C [Mt × Nt]
// tiles row-major in DRAM. Mt/Kt/Nt are runtime args, so a single ELF set
// drives every Qwen3 matmul shape.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

MatMulOp make_matmul(tt::foil::Device& dev,
                     const TensorDesc& a, const TensorDesc& b,
                     TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     tt::foil::CoreCoord core,
                     const std::string& kernel_dir) {
    if (a.num_tiles != Mt * Kt)
        throw std::runtime_error("op_lib::make_matmul: a.num_tiles != Mt*Kt");
    if (b.num_tiles != Kt * Nt)
        throw std::runtime_error("op_lib::make_matmul: b.num_tiles != Kt*Nt");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, Mt * Nt);
    else if (out.num_tiles != Mt * Nt)
        throw std::runtime_error("op_lib::make_matmul: out.num_tiles != Mt*Nt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "matmul");

    MatMulOp op;
    // Single-tile L1 staging for A, B, OUT (reader pushes one tile at a time).
    op.l1_a   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_b   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/matmul.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/matmul.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/matmul.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 3> cbs = {{
        {0,  op.l1_a->device_addr,   kTileBytes, 1, kTileBytes},
        {1,  op.l1_b->device_addr,   kTileBytes, 1, kTileBytes},
        {16, op.l1_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_matmul_args(dev, op, a, b, out, Mt, Kt, Nt);
    return op;
}

void set_matmul_args(tt::foil::Device& dev, MatMulOp& op,
                     const TensorDesc& a, const TensorDesc& b,
                     const TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    using R = tt::foil::RiscBinary;
    const uint64_t a_noc   = tt::foil::make_noc_dram_addr(dev, a.buf->device_addr);
    const uint64_t b_noc   = tt::foil::make_noc_dram_addr(dev, b.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 7> ra_brisc = {
        (uint32_t)a_noc, (uint32_t)(a_noc >> 32),
        (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
        Mt, Kt, Nt,
    };
    std::array<uint32_t, 3> ra_trisc  = {Mt, Kt, Nt};
    std::array<uint32_t, 3> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32), Mt * Nt,
    };

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
}

void execute(tt::foil::Device& dev, MatMulOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
