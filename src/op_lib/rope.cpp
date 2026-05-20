// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: RopeOp implementation.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

RopeOp make_rope(tt::foil::Device& dev,
                 const TensorDesc& x,
                 const TensorDesc& cos, const TensorDesc& sin,
                 TensorDesc& out,
                 uint32_t St, uint32_t num_heads, uint32_t Dt_half,
                 tt::foil::CoreCoord core,
                 const std::string& kernel_dir) {
    const uint32_t Dt        = 2 * Dt_half;
    const uint32_t total_Dt  = num_heads * Dt;
    const uint32_t total_iters = St * num_heads * Dt_half;

    if (x.num_tiles != St * total_Dt)
        throw std::runtime_error("op_lib::make_rope: x.num_tiles != St * num_heads * Dt");
    if (cos.num_tiles != St * Dt_half || sin.num_tiles != St * Dt_half)
        throw std::runtime_error("op_lib::make_rope: cos/sin num_tiles != St * Dt_half");

    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, x.num_tiles);
    else if (out.num_tiles != x.num_tiles)
        throw std::runtime_error("op_lib::make_rope: out.num_tiles != x.num_tiles");

    const std::string dir = resolve_kernel_dir(kernel_dir, "rope");

    RopeOp op;

    // L1 CBs: x0(0), x1(1), cos(2), sin(3), tmp0(4), tmp1(5), out(16)
    auto alloc_cb = [&](std::size_t bytes) {
        return tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, bytes, core);
    };
    op.l1_cbs.resize(7);
    op.l1_cbs[0] = alloc_cb(kTileBytes);        // cb_x0
    op.l1_cbs[1] = alloc_cb(kTileBytes);        // cb_x1
    op.l1_cbs[2] = alloc_cb(kTileBytes);        // cb_cos
    op.l1_cbs[3] = alloc_cb(kTileBytes);        // cb_sin
    op.l1_cbs[4] = alloc_cb(kTileBytes);        // cb_tmp0
    op.l1_cbs[5] = alloc_cb(kTileBytes);        // cb_tmp1
    op.l1_cbs[6] = alloc_cb(2 * kTileBytes);    // cb_out (depth=2)

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/rope.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/rope.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/rope.trisc2.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 7> cbs = {{
        {0,  op.l1_cbs[0]->device_addr, kTileBytes,     1, kTileBytes},
        {1,  op.l1_cbs[1]->device_addr, kTileBytes,     1, kTileBytes},
        {2,  op.l1_cbs[2]->device_addr, kTileBytes,     1, kTileBytes},
        {3,  op.l1_cbs[3]->device_addr, kTileBytes,     1, kTileBytes},
        {4,  op.l1_cbs[4]->device_addr, kTileBytes,     1, kTileBytes},
        {5,  op.l1_cbs[5]->device_addr, kTileBytes,     1, kTileBytes},
        {16, op.l1_cbs[6]->device_addr, 2 * kTileBytes, 2, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    const uint64_t x_noc   = tt::foil::make_noc_dram_addr(dev, x.buf->device_addr);
    const uint64_t cos_noc = tt::foil::make_noc_dram_addr(dev, cos.buf->device_addr);
    const uint64_t sin_noc = tt::foil::make_noc_dram_addr(dev, sin.buf->device_addr);
    const uint64_t out_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    // BRISC reader: x_noc, cos_noc, sin_noc, St, num_heads, Dt_half
    std::array<uint32_t, 9> ra_brisc = {{
        (uint32_t)x_noc,   (uint32_t)(x_noc >> 32),
        (uint32_t)cos_noc, (uint32_t)(cos_noc >> 32),
        (uint32_t)sin_noc, (uint32_t)(sin_noc >> 32),
        St, num_heads, Dt_half,
    }};

    // NCRISC writer: out_noc, St, num_heads, Dt_half
    std::array<uint32_t, 5> ra_ncrisc = {{
        (uint32_t)out_noc, (uint32_t)(out_noc >> 32),
        St, num_heads, Dt_half,
    }};

    // TRISCs: total_iters
    std::array<uint32_t, 1> ra_trisc = {{total_iters}};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC0, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC1, ra_trisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::TRISC2, ra_trisc);

    return op;
}

void execute(tt::foil::Device& dev, RopeOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
