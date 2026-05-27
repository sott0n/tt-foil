// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: ArgmaxRow0Op implementation (BRISC scan + NCRISC writer).

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::resolve_kernel_dir;

// 2048-byte (one tile) CB page: BRISC reads each logits tile into this
// slot, then overwrites the first 4 bytes with the final uint32_t result
// for NCRISC to drain.
constexpr uint32_t kArgmaxCbBytes = 2048;

ArgmaxRow0Op make_argmax_row0(tt::foil::Device& dev,
                              const TensorDesc& logits,
                              uint32_t Vt,
                              TensorDesc& out,
                              tt::foil::CoreCoord core,
                              const std::string& kernel_dir,
                              uint32_t row_in_tile) {
    if (!logits.buf)
        throw std::runtime_error("op_lib::make_argmax_row0: logits.buf is null");
    if (Vt == 0)
        throw std::runtime_error("op_lib::make_argmax_row0: Vt must be > 0");
    if (row_in_tile >= 32)
        throw std::runtime_error("op_lib::make_argmax_row0: row_in_tile must be < 32");

    if (!out.buf) {
        out.buf = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, 4);
        out.num_tiles = 0;
    } else if (out.buf->size_bytes < 4) {
        throw std::runtime_error("op_lib::make_argmax_row0: out buffer < 4 B");
    }

    const std::string dir = resolve_kernel_dir(kernel_dir, "argmax_row0");

    ArgmaxRow0Op op;
    op.l1_out = tt::foil::allocate_buffer(
        dev, tt::foil::BufferLocation::L1, kArgmaxCbBytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 2> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 1> cbs = {{
        {16, op.l1_out->device_addr, kArgmaxCbBytes, 1, kArgmaxCbBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_argmax_row0_args(dev, op, logits, Vt, out, row_in_tile);
    return op;
}

void set_argmax_row0_args(tt::foil::Device& dev, ArgmaxRow0Op& op,
                          const TensorDesc& logits, uint32_t Vt,
                          const TensorDesc& out,
                          uint32_t row_in_tile) {
    using R = tt::foil::RiscBinary;
    const uint64_t src_noc = tt::foil::make_noc_dram_addr(dev, logits.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::array<uint32_t, 4> ra_brisc = {
        static_cast<uint32_t>(src_noc),
        static_cast<uint32_t>(src_noc >> 32),
        Vt,
        row_in_tile,
    };
    std::array<uint32_t, 2> ra_ncrisc = {
        static_cast<uint32_t>(dst_noc),
        static_cast<uint32_t>(dst_noc >> 32),
    };
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);
}

void execute(tt::foil::Device& dev, ArgmaxRow0Op& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
