// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: EmbeddingOp implementation (BRISC + NCRISC, no compute).

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::resolve_kernel_dir;

EmbeddingOp make_embedding(tt::foil::Device& dev,
                           const TensorDesc& table,
                           const std::vector<uint32_t>& token_ids,
                           uint32_t D,
                           TensorDesc& out,
                           tt::foil::CoreCoord core,
                           const std::string& kernel_dir) {
    if (!table.buf)
        throw std::runtime_error("op_lib::make_embedding: table.buf is null");
    if (token_ids.empty())
        throw std::runtime_error("op_lib::make_embedding: empty token_ids");

    const uint32_t N        = static_cast<uint32_t>(token_ids.size());
    const uint32_t D_bytes  = D * 2;
    const uint32_t out_bytes = N * D_bytes;
    if (!out.buf) {
        out.buf = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::DRAM, out_bytes);
        out.num_tiles = 0;  // row-major output, not tile-format
    } else if (out.buf->size_bytes < out_bytes) {
        throw std::runtime_error("op_lib::make_embedding: out buffer too small");
    }

    const std::string dir = resolve_kernel_dir(kernel_dir, "embedding");

    EmbeddingOp op;
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, out_bytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 2> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 1> cbs = {{
        {16, op.l1_out->device_addr, out_bytes, 1, out_bytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    const uint64_t emb_noc = tt::foil::make_noc_dram_addr(dev, table.buf->device_addr);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out.buf->device_addr);

    std::vector<uint32_t> ra_brisc;
    ra_brisc.reserve(4 + N);
    ra_brisc.push_back((uint32_t)emb_noc);
    ra_brisc.push_back((uint32_t)(emb_noc >> 32));
    ra_brisc.push_back(N);
    ra_brisc.push_back(D_bytes);
    for (uint32_t id : token_ids) ra_brisc.push_back(id);

    std::array<uint32_t, 3> ra_ncrisc = {(uint32_t)dst_noc, (uint32_t)(dst_noc >> 32), out_bytes};

    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::NCRISC, ra_ncrisc);

    return op;
}

void execute(tt::foil::Device& dev, EmbeddingOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
