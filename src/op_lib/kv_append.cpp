// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: KvAppendOp implementation (BRISC-only, no compute).
// See `ops/kv_append/append.cpp` for the kernel logic and the L1
// scratch layout this op_lib has to allocate.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::resolve_kernel_dir;

KvAppendOp make_kv_append(tt::foil::Device& dev,
                          const TensorDesc& kr, const TensorDesc& v,
                          const TensorDesc& kt_cache,
                          const TensorDesc& v_cache,
                          uint32_t slot1_r, uint32_t Nk, uint32_t StKv,
                          tt::foil::CoreCoord core,
                          const std::string& kernel_dir) {
    if (!kr.buf || !v.buf || !kt_cache.buf || !v_cache.buf)
        throw std::runtime_error("op_lib::make_kv_append: null buffer");
    if (Nk == 0 || StKv == 0)
        throw std::runtime_error("op_lib::make_kv_append: zero dim");

    const std::string dir = resolve_kernel_dir(kernel_dir, "kv_append");

    // L1 scratch: 2 × Nk tile-bytes for T_Kr/T_V slurp + 512 B for the
    // K-cache face read-modify-write buffer.
    const std::size_t scratch_bytes =
        static_cast<std::size_t>(2) * Nk * 2048 + /*face_buf=*/512;

    KvAppendOp op;
    op.l1_scratch = tt::foil::allocate_buffer(
        dev, tt::foil::BufferLocation::L1, scratch_bytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 1> bins = {{
        {R::RiscId::BRISC, dir + "/append.brisc.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 1> cbs = {{
        {16, op.l1_scratch->device_addr, static_cast<uint32_t>(scratch_bytes),
         1, static_cast<uint32_t>(scratch_bytes)},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_kv_append_args(dev, op, kr, v, kt_cache, v_cache, slot1_r, Nk, StKv);
    return op;
}

void set_kv_append_args(tt::foil::Device& dev, KvAppendOp& op,
                        const TensorDesc& kr, const TensorDesc& v,
                        const TensorDesc& kt_cache, const TensorDesc& v_cache,
                        uint32_t slot1_r, uint32_t Nk, uint32_t StKv) {
    using R = tt::foil::RiscBinary;
    const uint64_t kr_noc = tt::foil::make_noc_dram_addr(dev, kr.buf->device_addr);
    const uint64_t v_noc  = tt::foil::make_noc_dram_addr(dev, v.buf->device_addr);
    const uint64_t kt_noc = tt::foil::make_noc_dram_addr(dev, kt_cache.buf->device_addr);
    const uint64_t vc_noc = tt::foil::make_noc_dram_addr(dev, v_cache.buf->device_addr);

    std::array<uint32_t, 11> ra = {
        static_cast<uint32_t>(kr_noc), static_cast<uint32_t>(kr_noc >> 32),
        static_cast<uint32_t>(v_noc),  static_cast<uint32_t>(v_noc  >> 32),
        static_cast<uint32_t>(kt_noc), static_cast<uint32_t>(kt_noc >> 32),
        static_cast<uint32_t>(vc_noc), static_cast<uint32_t>(vc_noc >> 32),
        slot1_r, Nk, StKv,
    };
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC, ra);
}

void execute(tt::foil::Device& dev, KvAppendOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
