// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: KvSnapshotOp implementation (BRISC-only, no compute).
// See `ops/kv_snapshot/snapshot.cpp` for the kernel logic and the L1
// scratch layout this op_lib allocates.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "cb_config.hpp"
#include "op_cache.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::resolve_kernel_dir;

static KvSnapshotOp make_kv_snapshot_impl(
        tt::foil::Device& dev,
        const TensorDesc& kt_pre, const TensorDesc& v_pre,
        const TensorDesc& kt_cache, const TensorDesc& v_cache,
        uint32_t kSt, uint32_t kNkDt, uint32_t kStKv,
        tt::foil::CoreCoord core,
        const std::string& kernel_dir) {
    if (!kt_pre.buf || !v_pre.buf || !kt_cache.buf || !v_cache.buf)
        throw std::runtime_error("op_lib::make_kv_snapshot: null buffer");
    if (kSt == 0 || kNkDt == 0 || kStKv == 0)
        throw std::runtime_error("op_lib::make_kv_snapshot: zero dim");
    if (kStKv < kSt)
        throw std::runtime_error("op_lib::make_kv_snapshot: kStKv < kSt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "kv_snapshot");

    // L1 scratch: K^T source (kNkDt*kSt tiles) + V source (kSt*kNkDt tiles)
    // + one tile of zeros for decode-slot fill.
    constexpr uint32_t kTileBytes = 2048;
    const std::size_t kt_bytes = static_cast<std::size_t>(kNkDt) * kSt * kTileBytes;
    const std::size_t v_bytes  = static_cast<std::size_t>(kSt) * kNkDt * kTileBytes;
    const std::size_t scratch_bytes = kt_bytes + v_bytes + kTileBytes;

    KvSnapshotOp op;
    op.l1_scratch = tt::foil::allocate_buffer(
        dev, tt::foil::BufferLocation::L1, scratch_bytes, core);

    using R = tt::foil::RiscBinary;
    std::array<R, 1> bins = {{
        {R::RiscId::BRISC, dir + "/snapshot.brisc.elf"},
    }};
    op.kernel = tt::foil::load_kernel(dev, bins, core);

    std::array<tt::foil::CbConfig, 1> cbs = {{
        {16, op.l1_scratch->device_addr, static_cast<uint32_t>(scratch_bytes),
         1, static_cast<uint32_t>(scratch_bytes)},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_kv_snapshot_args(dev, op, kt_pre, v_pre, kt_cache, v_cache,
                         kSt, kNkDt, kStKv);
    return op;
}

KvSnapshotOp make_kv_snapshot(tt::foil::Device& dev,
                              const TensorDesc& kt_pre,
                              const TensorDesc& v_pre,
                              const TensorDesc& kt_cache,
                              const TensorDesc& v_cache,
                              uint32_t kSt, uint32_t kNkDt, uint32_t kStKv,
                              tt::foil::CoreCoord core,
                              const std::string& kernel_dir) {
    static thread_local OpCache<KvSnapshotOp> g_cache;
    ShapeKey key{
        .op_name  = "kv_snapshot",
        .params   = {kSt, kNkDt, kStKv, 0, 0, 0, 0, 0},
        .core_key = core_key_of(core),
    };
    KvSnapshotOp& op = g_cache.get_or_create(dev, core, key, [&] {
        return make_kv_snapshot_impl(dev, kt_pre, v_pre, kt_cache, v_cache,
                                     kSt, kNkDt, kStKv, core, kernel_dir);
    });
    set_kv_snapshot_args(dev, op, kt_pre, v_pre, kt_cache, v_cache,
                         kSt, kNkDt, kStKv);
    return op;
}

void set_kv_snapshot_args(tt::foil::Device& dev, KvSnapshotOp& op,
                          const TensorDesc& kt_pre, const TensorDesc& v_pre,
                          const TensorDesc& kt_cache, const TensorDesc& v_cache,
                          uint32_t kSt, uint32_t kNkDt, uint32_t kStKv) {
    using R = tt::foil::RiscBinary;
    const uint64_t kt_src = tt::foil::make_noc_dram_addr(dev, kt_pre.buf->device_addr);
    const uint64_t v_src  = tt::foil::make_noc_dram_addr(dev, v_pre.buf->device_addr);
    const uint64_t kt_dst = tt::foil::make_noc_dram_addr(dev, kt_cache.buf->device_addr);
    const uint64_t v_dst  = tt::foil::make_noc_dram_addr(dev, v_cache.buf->device_addr);

    std::array<uint32_t, 11> ra = {
        static_cast<uint32_t>(kt_src), static_cast<uint32_t>(kt_src >> 32),
        static_cast<uint32_t>(v_src),  static_cast<uint32_t>(v_src  >> 32),
        static_cast<uint32_t>(kt_dst), static_cast<uint32_t>(kt_dst >> 32),
        static_cast<uint32_t>(v_dst),  static_cast<uint32_t>(v_dst  >> 32),
        kSt, kNkDt, kStKv,
    };
    tt::foil::set_runtime_args(dev, *op.kernel, R::RiscId::BRISC, ra);
}

void execute(tt::foil::Device& dev, KvSnapshotOp& op) {
    tt::foil::execute(dev, *op.kernel);
}

}  // namespace tt::foil::op_lib
