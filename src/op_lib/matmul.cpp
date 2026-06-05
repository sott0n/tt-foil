// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// op_lib: MatMulOp — C = A · B with A [Mt × Kt], B [Kt × Nt], C [Mt × Nt]
// tiles row-major in DRAM. Mt/Kt/Nt are runtime args, so a single ELF set
// drives every Qwen3 matmul shape.

#include "tt_foil/ops.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <span>

#include "cb_config.hpp"
#include "dispatch.hpp"
#include "op_lib_internal.hpp"

namespace tt::foil::op_lib {

using detail::kTileBytes;
using detail::resolve_kernel_dir;

// Mb block height for the weight-stationary reader: how many A-rows the
// configured cb_a can hold (cb_a depth / Kt), capped at Mt. Mb=1 (the default,
// when cb_a was sized to a single Kt-deep row) reproduces the original
// per-mt-row matmul byte-for-byte. Derived from cb_a's size so set_*_args
// needs no extra parameter.
static uint32_t matmul_mb(const tt::foil::Buffer& l1_a, uint32_t Kt, uint32_t Mt) {
    uint32_t depth  = static_cast<uint32_t>(l1_a.size_bytes / kTileBytes);
    uint32_t mb_max = (depth >= Kt) ? (depth / Kt) : 1u;
    return (mb_max < Mt) ? mb_max : Mt;
}

MatMulOp make_matmul(tt::foil::Device& dev,
                     const TensorDesc& a, const TensorDesc& b,
                     TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     tt::foil::CoreCoord core,
                     const std::string& kernel_dir,
                     uint32_t mb_max) {
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
    // iter12/13/14: cb_a caches all Kt A tiles for one mt row; cb_b
    // batches Kt B tiles per nt. When L1 allows (6*Kt+2 ≤ ~855 KB,
    // i.e. Kt ≤ 142), cb_b is double-deep (2*Kt) so the reader can
    // prefetch the next nt-batch while the consumer is matmuling the
    // current one — overlapping NOC reads with compute.
    //
    // Per-core L1 ceilings (D_b = cb_b tile depth):
    //   D_b = 2*Kt: 6*Kt + 2 KB ≤ 855  →  Kt ≤ 142 (qkv/o/ffn-gateup/lm_head, all Kt=64)
    //   D_b =   Kt: 4*Kt + 2 KB ≤ 855  →  Kt ≤ 213 (FFN-down Kt=192 stays here)
    // Weight-stationary: cb_a caches mb_max A-rows (mb_max*Kt tiles) so the
    // reader streams the B weight slice ceil(Mt/mb_max)× instead of Mt×. The
    // DEFAULT mb_max=1 sizes cb_a to a single Kt-deep row — byte-identical CB
    // layout (and behavior) to the pre-WS matmul. cb_out gets a second slot
    // only when mb_max>1 so the writer can overlap; mb_max=1 keeps depth 1.
    const uint32_t cb_a_tiles = mb_max * Kt;
    const uint32_t cb_a_bytes = cb_a_tiles * kTileBytes;
    const uint32_t cb_b_tiles = ((6 * Kt + 2) <= 855) ? (2 * Kt) : Kt;
    const uint32_t cb_b_bytes = cb_b_tiles * kTileBytes;
    const uint32_t cb_out_tiles = (mb_max > 1) ? 2u : 1u;
    op.l1_a   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_a_bytes, core);
    op.l1_b   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_b_bytes, core);
    op.l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_out_tiles * kTileBytes, core);

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
        {0,  op.l1_a->device_addr,   cb_a_bytes, cb_a_tiles,   kTileBytes},
        {1,  op.l1_b->device_addr,   cb_b_bytes, cb_b_tiles,   kTileBytes},
        {16, op.l1_out->device_addr, cb_out_tiles * kTileBytes, cb_out_tiles, kTileBytes},
    }};
    tt::foil::register_cbs(dev, *op.kernel, cbs);

    set_matmul_args(dev, op, a, b, out, Mt, Kt, Nt);
    return op;
}

void set_matmul_args(tt::foil::Device& dev, MatMulOp& op,
                     const TensorDesc& a, const TensorDesc& b,
                     const TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    // Single-core call: Nt_stride collapses to Nt (no column sharding).
    set_matmul_args(dev, op, a, b, out, Mt, Kt, Nt, /*Nt_stride=*/Nt,
                    /*a_tile_offset=*/0,
                    /*b_tile_offset=*/0,
                    /*out_tile_offset=*/0);
}

void set_matmul_args(tt::foil::Device& dev, MatMulOp& op,
                     const TensorDesc& a, const TensorDesc& b,
                     const TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     uint32_t Nt_stride,
                     uint64_t a_tile_offset,
                     uint64_t b_tile_offset,
                     uint64_t out_tile_offset) {
    using R = tt::foil::RiscBinary;
    constexpr std::size_t kTileBytes = 32 * 32 * 2;
    const uint64_t a_noc   = tt::foil::make_noc_dram_addr(
        dev, a.buf->device_addr + a_tile_offset * kTileBytes);
    const uint64_t b_noc   = tt::foil::make_noc_dram_addr(
        dev, b.buf->device_addr + b_tile_offset * kTileBytes);
    const uint64_t dst_noc = tt::foil::make_noc_dram_addr(
        dev, out.buf->device_addr + out_tile_offset * kTileBytes);

    const uint32_t Mb = matmul_mb(*op.l1_a, Kt, Mt);

    std::array<uint32_t, 9> ra_brisc = {
        (uint32_t)a_noc, (uint32_t)(a_noc >> 32),
        (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
        Mt, Kt, Nt, Nt_stride, Mb,
    };
    std::array<uint32_t, 4> ra_trisc  = {Mt, Kt, Nt, Mb};
    std::array<uint32_t, 6> ra_ncrisc = {
        (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
        Mt, Nt, Nt_stride, Mb,
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

// ---------------------------------------------------------------------------
// MatMulGrid — Nt-sharded matmul across N Tensix cores.
// ---------------------------------------------------------------------------

// Allocate the per-core L1 CBs + load kernels for an Nt-sharded grid, but DO
// NOT set runtime args. Shared by make_matmul_grid (stock B) and the
// channel-sharded-weight path, which differ only in B addressing (RTA).
static MatMulGridOp alloc_matmul_grid(tt::foil::Device& dev, TensorDesc& out,
                                      uint32_t Mt, uint32_t Kt, uint32_t Nt,
                                      const std::vector<tt::foil::CoreCoord>& cores,
                                      const std::string& kernel_dir,
                                      uint32_t mb_max) {
    if (cores.empty())
        throw std::runtime_error("op_lib::make_matmul_grid: cores must be non-empty");
    const uint32_t n_cores = static_cast<uint32_t>(cores.size());
    // Ragged shards allowed: when Nt % n_cores != 0 (e.g. lm_head Vt=4748
    // on 8 cores), the first `rem` cores get one extra column tile and
    // the rest get base = Nt / n_cores. Per-core RTAs carry their own
    // Nt_per_core + column offset; the kernel doesn't know n_cores.
    if (Nt < n_cores)
        throw std::runtime_error("op_lib::make_matmul_grid: Nt must be >= cores.size()");
    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, Mt * Nt);
    else if (out.num_tiles != Mt * Nt)
        throw std::runtime_error("op_lib::make_matmul_grid: out.num_tiles != Mt*Nt");

    const std::string dir = resolve_kernel_dir(kernel_dir, "matmul");
    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, dir + "/matmul.trisc0.elf"},
        {R::RiscId::TRISC1, dir + "/matmul.trisc1.elf"},
        {R::RiscId::TRISC2, dir + "/matmul.trisc2.elf"},
    }};

    MatMulGridOp op;
    op.kernels.reserve(n_cores);
    op.l1_bufs.reserve(static_cast<std::size_t>(n_cores) * 3);
    // iter12/13/14 + WS: see make_matmul. mb_max=1 (default) is the original
    // per-mt-row layout; mb_max>1 deepens cb_a to cache mb_max A-rows.
    const uint32_t cb_a_tiles = mb_max * Kt;
    const uint32_t cb_a_bytes = cb_a_tiles * kTileBytes;
    const uint32_t cb_b_tiles = ((6 * Kt + 2) <= 855) ? (2 * Kt) : Kt;
    const uint32_t cb_b_bytes = cb_b_tiles * kTileBytes;
    const uint32_t cb_out_tiles = (mb_max > 1) ? 2u : 1u;
    for (const auto& core : cores) {
        auto l1_a   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_a_bytes, core);
        auto l1_b   = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_b_bytes, core);
        auto l1_out = tt::foil::allocate_buffer(dev, tt::foil::BufferLocation::L1, cb_out_tiles * kTileBytes, core);
        auto kernel = tt::foil::load_kernel(dev, bins, core);
        std::array<tt::foil::CbConfig, 3> cbs = {{
            {0,  l1_a->device_addr,   cb_a_bytes, cb_a_tiles,   kTileBytes},
            {1,  l1_b->device_addr,   cb_b_bytes, cb_b_tiles,   kTileBytes},
            {16, l1_out->device_addr, cb_out_tiles * kTileBytes, cb_out_tiles, kTileBytes},
        }};
        tt::foil::register_cbs(dev, *kernel, cbs);
        op.kernels.push_back(kernel);
        op.l1_bufs.push_back(std::move(l1_a));
        op.l1_bufs.push_back(std::move(l1_b));
        op.l1_bufs.push_back(std::move(l1_out));
    }
    return op;
}

MatMulGridOp make_matmul_grid(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              TensorDesc& out,
                              uint32_t Mt, uint32_t Kt, uint32_t Nt,
                              const std::vector<tt::foil::CoreCoord>& cores,
                              const std::string& kernel_dir,
                              uint32_t mb_max) {
    if (a.num_tiles != Mt * Kt)
        throw std::runtime_error("op_lib::make_matmul_grid: a.num_tiles != Mt*Kt");
    if (b.num_tiles != Kt * Nt)
        throw std::runtime_error("op_lib::make_matmul_grid: b.num_tiles != Kt*Nt");
    auto op = alloc_matmul_grid(dev, out, Mt, Kt, Nt, cores, kernel_dir, mb_max);
    set_matmul_grid_args(dev, op, a, b, out, Mt, Kt, Nt, cores);
    return op;
}

void set_matmul_grid_args(tt::foil::Device& dev, MatMulGridOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& out,
                          uint32_t Mt, uint32_t Kt, uint32_t Nt,
                          const std::vector<tt::foil::CoreCoord>& cores) {
    using R = tt::foil::RiscBinary;
    const uint32_t n_cores      = static_cast<uint32_t>(cores.size());
    const uint32_t base         = Nt / n_cores;
    const uint32_t rem          = Nt % n_cores;
    const uint64_t a_dev_base   = a.buf->device_addr;
    const uint64_t b_dev_base   = b.buf->device_addr;
    const uint64_t out_dev_base = out.buf->device_addr;
    // cb_a depth is uniform across the grid (built from one Kt/mb_max).
    const uint32_t Mb = matmul_mb(*op.l1_bufs[0], Kt, Mt);

    uint32_t col_off_tiles = 0;
    for (uint32_t c = 0; c < n_cores; ++c) {
        const uint32_t Nt_per_core = base + (c < rem ? 1u : 0u);
        const uint64_t col_off_bytes =
            static_cast<uint64_t>(col_off_tiles) * kTileBytes;
        const uint64_t a_noc   = tt::foil::make_noc_dram_addr(dev, a_dev_base);
        const uint64_t b_noc   = tt::foil::make_noc_dram_addr(dev, b_dev_base   + col_off_bytes);
        const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out_dev_base + col_off_bytes);

        std::array<uint32_t, 9> ra_brisc = {
            (uint32_t)a_noc, (uint32_t)(a_noc >> 32),
            (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
            Mt, Kt, Nt_per_core, /*Nt_stride=*/Nt, Mb,
        };
        std::array<uint32_t, 4> ra_trisc = {Mt, Kt, Nt_per_core, Mb};
        std::array<uint32_t, 6> ra_ncrisc = {
            (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
            Mt, Nt_per_core, /*Nt_stride=*/Nt, Mb,
        };
        col_off_tiles += Nt_per_core;
        auto& k = *op.kernels[c];
        tt::foil::set_runtime_args(dev, k, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC0, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC1, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC2, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::NCRISC, ra_ncrisc);
    }
}

void set_matmul_grid_args_sharded(tt::foil::Device& dev, MatMulGridOp& op,
                                  const TensorDesc& a, const TensorDesc& b,
                                  const TensorDesc& out,
                                  uint32_t Mt, uint32_t Kt, uint32_t Nt,
                                  const std::vector<tt::foil::CoreCoord>& cores,
                                  uint32_t n_channels) {
    using R = tt::foil::RiscBinary;
    const uint32_t n_cores      = static_cast<uint32_t>(cores.size());
    const uint32_t base         = Nt / n_cores;
    const uint32_t rem          = Nt % n_cores;
    const uint64_t a_dev_base   = a.buf->device_addr;
    const uint64_t b_dev_base   = b.buf->device_addr;
    const uint64_t out_dev_base = out.buf->device_addr;
    const uint32_t Mb = matmul_mb(*op.l1_bufs[0], Kt, Mt);

    uint32_t col_off_tiles = 0;
    for (uint32_t c = 0; c < n_cores; ++c) {
        const uint32_t Nt_per_core = base + (c < rem ? 1u : 0u);
        const uint64_t col_off_bytes =
            static_cast<uint64_t>(col_off_tiles) * kTileBytes;
        // B from this core's channel; A + out stay on channel 0.
        const uint64_t a_noc   = tt::foil::make_noc_dram_addr(dev, a_dev_base);
        const uint64_t b_noc   = tt::foil::make_noc_dram_addr_channel(
            dev, c % n_channels, b_dev_base + col_off_bytes);
        const uint64_t dst_noc = tt::foil::make_noc_dram_addr(dev, out_dev_base + col_off_bytes);

        std::array<uint32_t, 9> ra_brisc = {
            (uint32_t)a_noc, (uint32_t)(a_noc >> 32),
            (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
            Mt, Kt, Nt_per_core, /*Nt_stride=*/Nt, Mb,
        };
        std::array<uint32_t, 4> ra_trisc = {Mt, Kt, Nt_per_core, Mb};
        std::array<uint32_t, 6> ra_ncrisc = {
            (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
            Mt, Nt_per_core, /*Nt_stride=*/Nt, Mb,
        };
        col_off_tiles += Nt_per_core;
        auto& k = *op.kernels[c];
        tt::foil::set_runtime_args(dev, k, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC0, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC1, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC2, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::NCRISC, ra_ncrisc);
    }
}

void execute(tt::foil::Device& dev, MatMulGridOp& op) {
    std::vector<tt::foil::Kernel*> ptrs;
    ptrs.reserve(op.kernels.size());
    for (auto& k : op.kernels) ptrs.push_back(k.get());
    tt::foil::dispatch_execute_multi(dev, std::span<tt::foil::Kernel* const>(ptrs.data(), ptrs.size()));
}

// ---------------------------------------------------------------------------
// make_matmul_grid_cached — persistent variant.
// ---------------------------------------------------------------------------

MatMulGridOp make_matmul_grid_cached(tt::foil::Device& dev,
                                     const TensorDesc& a, const TensorDesc& b,
                                     TensorDesc& out,
                                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                                     const std::vector<tt::foil::CoreCoord>& cores,
                                     const std::string& kernel_dir,
                                     uint32_t mb_max) {
    if (cores.empty())
        throw std::runtime_error("op_lib::make_matmul_grid_cached: cores must be non-empty");

    // Cache key: (Kt, mb_max, n_cores, first_core). Kt + mb_max determine the
    // L1 CB layout (cb_a depth = mb_max*Kt); n_cores + first_core distinguish
    // disjoint grids. Mt/Nt are RTAs so one entry serves many shapes — but a
    // decode (mb_max=1) and prefill (mb_max>1) caller on the same cores get
    // separate pinned entries, since their cb_a depths differ.
    struct Key {
        uint32_t Kt;
        uint32_t mb_max;
        uint32_t n_cores;
        uint32_t x0;
        uint32_t y0;
        bool operator==(const Key& o) const noexcept {
            return Kt == o.Kt && mb_max == o.mb_max && n_cores == o.n_cores
                && x0 == o.x0 && y0 == o.y0;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            std::size_t h = k.Kt;
            h = h * 1315423911u + k.mb_max;
            h = h * 1315423911u + k.n_cores;
            h = h * 1315423911u + k.x0;
            h = h * 1315423911u + k.y0;
            return h;
        }
    };
    static thread_local std::unordered_map<Key, std::unique_ptr<MatMulGridOp>, KeyHash> g_cache;

    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, Mt * Nt);

    Key key{Kt, mb_max, static_cast<uint32_t>(cores.size()), cores[0].x, cores[0].y};
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
        auto op_ptr = std::make_unique<MatMulGridOp>(
            make_matmul_grid(dev, a, b, out, Mt, Kt, Nt, cores, kernel_dir, mb_max));
        // Pin every per-core kernel so release_kernels / reset_l1 from
        // surrounding transient ops on the same core doesn't disturb this
        // grid. Watermark on each pinned core freezes the L1 + kernel_config
        // high marks past our cb_a + cb_b + cb_out + kernel_text allocations.
        for (std::size_t i = 0; i < cores.size(); ++i) {
            tt::foil::pin_persistent(dev, *op_ptr->kernels[i], cores[i]);
        }
        it = g_cache.emplace(key, std::move(op_ptr)).first;
    }
    // RTA refresh — Mt/Nt/buffer addresses may have changed since last call.
    set_matmul_grid_args(dev, *it->second, a, b, out, Mt, Kt, Nt, cores);
    return *it->second;
}

// ---------------------------------------------------------------------------
// Channel-sharded weight path
// ---------------------------------------------------------------------------

ShardedWeight allocate_weight_sharded(tt::foil::Device& dev,
                                      const uint16_t* host_tiles,
                                      uint32_t Kt, uint32_t Nt, uint32_t n_shards) {
    if (n_shards == 0)
        throw std::runtime_error("allocate_weight_sharded: n_shards must be > 0");
    if (n_shards > tt::foil::num_dram_channels(dev))
        throw std::runtime_error("allocate_weight_sharded: n_shards > num_dram_channels");
    if (Nt < n_shards)
        throw std::runtime_error("allocate_weight_sharded: Nt must be >= n_shards");

    constexpr uint32_t kTileWords = 32 * 32;
    ShardedWeight w;
    w.Kt = Kt; w.Nt = Nt;
    const uint32_t base = Nt / n_shards;
    const uint32_t rem  = Nt % n_shards;
    std::vector<uint16_t> scratch;
    uint32_t col_off = 0;
    for (uint32_t c = 0; c < n_shards; ++c) {
        const uint32_t wc = base + (c < rem ? 1u : 0u);
        // Pack a contiguous [Kt x wc] kt-major sub-tensor: row kt's wc columns
        // come from host_tiles columns [col_off, col_off+wc).
        scratch.resize(static_cast<std::size_t>(Kt) * wc * kTileWords);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            const uint16_t* src =
                host_tiles + static_cast<std::size_t>(kt * Nt + col_off) * kTileWords;
            uint16_t* dst = scratch.data() + static_cast<std::size_t>(kt) * wc * kTileWords;
            std::memcpy(dst, src, static_cast<std::size_t>(wc) * kTileWords * sizeof(uint16_t));
        }
        const std::size_t bytes = static_cast<std::size_t>(Kt) * wc * kTileBytes;
        const uint64_t addr = tt::foil::alloc_dram_channel(dev, c, bytes, /*align=*/32);
        tt::foil::write_dram_channel(dev, c, addr, scratch.data(), bytes);
        w.channels.push_back(c);
        w.addrs.push_back(addr);
        w.widths.push_back(wc);
        w.col_off.push_back(col_off);
        col_off += wc;
    }
    return w;
}

// RTA for a ShardedWeight B: core c reads its shard from channel
// b.channels[c] at b.addrs[c], local Nt-stride = b.widths[c] (the shard is a
// standalone [Kt x widths[c]] tensor). A is replicated on channel 0; output is
// unsharded on channel 0, written at the shard's global column offset with the
// global Nt stride.
static void set_matmul_grid_args_weight_sharded(
        tt::foil::Device& dev, MatMulGridOp& op,
        const TensorDesc& a, const ShardedWeight& b, const TensorDesc& out,
        uint32_t Mt, const std::vector<tt::foil::CoreCoord>& cores) {
    using R = tt::foil::RiscBinary;
    const uint32_t Kt = b.Kt;
    const uint32_t Nt = b.Nt;
    const uint64_t a_dev_base   = a.buf->device_addr;
    const uint64_t out_dev_base = out.buf->device_addr;
    const uint32_t Mb = matmul_mb(*op.l1_bufs[0], Kt, Mt);
    const uint64_t a_noc = tt::foil::make_noc_dram_addr(dev, a_dev_base);
    for (uint32_t c = 0; c < static_cast<uint32_t>(cores.size()); ++c) {
        const uint32_t wc = b.widths[c];
        const uint64_t b_noc = tt::foil::make_noc_dram_addr_channel(
            dev, b.channels[c], b.addrs[c]);
        const uint64_t dst_noc = tt::foil::make_noc_dram_addr(
            dev, out_dev_base + static_cast<uint64_t>(b.col_off[c]) * kTileBytes);
        std::array<uint32_t, 9> ra_brisc = {
            (uint32_t)a_noc, (uint32_t)(a_noc >> 32),
            (uint32_t)b_noc, (uint32_t)(b_noc >> 32),
            Mt, Kt, wc, /*Nt_stride(B)=*/wc, Mb,
        };
        std::array<uint32_t, 4> ra_trisc = {Mt, Kt, wc, Mb};
        std::array<uint32_t, 6> ra_ncrisc = {
            (uint32_t)dst_noc, (uint32_t)(dst_noc >> 32),
            Mt, wc, /*Nt_stride(C)=*/Nt, Mb,
        };
        auto& k = *op.kernels[c];
        tt::foil::set_runtime_args(dev, k, R::RiscId::BRISC,  ra_brisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC0, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC1, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::TRISC2, ra_trisc);
        tt::foil::set_runtime_args(dev, k, R::RiscId::NCRISC, ra_ncrisc);
    }
}

MatMulGridOp make_matmul_grid_weight_sharded_cached(
        tt::foil::Device& dev,
        const TensorDesc& a, const ShardedWeight& b, TensorDesc& out,
        uint32_t Mt, const std::vector<tt::foil::CoreCoord>& cores,
        const std::string& kernel_dir, uint32_t mb_max) {
    if (cores.size() != b.channels.size())
        throw std::runtime_error(
            "make_matmul_grid_weight_sharded_cached: cores.size() != shard count");
    const uint32_t Kt = b.Kt;
    const uint32_t Nt = b.Nt;
    struct Key { uint32_t Kt, mb_max, n_cores, x0, y0;
        bool operator==(const Key& o) const noexcept {
            return Kt==o.Kt && mb_max==o.mb_max && n_cores==o.n_cores && x0==o.x0 && y0==o.y0;
        } };
    struct KeyHash { std::size_t operator()(const Key& k) const noexcept {
        std::size_t h=k.Kt; h=h*1315423911u+k.mb_max; h=h*1315423911u+k.n_cores;
        h=h*1315423911u+k.x0; h=h*1315423911u+k.y0; return h; } };
    static thread_local std::unordered_map<Key, std::unique_ptr<MatMulGridOp>, KeyHash> g_cache;

    if (out.num_tiles == 0)
        out = allocate_tensor_dram(dev, Mt * Nt);

    Key key{Kt, mb_max, (uint32_t)cores.size(), cores[0].x, cores[0].y};
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
        auto op_ptr = std::make_unique<MatMulGridOp>(
            alloc_matmul_grid(dev, out, Mt, Kt, Nt, cores, kernel_dir, mb_max));
        for (std::size_t i = 0; i < cores.size(); ++i)
            tt::foil::pin_persistent(dev, *op_ptr->kernels[i], cores[i]);
        it = g_cache.emplace(key, std::move(op_ptr)).first;
    }
    set_matmul_grid_args_weight_sharded(dev, *it->second, a, b, out, Mt, cores);
    return *it->second;
}

}  // namespace tt::foil::op_lib
