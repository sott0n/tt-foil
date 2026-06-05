// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Primitive Op Library — host-side abstraction over tt-foil kernels.
//
// Phase 1-A first cut: wrappers around the six Phase 1-B kernels (SiLU,
// ElementwiseMul, RMSNorm, Softmax, MHA, Embedding). Goals:
//   - hide load_kernel / register_cbs / set_runtime_args boilerplate
//   - compose ops by passing TensorDesc handles instead of juggling Buffer
//     + L1 CB layouts
//   - keep the surface small so we can iterate the shape before wiring
//     full Qwen3 layers
//
// Conventions:
//   - tensors are stored in DRAM as tile-formatted BF16 streams
//   - tensor logical layout is described by `num_tiles` (linear stream);
//     ops that need 2D structure take explicit St/Dt arguments
//   - L1 CBs are implementation details owned by each Op handle

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

namespace tt::foil::op_lib {

// ---------------------------------------------------------------------------
// TensorDesc — a tile-stream tensor backed by a DRAM Buffer.
// ---------------------------------------------------------------------------
struct TensorDesc {
    std::shared_ptr<tt::foil::Buffer> buf;
    uint32_t num_tiles{0};                // total tile count in `buf`
    std::vector<uint32_t> shape;          // optional logical shape
};

// Allocate a DRAM-backed tile-stream tensor of `num_tiles` BF16 tiles.
TensorDesc allocate_tensor_dram(tt::foil::Device& dev, uint32_t num_tiles);

// =====================================================================
// SiLU
//   y = silu(x) = x * sigmoid(x)
// =====================================================================
struct SiluOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_in;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
SiluOp make_silu(tt::foil::Device& dev,
                 const TensorDesc& x, TensorDesc& out,
                 tt::foil::CoreCoord core = {},
                 const std::string& kernel_dir = "");
void set_silu_args(tt::foil::Device& dev, SiluOp& op,
                   const TensorDesc& x, const TensorDesc& out);
void execute(tt::foil::Device& dev, SiluOp& op);

// =====================================================================
// MatMul
//   C = A · B
//   A: [Mt × Kt] tiles row-major, B: [Kt × Nt] tiles row-major,
//   C: [Mt × Nt] tiles row-major. Single ELF — Mt/Kt/Nt are runtime args
//   so the same handle supports every Qwen3 matmul shape.
// =====================================================================
struct MatMulOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_a;
    std::shared_ptr<tt::foil::Buffer> l1_b;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
// mb_max = weight-stationary block height (A-rows cached in cb_a). Default 1
// reproduces the original per-mt-row matmul exactly (cb_a = Kt deep). Pass >1
// (prefill, Mt>1) to cache mb_max A-rows so the B weight slice is streamed
// ceil(Mt/mb_max)× instead of Mt× — caller must keep mb_max*Kt + cb_b in L1.
MatMulOp make_matmul(tt::foil::Device& dev,
                     const TensorDesc& a, const TensorDesc& b,
                     TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     tt::foil::CoreCoord core = {},
                     const std::string& kernel_dir = "",
                     uint32_t mb_max = 1);
// Update RTAs only (no kernel reload, no L1/CB alloc). Lets one op handle
// drive matmuls of arbitrary shape so the dispatch ELF-cache hits.
void set_matmul_args(tt::foil::Device& dev, MatMulOp& op,
                     const TensorDesc& a, const TensorDesc& b,
                     const TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt);
// Column-shard variant: this kernel handles tiles
//   B[kt, b_tile_offset + nt'] and writes C[mt, out_tile_offset + nt']
// for nt' in [0, Nt). B reads use Nt_stride as the row stride (= the
// global Nt), so a column-shard of B can be expressed as a per-core
// base offset + the global stride. out_tile_offset works the same way:
// each core writes its column slice into a single global C buffer at
// `out_base + (mt * Nt_stride + nt') * tile_bytes`.
void set_matmul_args(tt::foil::Device& dev, MatMulOp& op,
                     const TensorDesc& a, const TensorDesc& b,
                     const TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     uint32_t Nt_stride,
                     uint64_t a_tile_offset,
                     uint64_t b_tile_offset,
                     uint64_t out_tile_offset);
void execute(tt::foil::Device& dev, MatMulOp& op);

// =====================================================================
// MatMulGrid — Nt-sharded matmul across N Tensix cores. Each per-core
// kernel handles `Nt_per_core = Nt_global / cores.size()` consecutive
// output columns. B reads + C writes use Nt_global as their row stride
// so all cores read/write into one global B / one global C with no
// copying. A is shared (full Mt × Kt) — each core reads it identically.
// Single dispatch fires all kernels through dispatch_execute_multi.
// =====================================================================
struct MatMulGridOp {
    std::vector<std::shared_ptr<tt::foil::Kernel>> kernels;
    // Per-core L1 CBs in flattened groups of 3 (a, b, out) so we keep
    // ownership of every allocated L1 region in one place.
    std::vector<std::shared_ptr<tt::foil::Buffer>> l1_bufs;
};
MatMulGridOp make_matmul_grid(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              TensorDesc& out,
                              uint32_t Mt, uint32_t Kt, uint32_t Nt,
                              const std::vector<tt::foil::CoreCoord>& cores,
                              const std::string& kernel_dir = "",
                              uint32_t mb_max = 1);  // see make_matmul
void set_matmul_grid_args(tt::foil::Device& dev, MatMulGridOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& out,
                          uint32_t Mt, uint32_t Kt, uint32_t Nt,
                          const std::vector<tt::foil::CoreCoord>& cores);
// Multi-channel-B variant: identical to set_matmul_grid_args except core c
// reads its B column-slab from DRAM channel (c % n_channels) instead of
// channel 0, so n_channels reader cores stream weights from distinct DRAM
// channels concurrently. A (replicated) and out stay on channel 0. The B
// tensor must be present at b.buf->device_addr on every channel 0..n_channels-1
// (caller's responsibility — see write_dram_channel). Used to measure /
// exploit the per-channel decode read-bandwidth ceiling.
void set_matmul_grid_args_sharded(tt::foil::Device& dev, MatMulGridOp& op,
                                  const TensorDesc& a, const TensorDesc& b,
                                  const TensorDesc& out,
                                  uint32_t Mt, uint32_t Kt, uint32_t Nt,
                                  const std::vector<tt::foil::CoreCoord>& cores,
                                  uint32_t n_channels);
void execute(tt::foil::Device& dev, MatMulGridOp& op);

// Shape-keyed persistent matmul grid. Cache key is (Kt, n_cores, first_core),
// so all callers that share those three values reuse the same pinned
// kernel + L1 CB-backing buffers. Mt/Nt are runtime args, so different
// logical matmuls (qkv, o, ffn_gate+up, lm_head) all collapse onto one
// cache entry as long as their Kt + grid match. ffn_down (Kt=192) needs
// a different L1 layout, so it lives on a separate grid.
//
// First call: full make_matmul_grid + pin_persistent on every (kernel,
// core) pair, then set_matmul_grid_args. Subsequent calls: pure RTA
// refresh.
//
// Contract: cores list MUST be identical across calls with the same
// cache key. Callers MUST NOT call release_kernels / reset_l1 on the
// pinned cores (use disjoint grids for transient ops).
MatMulGridOp make_matmul_grid_cached(tt::foil::Device& dev,
                                     const TensorDesc& a, const TensorDesc& b,
                                     TensorDesc& out,
                                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                                     const std::vector<tt::foil::CoreCoord>& cores,
                                     const std::string& kernel_dir = "",
                                     uint32_t mb_max = 1);  // see make_matmul

// =====================================================================
// ElementwiseMul
//   y = x * z   (elementwise, per-tile)
// =====================================================================
struct EltwiseMulOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_a;
    std::shared_ptr<tt::foil::Buffer> l1_b;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
EltwiseMulOp make_eltwise_mul(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              TensorDesc& out,
                              tt::foil::CoreCoord core = {},
                              const std::string& kernel_dir = "");
void set_eltwise_mul_args(tt::foil::Device& dev, EltwiseMulOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& out);
void execute(tt::foil::Device& dev, EltwiseMulOp& op);

// =====================================================================
// SiluMul  (fused SwiGLU): out[i] = SiLU(a[i]) * b[i] per tile.
// Replaces the silu→mul two-op chain that pays a 2× dispatch floor.
// =====================================================================
struct SiluMulOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_a;
    std::shared_ptr<tt::foil::Buffer> l1_b;
    std::shared_ptr<tt::foil::Buffer> l1_s;   // intermediate SiLU(a)
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
SiluMulOp make_silu_mul(tt::foil::Device& dev,
                        const TensorDesc& a, const TensorDesc& b,
                        TensorDesc& out,
                        tt::foil::CoreCoord core = {},
                        const std::string& kernel_dir = "");
void set_silu_mul_args(tt::foil::Device& dev, SiluMulOp& op,
                       const TensorDesc& a, const TensorDesc& b,
                       const TensorDesc& out);
void execute(tt::foil::Device& dev, SiluMulOp& op);

// =====================================================================
// AddRmsNorm (fused): S = A + B; Y = RMSNorm(S, gamma).
// Replaces the add → rmsnorm two-op chain that pays 2× dispatch floor.
// Writes both S (residual sum, for the next residual add) and Y (normed)
// to separate DRAM destinations.
// =====================================================================
struct AddRmsNormOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    // L1 staging
    std::shared_ptr<tt::foil::Buffer> l1_a;
    std::shared_ptr<tt::foil::Buffer> l1_b;
    std::shared_ptr<tt::foil::Buffer> l1_reduce;
    std::shared_ptr<tt::foil::Buffer> l1_gamma;
    std::shared_ptr<tt::foil::Buffer> l1_eps;
    std::shared_ptr<tt::foil::Buffer> l1_x2;
    std::shared_ptr<tt::foil::Buffer> l1_var;
    std::shared_ptr<tt::foil::Buffer> l1_recip_sqrt;
    std::shared_ptr<tt::foil::Buffer> l1_x_normed;
    std::shared_ptr<tt::foil::Buffer> l1_sum;
    std::shared_ptr<tt::foil::Buffer> l1_s_out;
    std::shared_ptr<tt::foil::Buffer> l1_out;
    // DRAM constants
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
    std::shared_ptr<tt::foil::Buffer> dram_eps;
};
// a, b in; sum_out and normed_out are written. Sum/normed are
// allocated by the caller (typical pattern in qwen3_run).
AddRmsNormOp make_add_rmsnorm(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              const TensorDesc& gamma,
                              TensorDesc& sum_out, TensorDesc& normed_out,
                              uint32_t NCHt, uint32_t Wt,
                              float eps,
                              tt::foil::CoreCoord core = {},
                              const std::string& kernel_dir = "");
void set_add_rmsnorm_args(tt::foil::Device& dev, AddRmsNormOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& gamma,
                          const TensorDesc& sum_out, const TensorDesc& normed_out,
                          uint32_t NCHt, uint32_t Wt);
void execute(tt::foil::Device& dev, AddRmsNormOp& op);

// =====================================================================
// RmsNormRope (fused): Y = RoPE( RMSNorm(x, gamma), cos, sin ).
// Used for Q and K projections — replaces the rmsnorm_qk → rope chain.
// =====================================================================
struct RmsNormRopeOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_x;
    std::shared_ptr<tt::foil::Buffer> l1_reduce;
    std::shared_ptr<tt::foil::Buffer> l1_gamma;
    std::shared_ptr<tt::foil::Buffer> l1_eps;
    std::shared_ptr<tt::foil::Buffer> l1_x2;
    std::shared_ptr<tt::foil::Buffer> l1_var;
    std::shared_ptr<tt::foil::Buffer> l1_recip_sqrt;
    std::shared_ptr<tt::foil::Buffer> l1_x_normed;
    std::shared_ptr<tt::foil::Buffer> l1_cos;
    std::shared_ptr<tt::foil::Buffer> l1_sin;
    std::shared_ptr<tt::foil::Buffer> l1_normed;
    std::shared_ptr<tt::foil::Buffer> l1_tmp0;
    std::shared_ptr<tt::foil::Buffer> l1_tmp1;
    std::shared_ptr<tt::foil::Buffer> l1_out;
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
    std::shared_ptr<tt::foil::Buffer> dram_eps;
};
// x: [St*num_heads, Dt] tiles; gamma: [Dt]; cos/sin: [St, Dt_half].
// out: [St, num_heads*Dt] (packed multi-head RoPE-rotated).
RmsNormRopeOp make_rmsnorm_rope(tt::foil::Device& dev,
                                const TensorDesc& x, const TensorDesc& gamma,
                                const TensorDesc& cos, const TensorDesc& sin,
                                TensorDesc& out,
                                uint32_t St, uint32_t num_heads, uint32_t Dt_half,
                                float eps,
                                tt::foil::CoreCoord core = {},
                                const std::string& kernel_dir = "");
void set_rmsnorm_rope_args(tt::foil::Device& dev, RmsNormRopeOp& op,
                           const TensorDesc& x, const TensorDesc& gamma,
                           const TensorDesc& cos, const TensorDesc& sin,
                           const TensorDesc& out,
                           uint32_t St, uint32_t num_heads, uint32_t Dt_half);
void execute(tt::foil::Device& dev, RmsNormRopeOp& op);

// =====================================================================
// ElementwiseAdd
//   y = a + b   (elementwise, per-tile) — Transformer residual connection
// =====================================================================
struct EltwiseAddOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_a;
    std::shared_ptr<tt::foil::Buffer> l1_b;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
EltwiseAddOp make_eltwise_add(tt::foil::Device& dev,
                              const TensorDesc& a, const TensorDesc& b,
                              TensorDesc& out,
                              tt::foil::CoreCoord core = {},
                              const std::string& kernel_dir = "");
void set_eltwise_add_args(tt::foil::Device& dev, EltwiseAddOp& op,
                          const TensorDesc& a, const TensorDesc& b,
                          const TensorDesc& out);
void execute(tt::foil::Device& dev, EltwiseAddOp& op);

// =====================================================================
// RMSNorm
//   y = x * 1/sqrt(mean(x²) + eps) * gamma
//
// Internally allocates the scaler (=1/H) and eps tiles in DRAM and
// populates them at make_rmsnorm time. The caller passes gamma as a
// TensorDesc with Wt tiles (one per hidden-dim tile column, replicated
// across all 32 token rows of the tile).
// =====================================================================
struct RmsNormOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_inp;
    std::shared_ptr<tt::foil::Buffer> l1_reduce;
    std::shared_ptr<tt::foil::Buffer> l1_gamma;
    std::shared_ptr<tt::foil::Buffer> l1_eps;
    std::shared_ptr<tt::foil::Buffer> l1_x2;
    std::shared_ptr<tt::foil::Buffer> l1_var;
    std::shared_ptr<tt::foil::Buffer> l1_recip_sqrt;
    std::shared_ptr<tt::foil::Buffer> l1_x_normed;
    std::shared_ptr<tt::foil::Buffer> l1_out;
    // DRAM-resident scaler / eps constants generated by make_rmsnorm.
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
    std::shared_ptr<tt::foil::Buffer> dram_eps;
};
RmsNormOp make_rmsnorm(tt::foil::Device& dev,
                       const TensorDesc& x, const TensorDesc& gamma,
                       TensorDesc& out,
                       uint32_t NCHt, uint32_t Wt,
                       float eps = 1e-5f,
                       tt::foil::CoreCoord core = {},
                       const std::string& kernel_dir = "");
// Reusable-handle setter — keeps the L1/CB layout fixed and rewrites only
// the DRAM source/dest addresses + NCHt/Wt. Same (NCHt, Wt) must be used
// across calls because the CB sizes were baked in at make-time. `eps`
// changes are honored only at make-time (dram_eps tile is fixed).
void set_rmsnorm_args(tt::foil::Device& dev, RmsNormOp& op,
                      const TensorDesc& x, const TensorDesc& gamma,
                      const TensorDesc& out,
                      uint32_t NCHt, uint32_t Wt);
void execute(tt::foil::Device& dev, RmsNormOp& op);

// =====================================================================
// Softmax (per-row)
//   y[r, c] = exp(x[r, c]) / Σ_c' exp(x[r, c'])
// =====================================================================
struct SoftmaxOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_inp;
    std::shared_ptr<tt::foil::Buffer> l1_reduce;
    std::shared_ptr<tt::foil::Buffer> l1_exp;
    std::shared_ptr<tt::foil::Buffer> l1_sum;
    std::shared_ptr<tt::foil::Buffer> l1_recip;
    std::shared_ptr<tt::foil::Buffer> l1_out;
    std::shared_ptr<tt::foil::Buffer> dram_scaler;  // = 1.0 tile, generated
};
SoftmaxOp make_softmax(tt::foil::Device& dev,
                       const TensorDesc& x, TensorDesc& out,
                       uint32_t NCHt, uint32_t Wt,
                       tt::foil::CoreCoord core = {},
                       const std::string& kernel_dir = "");
void execute(tt::foil::Device& dev, SoftmaxOp& op);

// =====================================================================
// MHA (with multiplicative mask)
//   out = softmax((Q · K^T) ⊙ mask) · V
//
// Caller responsibilities (still required at this layer):
//   - scale Q by 1/sqrt(D) before populating x_q.buf
//   - transpose K to KT layout before populating x_kt.buf
//   - provide a mask TensorDesc (St*St tiles, BF16 0/1)
// =====================================================================
struct MhaOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    // L1 CBs (11 of them) held internally — sized by St/Dt.
    std::vector<std::shared_ptr<tt::foil::Buffer>> l1_cbs;
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
};
// Optional per-buffer DRAM byte offsets — handy when Q / KT / V / out are
// "views" into bigger multi-head buffers and we want to launch MHA per head
// without copying data first. Set to 0 for the contiguous case.
struct MhaOffsets {
    uint64_t q_bytes  = 0;
    uint64_t kt_bytes = 0;
    uint64_t v_bytes  = 0;
    uint64_t out_bytes = 0;
};
MhaOp make_mha(tt::foil::Device& dev,
               const TensorDesc& q, const TensorDesc& kt,
               const TensorDesc& v, const TensorDesc& mask,
               TensorDesc& out,
               uint32_t St, uint32_t Dt,
               MhaOffsets offsets = {},
               tt::foil::CoreCoord core = {},
               const std::string& kernel_dir = "");
void execute(tt::foil::Device& dev, MhaOp& op);

// =====================================================================
// Embedding lookup (BRISC + NCRISC only, no compute)
//   out[r, :] = table[token_ids[r], :]
//
// Output is row-major BF16 (not tile-format); see ops/embedding/reader.cpp.
// =====================================================================
struct EmbeddingOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
// `out` is interpreted as a row-major [N, D] BF16 buffer of size N*D*2 bytes
// (i.e. out.num_tiles is not used here; out.buf->size_bytes must be N*D*2).
EmbeddingOp make_embedding(tt::foil::Device& dev,
                           const TensorDesc& table,
                           const std::vector<uint32_t>& token_ids,
                           uint32_t D,
                           TensorDesc& out,
                           tt::foil::CoreCoord core = {},
                           const std::string& kernel_dir = "");
// Update token_ids only — table, output, N and D must match the make-time
// values (CB blob + L1 alloc size are baked at make-time). token_ids.size()
// must equal the original N.
void set_embedding_args(tt::foil::Device& dev, EmbeddingOp& op,
                        const TensorDesc& table,
                        const std::vector<uint32_t>& token_ids,
                        uint32_t D,
                        const TensorDesc& out);
void execute(tt::foil::Device& dev, EmbeddingOp& op);

// =====================================================================
// ArgmaxRow0 (BRISC scan + NCRISC writer, no compute)
//   Given a [Mt=1, Vt] tile-format BF16 buffer, find the column index of
//   the maximum value in row 0 and write a 4-byte uint32_t to the output
//   DRAM buffer.
//
// Output (`out`) must be a 4-byte DRAM buffer (allocate one yourself or
// leave the shared_ptr null and the factory will allocate it).
// =====================================================================
struct ArgmaxRow0Op {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
// `row_in_tile` selects which row (0..31) of the [Mt=1, Vt] tile to scan.
// Default 0 keeps the original behaviour; prefill uses 31 to pick the
// last-position logits when the prediction point sits at the end of a tile.
ArgmaxRow0Op make_argmax_row0(tt::foil::Device& dev,
                              const TensorDesc& logits,
                              uint32_t Vt,
                              TensorDesc& out,
                              tt::foil::CoreCoord core = {},
                              const std::string& kernel_dir = "",
                              uint32_t row_in_tile = 0);
void set_argmax_row0_args(tt::foil::Device& dev, ArgmaxRow0Op& op,
                          const TensorDesc& logits, uint32_t Vt,
                          const TensorDesc& out,
                          uint32_t row_in_tile = 0);
void execute(tt::foil::Device& dev, ArgmaxRow0Op& op);

// =====================================================================
// KvAppend — device-side decode KV-cache append (BRISC only).
//   Reads the post-RoPE K row and the V row from T_Kr / T_V, writes
//   them into the per-layer K^T cache (slot1, col slot1_r) and V cache
//   (slot1, row slot1_r). Replaces qwen3_run's host-side
//   `dec:kv_slot1_rebuild` step.
// =====================================================================
struct KvAppendOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_scratch;
};
KvAppendOp make_kv_append(tt::foil::Device& dev,
                          const TensorDesc& kr, const TensorDesc& v,
                          const TensorDesc& kt_cache,
                          const TensorDesc& v_cache,
                          uint32_t slot1_r, uint32_t Nk, uint32_t StKv,
                          tt::foil::CoreCoord core = {},
                          const std::string& kernel_dir = "");
void set_kv_append_args(tt::foil::Device& dev, KvAppendOp& op,
                        const TensorDesc& kr, const TensorDesc& v,
                        const TensorDesc& kt_cache, const TensorDesc& v_cache,
                        uint32_t slot1_r, uint32_t Nk, uint32_t StKv);
void execute(tt::foil::Device& dev, KvAppendOp& op);

// =====================================================================
// KvSnapshot — device-side prefill KV-cache snapshot (BRISC only).
//   Replaces the host-side `pre:kv_cache_snapshot` step in qwen3vl_run:
//   reads block-major K^T (output of pre:transpose) and slot-major V
//   from L1-resident DRAM, writes both out in slot-major layout into
//   the per-layer K^T / V caches, and zero-fills the decode slots.
// =====================================================================
struct KvSnapshotOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_scratch;
};
KvSnapshotOp make_kv_snapshot(tt::foil::Device& dev,
                              const TensorDesc& kt_pre,
                              const TensorDesc& v_pre,
                              const TensorDesc& kt_cache,
                              const TensorDesc& v_cache,
                              uint32_t kSt, uint32_t kNkDt, uint32_t kStKv,
                              tt::foil::CoreCoord core = {},
                              const std::string& kernel_dir = "");
void set_kv_snapshot_args(tt::foil::Device& dev, KvSnapshotOp& op,
                          const TensorDesc& kt_pre, const TensorDesc& v_pre,
                          const TensorDesc& kt_cache, const TensorDesc& v_cache,
                          uint32_t kSt, uint32_t kNkDt, uint32_t kStKv);
void execute(tt::foil::Device& dev, KvSnapshotOp& op);

// =====================================================================
// RoPE (Rotary Position Embedding)
//   Applies RoPE in-place to a packed multi-head Q or K buffer.
//
//   x_out[st, h*Dt : (h+1)*Dt] = rope(x[st, h*Dt:(h+1)*Dt], cos[st], sin[st])
//
//   Rotation formula (split-half, matching Llama/Qwen3 style):
//     out_first  = x_first * cos - x_second * sin
//     out_second = x_second * cos + x_first * sin
//
//   cos/sin tables have shape [St, Dt_half] where Dt_half = Dt_per_head / 2.
//   The same cos/sin are shared across all heads (position-only encoding).
//
// Parameters:
//   x       : input  [St, num_heads * Dt_per_head] tiles  (modified in-place)
//   cos_sin : pair of DRAM tensors with Dt_half tiles each (St * Dt_half total)
//   out     : output [St, num_heads * Dt_per_head] — may alias x.buf for in-place
//   St      : sequence tile count
//   num_heads : number of attention heads to rotate
//   Dt_half : Dt_per_head / 2  (= head_dim / 64 for BF16 tile size)
// =====================================================================
struct RopeOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::vector<std::shared_ptr<tt::foil::Buffer>> l1_cbs;  // x0,x1,cos,sin,tmp0,tmp1,out
};
RopeOp make_rope(tt::foil::Device& dev,
                 const TensorDesc& x,
                 const TensorDesc& cos, const TensorDesc& sin,
                 TensorDesc& out,
                 uint32_t St, uint32_t num_heads, uint32_t Dt_half,
                 tt::foil::CoreCoord core = {},
                 const std::string& kernel_dir = "");
void set_rope_args(tt::foil::Device& dev, RopeOp& op,
                   const TensorDesc& x,
                   const TensorDesc& cos, const TensorDesc& sin,
                   const TensorDesc& out,
                   uint32_t St, uint32_t num_heads, uint32_t Dt_half);
void execute(tt::foil::Device& dev, RopeOp& op);

// =====================================================================
// Transpose2d
//   Transposes a [Rt, Ct] tile-format tensor to [Ct, Rt], with within-tile
//   WH transpose so the result is correct in tile-format too. Used to turn
//   K (post-RoPE, layout [St, num_kv*Dt]) into K^T (layout [num_kv*Dt, St])
//   so the fused per-head MHA loop can read it without per-head host trips.
// =====================================================================
struct Transpose2dOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::shared_ptr<tt::foil::Buffer> l1_in;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};
Transpose2dOp make_transpose_2d(tt::foil::Device& dev,
                                const TensorDesc& in, TensorDesc& out,
                                uint32_t Rt, uint32_t Ct,
                                tt::foil::CoreCoord core = {},
                                const std::string& kernel_dir = "");
void set_transpose_2d_args(tt::foil::Device& dev, Transpose2dOp& op,
                           const TensorDesc& in, const TensorDesc& out,
                           uint32_t Rt, uint32_t Ct);
void execute(tt::foil::Device& dev, Transpose2dOp& op);

// =====================================================================
// GqaFused
//   Single-kernel multi-head GQA causal attention. Q, KT (= K^T from
//   transpose_2d), V are the full multi-head tensors:
//     Q   shape [St, num_q  * Dt]
//     KT  shape [num_kv*Dt, St]
//     V   shape [St, num_kv * Dt]
//   The kernel loops over the num_q query heads internally, with
//   kv_head = q_head / (num_q / num_kv). One launch replaces the
//   num_q × (per-head MHA) host loop.
// =====================================================================
struct GqaFusedOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::vector<std::shared_ptr<tt::foil::Buffer>> l1_cbs;
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
};
GqaFusedOp make_gqa_fused(tt::foil::Device& dev,
                          const TensorDesc& q, const TensorDesc& kt,
                          const TensorDesc& v, const TensorDesc& mask,
                          TensorDesc& out,
                          uint32_t St, uint32_t Dt,
                          uint32_t num_q, uint32_t num_kv,
                          tt::foil::CoreCoord core = {},
                          const std::string& kernel_dir = "");
void set_gqa_fused_args(tt::foil::Device& dev, GqaFusedOp& op,
                        const TensorDesc& q, const TensorDesc& kt,
                        const TensorDesc& v, const TensorDesc& mask,
                        const TensorDesc& out,
                        uint32_t St, uint32_t Dt,
                        uint32_t num_q, uint32_t num_kv);
void execute(tt::foil::Device& dev, GqaFusedOp& op);

// =====================================================================
// GqaDecode
//   Multi-head GQA attention with decoupled Q and K/V sequence dimensions:
//     Q   [St_q, num_q  * Dt]
//     KT  [num_kv * Dt, St_kv]
//     V   [St_kv, num_kv * Dt]
//     mask[St_q, St_kv]                    — 0/1 BF16, masks padding K rows
//     out [St_q, num_q  * Dt]
//   Typical decode: St_q = 1 (a single query padded into a 32-row tile),
//   St_kv = ceil((prev_seq + 1) / 32).
// =====================================================================
struct GqaDecodeOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    std::vector<std::shared_ptr<tt::foil::Buffer>> l1_cbs;
    std::shared_ptr<tt::foil::Buffer> dram_scaler;
};
GqaDecodeOp make_gqa_decode(tt::foil::Device& dev,
                            const TensorDesc& q, const TensorDesc& kt,
                            const TensorDesc& v, const TensorDesc& mask,
                            TensorDesc& out,
                            uint32_t St_q, uint32_t St_kv, uint32_t Dt,
                            uint32_t num_q, uint32_t num_kv,
                            tt::foil::CoreCoord core = {},
                            const std::string& kernel_dir = "");
void set_gqa_decode_args(tt::foil::Device& dev, GqaDecodeOp& op,
                         const TensorDesc& q, const TensorDesc& kt,
                         const TensorDesc& v, const TensorDesc& mask,
                         const TensorDesc& out,
                         uint32_t St_q, uint32_t St_kv, uint32_t Dt,
                         uint32_t num_q, uint32_t num_kv);
void execute(tt::foil::Device& dev, GqaDecodeOp& op);

}  // namespace tt::foil::op_lib
