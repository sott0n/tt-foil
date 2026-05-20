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
MatMulOp make_matmul(tt::foil::Device& dev,
                     const TensorDesc& a, const TensorDesc& b,
                     TensorDesc& out,
                     uint32_t Mt, uint32_t Kt, uint32_t Nt,
                     tt::foil::CoreCoord core = {},
                     const std::string& kernel_dir = "");
void execute(tt::foil::Device& dev, MatMulOp& op);

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
void execute(tt::foil::Device& dev, EltwiseMulOp& op);

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
void execute(tt::foil::Device& dev, EmbeddingOp& op);

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
void execute(tt::foil::Device& dev, Transpose2dOp& op);

}  // namespace tt::foil::op_lib
