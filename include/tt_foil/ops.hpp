// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Primitive Op Library — host-side abstraction over tt-foil kernels.
//
// First cut (`silu` only). Goals:
//   - hide load_kernel / register_cbs / set_runtime_args boilerplate
//   - let callers compose ops by passing TensorDesc handles instead of
//     juggling raw Buffer + L1 CB layouts
//   - keep the surface tiny so the right API shape can settle before
//     building the full Qwen3 op set
//
// Conventions for this cut:
//   - tensors are stored in DRAM as tile-formatted BF16 streams
//   - tensor logical layout is described by `num_tiles` (linear stream);
//     higher-rank shapes will be layered on top later
//   - L1 CBs are implementation details owned by the OpHandle

#pragma once

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
// Equivalent to `allocate_buffer(dev, DRAM, num_tiles * 2048)` wrapped in
// a TensorDesc with empty shape.
TensorDesc allocate_tensor_dram(tt::foil::Device& dev, uint32_t num_tiles);

// ---------------------------------------------------------------------------
// SiLU op
// ---------------------------------------------------------------------------

// Opaque handle: bundles the loaded kernel + the L1 CBs + RTAs that were
// set up for one (x, out) pair. Re-fire via `execute` without reconfiguring.
struct SiluOp {
    std::shared_ptr<tt::foil::Kernel> kernel;
    // L1 CB buffers held internally so the op outlives a single execute.
    std::shared_ptr<tt::foil::Buffer> l1_in;
    std::shared_ptr<tt::foil::Buffer> l1_out;
};

// Build a SiLU op: y = silu(x), with x and out same num_tiles, both DRAM.
//   kernel_dir : path to the prebuilt ELFs. If empty, falls back to
//                "$TT_FOIL_OPS_DIR/eltwise_unary/prebuilt", then to
//                "ops/eltwise_unary/prebuilt" relative to CWD.
SiluOp make_silu(tt::foil::Device& dev,
                 const TensorDesc& x,
                 TensorDesc& out,
                 tt::foil::CoreCoord core = {},
                 const std::string& kernel_dir = "");

// Blocking execute.
void execute(tt::foil::Device& dev, SiluOp& op);

}  // namespace tt::foil::op_lib
