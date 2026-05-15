// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the tile_copy example.
//
// v4-2 stage: this is a *blank* compute kernel — just an empty
// kernel_main(). The build pipeline compiles this source 3 times with
// the right TRISC + UCK_CHLKC defines and produces compute.trisc0.elf,
// compute.trisc1.elf, compute.trisc2.elf. Loading and running them
// across the UNPACK/MATH/PACK pipeline lands in v4-3..v4-5 (Circular
// Buffer setup + dispatch wiring).
//
// When this kernel grows beyond blank, it'll include
//   #include "compute_kernel_api/common.h"
//   #include "compute_kernel_api/eltwise_unary/copy_tile.h"
// and use the cb_wait_front / copy_tile / pack_tile / cb_push_back
// idiom inside MAIN.

#include <cstdint>

void kernel_main() {
    // nothing — see header comment
}
