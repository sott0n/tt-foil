// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v4-6: end-to-end tile copy through all 5 Tensix RISCs.
//
// Pipeline:
//   BRISC  (reader)  : L1 src_buf → CB c_0
//   TRISC0 (UNPACK)  ┐
//   TRISC1 (MATH)    │ copy_tile through dst regs
//   TRISC2 (PACK)    ┘ pack 1 tile from regs → CB c_16
//   NCRISC (writer)  : CB c_16 → L1 dst_buf
//
// One bf16 32×32 tile (2 KB). Host fills src with a deterministic
// pattern, runs the kernel, reads dst back, asserts dst == src.
//
// CURRENT STATUS (v4-6): the test hangs in BRISC's cb_reserve_back.
// Diagnosis (verified by instrumented kernels + L1 read-back):
//   1. Host-side state is correct:
//        - launch_msg.kernel_config_base[0] = 0x9e00 (TENSIX MEM_MAP_END)
//        - launch_msg.local_cb_offset = 0x30b0
//        - launch_msg.local_cb_mask = 0x10001 (c_0 + c_16)
//        - launch_msg.enables = 0x1f (all 5 RISCs)
//        - launch_msg.min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS
//        - launch_msg.kernel_text_offset[0..4] all populated
//        - CB descriptor blob at 0xceb0 = [fifo=0x1c200, size=2048,
//          pages=1, page_size=2048] for c_0, gap zeros 1..15, then c_16
//   2. BRISC kernel runs (reaches kernel_main, dbg sentinel written).
//   3. BUT cb_interface[0..16] in BRISC's local data memory (0xffb0046c+)
//      are entirely zero — firmware's setup_local_cb_read_write_interfaces
//      did NOT execute its writes, even though brisc.elf disassembly
//      contains the inline-asm setup loop at 0x3f70 gated only by
//      `enables & 1`.
//   4. Manually populating cb_interface[0] from BRISC kernel-side (in
//      reader.cpp) makes cb_reserve_back / cb_push_back work, confirming
//      the rest of the BRISC pipeline is fine.
//
// Root cause still unknown. Most likely candidates:
//   - A subtle cold-boot prerequisite tt-foil is missing (e.g., a specific
//     mailbox/scratch init that tt-metal does after BRISC firmware deassert
//     but before launch_msg dispatch).
//   - Firmware reads `enables` from a different launch_msg slot than the
//     one we wrote to. (We always write launch[0]; mailboxes->launch_msg_rd_ptr
//     is initialized to 0 by BRISC firmware itself — should match.)
//
// Until this is resolved, the test is gated under TT_FOIL_HW_TESTS but
// will time out. The kernel ELFs and CB plumbing are still produced and
// verified by test_cb_config.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/tile_copy/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_tile_copy

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

#include "cb_config.hpp"

static std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    constexpr uint32_t kTileBytes = 32 * 32 * 2;   // bf16 32×32
    constexpr uint32_t kTileWords = kTileBytes / sizeof(uint16_t);

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    // ---- L1 buffers: src tile, dst tile, CB rings ----
    auto buf_src    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_dst    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_in  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    // ---- Fill src with a deterministic bf16 pattern ----
    std::vector<uint16_t> src(kTileWords);
    for (uint32_t i = 0; i < kTileWords; ++i) {
        src[i] = static_cast<uint16_t>(i ^ 0xA5A5u);
    }
    tt::foil::write_buffer(*dev, *buf_src, src.data(), kTileBytes);

    std::vector<uint16_t> zero(kTileWords, 0);
    tt::foil::write_buffer(*dev, *buf_dst, zero.data(), kTileBytes);

    // ---- Load 5-RISC kernel ----
    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // ---- Register CBs: c_0 (input) + c_16 (output), one tile each ----
    std::array<tt::foil::CbConfig, 2> cbs = {{
        {/*idx*/ 0,
         /*fifo_addr*/ buf_cb_in->device_addr,
         /*size*/      kTileBytes,
         /*pages*/     1,
         /*page_size*/ kTileBytes},
        {/*idx*/ 16,
         /*fifo_addr*/ buf_cb_out->device_addr,
         /*size*/      kTileBytes,
         /*pages*/     1,
         /*page_size*/ kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    // ---- Runtime args: reader gets src, writer gets dst ----
    std::array<uint32_t, 1> ra_brisc  = {static_cast<uint32_t>(buf_src->device_addr)};
    std::array<uint32_t, 1> ra_ncrisc = {static_cast<uint32_t>(buf_dst->device_addr)};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    // ---- Fire ----
    tt::foil::execute(*dev, *kernel);

    // ---- Read back and compare ----
    std::vector<uint16_t> got(kTileWords, 0);
    tt::foil::read_buffer(*dev, *buf_dst, got.data(), kTileBytes);

    uint32_t mismatches = 0;
    uint32_t first_bad  = kTileWords;
    for (uint32_t i = 0; i < kTileWords; ++i) {
        if (got[i] != src[i]) {
            if (first_bad == kTileWords) first_bad = i;
            ++mismatches;
        }
    }

    if (mismatches != 0) {
        std::fprintf(stderr,
            "test_tile_copy: %u/%u mismatches; first at word %u: got=0x%04x expected=0x%04x\n",
            mismatches, kTileWords, first_bad, got[first_bad], src[first_bad]);
        tt::foil::close_device(std::move(dev));
        std::puts("test_tile_copy: FAIL");
        return 1;
    }

    tt::foil::close_device(std::move(dev));
    std::puts("test_tile_copy: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_tile_copy: FAIL — %s\n", e.what());
    return 1;
}
