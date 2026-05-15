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
// CURRENT STATUS (v4-6c): the test hangs in BRISC's cb_reserve_back.
// Diagnosis collected over three rounds of on-device instrumentation:
//
//   Host-side state (verified via L1 read-back):
//     launch_msg.kernel_config_base[0] = 0x9e00 (TENSIX MEM_MAP_END) ✓
//     launch_msg.local_cb_offset       = 0x30b0 ✓
//     launch_msg.local_cb_mask         = 0x10001 (c_0 + c_16) ✓
//     launch_msg.enables               = 0x1f (all 5 RISCs) ✓
//     launch_msg.min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS ✓
//     launch_msg.kernel_text_offset[0..4]  all populated ✓
//     CB blob at L1 0xceb0 = c_0: [fifo=0x1c200, size=2048, pages=1,
//       page_size=2048]; gap zeros 1..15; c_16: [fifo=0x1ca00, ...]  ✓
//     LAUNCH_MSG_BUFFER_RD_PTR (0x6c)  = 0 ✓
//     GO_MSG_INDEX                     = 0 ✓
//     go_messages[0].signal            = 0x80 (RUN_MSG_GO) — firmware
//                                              saw the GO we sent
//     subordinate_sync                 = 0x80808080 — firmware sent
//                                              RUN_SYNC_MSG_GO to ncrisc
//                                              + all 3 triscs, which
//                                              only happens AFTER it
//                                              passes run_triscs() and
//                                              start_ncrisc_kernel_run_early()
//                                              in brisc.cc main loop
//                                              (~10 lines before the CB
//                                              setup call)
//
//   Device-side state (BRISC kernel-side raw inline-asm `lw` at
//   0xffb0046c and 0xffb0066c):
//     cb_interface[0]  = {0,0,0,0,0,0,0,0}  — all zero
//     cb_interface[16] = {0,0,0,0}          — all zero
//     Both kernel-entry and kernel-exit sentinels written  — so the
//     kernel ran end-to-end, and the inline-asm read reached the
//     instructions; firmware just never wrote to those bytes.
//
//   Workaround verified to expose the rest of the pipeline:
//     Manually populating get_local_cb_interface(0).{fifo_*} from the
//     BRISC kernel makes cb_reserve_back / cb_push_back work.
//
// Synthesis: BRISC firmware reaches the kernel-launch if-block (since
// subordinate_sync = 0x80808080 requires run_triscs + early_ncrisc,
// both of which are inside it) AND runs our BRISC kernel (since
// sentinels land in L1). But somehow `setup_local_cb_read_write_interfaces`
// — which sits between those two waypoints in brisc.cc — does not
// produce visible writes to cb_interface[].
//
// Disassembly verification (v4-6d, this round):
//   In the pre-built brisc.elf, setup_local_cb_read_write_interfaces is
//   present and inlined at 0x3f70. The store-side address calculation is:
//       3f70: lui  a3, 0xffb00
//       3f80: addi a0, a3, 1132    # = 0xffb0046c = &cb_interface[0]
//   And the inner loop body at 0x3f88 emits exactly the seven stores
//   matching LocalCBInterface offsets:
//       sw zero, 24(a0)   tiles_acked_received_init
//       sw a3,    0(a0)   fifo_size
//       sw a3,    4(a0)   fifo_limit   (= fifo_addr + fifo_size)
//       sw a2,   20(a0)   fifo_wr_ptr  (= fifo_addr)
//       sw a6,   12(a0)   fifo_num_pages
//       sw a2,   16(a0)   fifo_rd_ptr  (= fifo_addr)
//       sw a7,    8(a0)   fifo_page_size
//   So the binary targets the exact same 0xffb0046c that our kernel
//   reads from with inline-asm `lw 0(t0)`. The values to write (loaded
//   from the L1 blob via t3) and the mask shifting are correct.
//
//   Branch leading to the setup is at 0x3f64:
//       beqz a1, 4438       # a1 = enables & 1
//   and 0x4438 is the else-branch (setup_remote only, no BRISC kernel
//   jump). Since our BRISC kernel does run, the branch isn't taken, so
//   setup is reached. The post-setup path (upper-half = no-op for
//   mask_upper=0, setup_remote = no-op for end_cb_index=64, no barrier,
//   start_ncrisc_kernel_run = no-op on Blackhole, kernel jump at 0x40a0)
//   leaves cb_interface[] untouched.
//
//   Conclusion: at the binary level, the firmware should write
//   cb_interface[0] and cb_interface[16]. But on the device the writes
//   are not visible to the BRISC kernel that runs ~hundreds of
//   instructions later in the same core. We've ruled out:
//     - Wrong store offsets (match LocalCBInterface)
//     - Wrong store address (match cb_interface symbol)
//     - Setup being skipped (kernel jump implies setup ran)
//     - Kernel-side clobber (do_crt1 only touches [0xffb00c70..0xffb00c88))
//
// Hypotheses still on the table:
//   - The pre-built brisc.elf comes from a tt-metal revision that
//     diverges from the source on this branch — some later code on the
//     same kernel-launch path performs an additional store loop that
//     re-zeros cb_interface[]. (Worth grepping the brisc.elf binary
//     for stores into 0xffb004... ranges past the setup loop.)
//   - tt-metal's CreateDevice runs a step we don't (some additional
//     mailbox or scratch init that primes the firmware's BSS or stops
//     it from re-zeroing it).
//   - BRISC's local data memory has some posted-write / coherency
//     quirk on tt-bh-tensix that drops these specific writes. Unlikely
//     but worth ruling out by writing the same values from BRISC
//     kernel-side (we confirmed kernel-side writes survive; firmware
//     writes don't).
//
// Until this is resolved the test is gated under TT_FOIL_HW_TESTS and
// will time out at first launch.
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

    auto buf_src    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_dst    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_in  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);
    auto buf_cb_out = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, kTileBytes, core);

    std::vector<uint16_t> src(kTileWords);
    for (uint32_t i = 0; i < kTileWords; ++i) {
        src[i] = static_cast<uint16_t>(i ^ 0xA5A5u);
    }
    tt::foil::write_buffer(*dev, *buf_src, src.data(), kTileBytes);

    std::vector<uint16_t> zero(kTileWords, 0);
    tt::foil::write_buffer(*dev, *buf_dst, zero.data(), kTileBytes);

    using R = tt::foil::RiscBinary;
    std::array<R, 5> bins = {{
        {R::RiscId::BRISC,  kernel_dir + "/reader.brisc.elf"},
        {R::RiscId::NCRISC, kernel_dir + "/writer.ncrisc.elf"},
        {R::RiscId::TRISC0, kernel_dir + "/compute.trisc0.elf"},
        {R::RiscId::TRISC1, kernel_dir + "/compute.trisc1.elf"},
        {R::RiscId::TRISC2, kernel_dir + "/compute.trisc2.elf"},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    std::array<tt::foil::CbConfig, 2> cbs = {{
        {0,  buf_cb_in->device_addr,  kTileBytes, 1, kTileBytes},
        {16, buf_cb_out->device_addr, kTileBytes, 1, kTileBytes},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    std::array<uint32_t, 1> ra_brisc  = {static_cast<uint32_t>(buf_src->device_addr)};
    std::array<uint32_t, 1> ra_ncrisc = {static_cast<uint32_t>(buf_dst->device_addr)};
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::BRISC,  ra_brisc);
    tt::foil::set_runtime_args(*dev, *kernel, R::RiscId::NCRISC, ra_ncrisc);

    tt::foil::execute(*dev, *kernel);

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
