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
// Root cause of the long debug saga (resolved in v4-6f): the firmware
// ELFs under tt_metal/pre-compiled/<hash>/ on this system are STALE
// relative to what tt-metal's runtime actually loads. tt-metal at
// runtime builds firmware into ~/.cache/tt-metal-cache/<hash>/firmware/
// with a different build hash, and the two brisc.elf files differ in
// bytes — the pre-compiled one has setup_local_cb_read_write_interfaces
// writes that never become visible to the kernel that runs immediately
// after on the same BRISC. Pointing tt-foil's firmware_paths at the
// JIT cache (and rebuilding kernels against the matching
// brisc_weakened.elf) makes the whole pipeline work. The dispatch
// alignment with tt-metal (send_reset_go_signal, l1_membar) from the
// earlier rounds was good hygiene but wasn't the trigger.
//
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
