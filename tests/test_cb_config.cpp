// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v4-3: validate the CB config blob layout written by register_cbs().
//
// Loads a no-op BRISC kernel just to get a Kernel handle + an allocator
// state, registers two CBs (one at c_0, one at c_16), then reads the
// resulting blob back from L1 via UMD and checks that:
//   - blob_l1_addr is inside the KERNEL_CONFIG region
//   - the 17 descriptor slots (c_0..c_16) are zero except for the two we
//     populated
//   - kernel.cb_alloc records the right mask + min_index + offset
//
// No firmware is launched — the kernel is loaded purely so register_cbs
// has somewhere to allocate from. This test would still pass even if CB
// firmware support were broken.
//
// Usage: TT_FOIL_KERNEL_DIR=examples/add_two_numbers/prebuilt \
//        TT_FOIL_DEVICE=3 ./test_cb_config

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
#include "device.hpp"
#include "kernel.hpp"
#include "llrt/hal.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

static std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}});
    tt::foil::CoreCoord core{0, 0};

    std::array<tt::foil::RiscBinary, 1> bins = {{
        {tt::foil::RiscBinary::RiscId::BRISC, kernel_dir + "/add_brisc.elf"}
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // Register two CBs: c_0 (input, 1 page of 1 KB) and c_16 (output, 1 page).
    std::array<tt::foil::CbConfig, 2> cbs = {{
        {/*idx*/ 0,  /*fifo_addr*/ 0x10000, /*size*/ 1024, /*pages*/ 1, /*page_size*/ 1024},
        {/*idx*/ 16, /*fifo_addr*/ 0x10400, /*size*/ 1024, /*pages*/ 1, /*page_size*/ 1024},
    }};
    tt::foil::register_cbs(*dev, *kernel, cbs);

    const auto& a = kernel->cb_alloc;
    std::printf("test_cb_config: blob L1=0x%lx offset=0x%x mask=0x%x min_idx=%u\n",
                a.blob_l1_addr, a.local_cb_offset, a.local_cb_mask, a.min_local_cb_start_index);

    bool ok = a.valid;
    if (!ok) std::fprintf(stderr, "  cb_alloc.valid == false\n");
    if (a.min_local_cb_start_index != 0)   { std::fprintf(stderr, "  min_idx=%u expected 0\n",   a.min_local_cb_start_index); ok = false; }
    if (a.local_cb_mask != ((1u << 0) | (1u << 16))) {
        std::fprintf(stderr, "  mask=0x%x expected 0x10001\n", a.local_cb_mask); ok = false;
    }

    // Read back the blob — 17 descriptors × 16 B = 272 B.
    const std::size_t blob_bytes = 17u * 4u * sizeof(uint32_t);
    std::vector<uint32_t> rb(blob_bytes / sizeof(uint32_t), 0);

    tt::umd::CoreCoord cc{
        kernel->virt_x, kernel->virt_y,
        tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
    dev->umd_driver->read_from_device(rb.data(), dev->chip_id, cc, a.blob_l1_addr, blob_bytes);

    auto check_slot = [&](const char* name, uint32_t cb_idx, uint32_t exp_addr,
                          uint32_t exp_size, uint32_t exp_pages, uint32_t exp_page_size) {
        const uint32_t* s = rb.data() + cb_idx * 4;
        std::printf("  %s: [%u %u %u %u]\n", name, s[0], s[1], s[2], s[3]);
        if (s[0] != exp_addr || s[1] != exp_size || s[2] != exp_pages || s[3] != exp_page_size) {
            std::fprintf(stderr, "    expected [%u %u %u %u]\n",
                exp_addr, exp_size, exp_pages, exp_page_size);
            ok = false;
        }
    };
    check_slot("c_0",  0,  0x10000, 1024, 1, 1024);
    check_slot("c_16", 16, 0x10400, 1024, 1, 1024);

    // Gaps must be zero.
    for (uint32_t i = 1; i < 16; ++i) {
        const uint32_t* s = rb.data() + i * 4;
        if (s[0] || s[1] || s[2] || s[3]) {
            std::fprintf(stderr, "  c_%u nonzero gap: [%u %u %u %u]\n", i, s[0], s[1], s[2], s[3]);
            ok = false;
        }
    }
    if (ok) std::puts("  gap entries c_1..c_15 all zero ✓");

    tt::foil::close_device(std::move(dev));
    if (!ok) { std::puts("test_cb_config: FAIL"); return 1; }
    std::puts("test_cb_config: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_cb_config: FAIL — %s\n", e.what());
    return 1;
}
