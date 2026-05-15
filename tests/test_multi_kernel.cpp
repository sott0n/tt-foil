// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v3-2: launch two BRISC kernels (one per Tensix core) with one execute
// call. Reuses examples/add_two_numbers/prebuilt/add_brisc.elf — same ELF on
// both cores, but each core sees a distinct (a, b, result) triple via per-
// kernel runtime args. After execute() returns both result slots must hold
// their respective sums.
//
// This exercises dispatch_execute_multi: every kernel's GO is fired before
// any DONE check, so a hang on either core would fail the test fast.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/add_two_numbers/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_multi_kernel

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tt_foil/runtime.hpp"

static std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) throw std::runtime_error(std::string("Missing env var: ") + name);
    return val;
}

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;

    auto dev = tt::foil::open_device(pcie_index, "", {{0, 0}, {0, 1}});
    std::puts("test_multi_kernel: 2 cores booted");

    tt::foil::CoreCoord c0{0, 0}, c1{0, 1};

    auto a0 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c0);
    auto b0 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c0);
    auto r0 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c0);
    auto a1 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c1);
    auto b1 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c1);
    auto r1 = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), c1);

    uint32_t a0_v = 7,  b0_v = 6,  zero = 0;
    uint32_t a1_v = 11, b1_v = 31;
    tt::foil::write_buffer(*dev, *a0, &a0_v, sizeof(a0_v));
    tt::foil::write_buffer(*dev, *b0, &b0_v, sizeof(b0_v));
    tt::foil::write_buffer(*dev, *r0, &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *a1, &a1_v, sizeof(a1_v));
    tt::foil::write_buffer(*dev, *b1, &b1_v, sizeof(b1_v));
    tt::foil::write_buffer(*dev, *r1, &zero, sizeof(zero));

    std::string brisc_elf = kernel_dir + "/add_brisc.elf";
    std::array<tt::foil::RiscBinary, 1> binaries = {{
        {tt::foil::RiscBinary::RiscId::BRISC, brisc_elf}
    }};
    auto k0 = tt::foil::load_kernel(*dev, binaries, c0);
    auto k1 = tt::foil::load_kernel(*dev, binaries, c1);

    std::vector<uint32_t> args0 = {
        static_cast<uint32_t>(a0->device_addr),
        static_cast<uint32_t>(b0->device_addr),
        static_cast<uint32_t>(r0->device_addr),
    };
    std::vector<uint32_t> args1 = {
        static_cast<uint32_t>(a1->device_addr),
        static_cast<uint32_t>(b1->device_addr),
        static_cast<uint32_t>(r1->device_addr),
    };
    tt::foil::set_runtime_args(*dev, *k0, tt::foil::RiscBinary::RiscId::BRISC, args0);
    tt::foil::set_runtime_args(*dev, *k1, tt::foil::RiscBinary::RiscId::BRISC, args1);

    std::puts("test_multi_kernel: launching both kernels");
    tt::foil::execute(*dev, {k0.get(), k1.get()});
    std::puts("test_multi_kernel: both kernels done");

    uint32_t got0 = 0, got1 = 0;
    tt::foil::read_buffer(*dev, *r0, &got0, sizeof(got0));
    tt::foil::read_buffer(*dev, *r1, &got1, sizeof(got1));

    const uint32_t exp0 = a0_v + b0_v;
    const uint32_t exp1 = a1_v + b1_v;
    std::printf("test_multi_kernel: (0,0) result=%u (expected %u)\n", got0, exp0);
    std::printf("test_multi_kernel: (0,1) result=%u (expected %u)\n", got1, exp1);

    bool ok = (got0 == exp0) && (got1 == exp1);
    tt::foil::close_device(std::move(dev));
    if (!ok) { std::puts("test_multi_kernel: FAIL"); return 1; }
    std::puts("test_multi_kernel: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_multi_kernel: FAIL — %s\n", e.what());
    return 1;
}
