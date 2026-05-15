// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// v3-5: end-to-end NOC inter-core passthrough.
//
// Producer on (0,0) computes sum = a + b → writes sum + flag to (0,1)'s L1
// via noc_async_write. Consumer on (0,1) busy-loops on the flag, then
// computes result = sum * 2 in its local L1. Host reads result and verifies.
//
//   a = 7, b = 6 → sum = 13 → result = 26
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/noc_passthrough/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_noc_passthrough

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_foil/runtime.hpp"

// (Only the public header is needed — make_noc_unicast_addr is exposed
// there now, no need to reach into device internals from a test.)

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
    std::puts("test_noc_passthrough: 2 cores booted");

    tt::foil::CoreCoord cp{0, 0};  // producer
    tt::foil::CoreCoord cc{0, 1};  // consumer

    // -- Producer-side L1 slots: inputs + own scratch for sum/flag --
    auto a_buf       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cp);
    auto b_buf       = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cp);
    auto sum_scratch = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cp);
    auto flag_scratch= tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cp);

    // -- Consumer-side L1 slots: producer writes sum/flag here, kernel writes result --
    auto sum_dst     = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cc);
    auto flag_dst    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cc);
    auto result_buf  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), cc);

    uint32_t a = 7, b = 6, zero = 0;
    tt::foil::write_buffer(*dev, *a_buf, &a, sizeof(a));
    tt::foil::write_buffer(*dev, *b_buf, &b, sizeof(b));
    tt::foil::write_buffer(*dev, *sum_scratch,  &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *flag_scratch, &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *sum_dst,      &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *flag_dst,     &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *result_buf,   &zero, sizeof(zero));

    // -- NOC dst addresses (public helper handles logical→translated lookup) --
    uint64_t sum_noc  = tt::foil::make_noc_unicast_addr(*dev, cc, sum_dst->device_addr);
    uint64_t flag_noc = tt::foil::make_noc_unicast_addr(*dev, cc, flag_dst->device_addr);
    std::printf("test_noc_passthrough: sum  NOC dst = 0x%016lx\n", sum_noc);
    std::printf("test_noc_passthrough: flag NOC dst = 0x%016lx\n", flag_noc);

    // -- Load kernels --
    std::array<tt::foil::RiscBinary, 1> p_bin = {{
        {tt::foil::RiscBinary::RiscId::BRISC, kernel_dir + "/producer.elf"}
    }};
    std::array<tt::foil::RiscBinary, 1> c_bin = {{
        {tt::foil::RiscBinary::RiscId::BRISC, kernel_dir + "/consumer.elf"}
    }};
    auto kp = tt::foil::load_kernel(*dev, p_bin, cp);
    auto kc = tt::foil::load_kernel(*dev, c_bin, cc);

    std::vector<uint32_t> p_args = {
        static_cast<uint32_t>(a_buf->device_addr),
        static_cast<uint32_t>(b_buf->device_addr),
        static_cast<uint32_t>(sum_scratch->device_addr),
        static_cast<uint32_t>(sum_noc  & 0xFFFFFFFFu),
        static_cast<uint32_t>(sum_noc  >> 32),
        static_cast<uint32_t>(flag_scratch->device_addr),
        static_cast<uint32_t>(flag_noc & 0xFFFFFFFFu),
        static_cast<uint32_t>(flag_noc >> 32),
    };
    std::vector<uint32_t> c_args = {
        static_cast<uint32_t>(flag_dst->device_addr),
        static_cast<uint32_t>(sum_dst->device_addr),
        static_cast<uint32_t>(result_buf->device_addr),
    };
    tt::foil::set_runtime_args(*dev, *kp, tt::foil::RiscBinary::RiscId::BRISC, p_args);
    tt::foil::set_runtime_args(*dev, *kc, tt::foil::RiscBinary::RiscId::BRISC, c_args);

    std::puts("test_noc_passthrough: launching producer + consumer");
    tt::foil::execute(*dev, {kp.get(), kc.get()});
    std::puts("test_noc_passthrough: execute returned");

    uint32_t result = 0;
    tt::foil::read_buffer(*dev, *result_buf, &result, sizeof(result));
    const uint32_t expected = (a + b) * 2u;
    std::printf("test_noc_passthrough: result = %u (expected %u)\n", result, expected);

    bool ok = (result == expected);
    tt::foil::close_device(std::move(dev));
    if (!ok) { std::puts("test_noc_passthrough: FAIL"); return 1; }
    std::puts("test_noc_passthrough: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_noc_passthrough: FAIL — %s\n", e.what());
    return 1;
}
