// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Integration test: add two numbers via a pre-compiled BRISC kernel.
// Requires a real Blackhole chip and pre-built firmware + kernel binaries.
//
// Usage:
//   TT_FOIL_KERNEL_DIR=/path/to/prebuilt ./test_add_kernel

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

#include "tt_foil/runtime.hpp"

static std::string required_env(const char* name) {
    const char* val = std::getenv(name);
    if (!val) {
        throw std::runtime_error(std::string("Missing env var: ") + name);
    }
    return val;
}

int main() try {
    const std::string kernel_dir = required_env("TT_FOIL_KERNEL_DIR");

    // ---- Open device (firmware init delegated to tt-metal CreateDevice) ----
    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;
    auto dev = tt::foil::open_device(pcie_index);
    std::puts("test_add_kernel: device opened");

    // ---- Target core: (0, 0) ----
    tt::foil::CoreCoord core{0, 0};

    // ---- Allocate L1 buffers for A, B, result ----
    auto buf_a = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);
    auto buf_b = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);
    auto buf_r = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);

    // ---- Write inputs ----
    uint32_t a = 42, b = 58;
    tt::foil::write_buffer(*dev, *buf_a, &a, sizeof(a));
    tt::foil::write_buffer(*dev, *buf_b, &b, sizeof(b));
    uint32_t zero = 0;
    tt::foil::write_buffer(*dev, *buf_r, &zero, sizeof(zero));
    std::puts("test_add_kernel: buffers written (42 + 58 = ?)");

    // ---- Load kernel ----
    std::string brisc_elf = kernel_dir + "/add_brisc.elf";
    std::array<tt::foil::RiscBinary, 1> binaries = {{
        {tt::foil::RiscBinary::RiscId::BRISC, brisc_elf},
    }};
    auto kernel = tt::foil::load_kernel(*dev, binaries, core);
    std::puts("test_add_kernel: kernel loaded");

    // ---- Set runtime args: addresses of A, B, result ----
    // The kernel reads: arg[0] = addr(A), arg[1] = addr(B), arg[2] = addr(result)
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        static_cast<uint32_t>(buf_r->device_addr),
    };
    tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::BRISC, args);

    // ---- Execute ----
    tt::foil::execute(*dev, *kernel);
    std::puts("test_add_kernel: execute returned");

    // ---- Read result ----
    uint32_t result = 0;
    tt::foil::read_buffer(*dev, *buf_r, &result, sizeof(result));

    std::printf("test_add_kernel: result = %u (expected 100)\n", result);
    assert(result == 100 && "kernel produced wrong result");

    // ---- Cleanup ----
    tt::foil::close_device(std::move(dev));
    std::puts("test_add_kernel: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_add_kernel: FAIL — %s\n", e.what());
    return 1;
}
