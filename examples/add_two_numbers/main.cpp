// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Example: add two numbers on Blackhole using a pre-compiled BRISC kernel.
//
// Prerequisites:
//   1. Pre-compiled kernel at $TT_FOIL_KERNEL_DIR/add_brisc.elf
//      (see kernels/add_brisc.cpp and the build script in prebuilt/)
//
// Run:
//   TT_FOIL_KERNEL_DIR=prebuilt ./add_two_numbers

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

#include "tt_foil/runtime.hpp"

int main() try {
    const char* kernel_dir_env = std::getenv("TT_FOIL_KERNEL_DIR");

    if (!kernel_dir_env) {
        std::puts("Usage: TT_FOIL_KERNEL_DIR=prebuilt ./add_two_numbers");
        return 1;
    }

    std::string kernel_dir = kernel_dir_env;

    // Open the first Blackhole chip (firmware init delegated to tt-metal CreateDevice).
    auto dev = tt::foil::open_device(0);

    // Use logical core (0,0).
    tt::foil::CoreCoord core{0, 0};

    // Allocate three 4-byte L1 buffers on that core.
    auto buf_a = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);
    auto buf_b = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);
    auto buf_r = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, 4, core);

    // Write operands.
    uint32_t a = 42, b = 58, zero = 0;
    tt::foil::write_buffer(*dev, *buf_a, &a, 4);
    tt::foil::write_buffer(*dev, *buf_b, &b, 4);
    tt::foil::write_buffer(*dev, *buf_r, &zero, 4);

    // Load pre-compiled BRISC kernel.
    std::string elf = kernel_dir + "/add_brisc.elf";
    std::array<tt::foil::RiscBinary, 1> bins = {{
        {tt::foil::RiscBinary::RiscId::BRISC, elf},
    }};
    auto kernel = tt::foil::load_kernel(*dev, bins, core);

    // Pass buffer addresses as runtime args.
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        static_cast<uint32_t>(buf_r->device_addr),
    };
    tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::BRISC, args);

    // Execute and wait for completion.
    tt::foil::execute(*dev, *kernel);

    // Read back the result.
    uint32_t result = 0;
    tt::foil::read_buffer(*dev, *buf_r, &result, 4);

    std::printf("add_two_numbers: %u + %u = %u\n", a, b, result);

    tt::foil::close_device(std::move(dev));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
