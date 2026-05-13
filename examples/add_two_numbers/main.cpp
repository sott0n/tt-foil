// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Example: add two numbers on Blackhole using a pre-compiled BRISC kernel.
//
// Prerequisites:
//   1. Pre-built management firmware in $TT_FOIL_FW_DIR
//   2. Pre-compiled kernel at $TT_FOIL_KERNEL_DIR/add_brisc.elf
//
// Build the kernel first:
//   riscv32-unknown-elf-g++ -march=rv32imc -mabi=ilp32 \
//     -O2 -fno-exceptions -fno-rtti \
//     -I<tt-metal>/tt_metal/hw/inc \
//     -I<tt-metal>/tt_metal/hw/inc/hostdev \
//     kernels/add_brisc.cpp -o add_brisc.elf
//
// Run:
//   TT_FOIL_FW_DIR=/path/fw TT_FOIL_KERNEL_DIR=/path/kernels ./add_two_numbers

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

#include "tt_foil/runtime.hpp"

int main() try {
    const char* fw_dir_env     = std::getenv("TT_FOIL_FW_DIR");
    const char* kernel_dir_env = std::getenv("TT_FOIL_KERNEL_DIR");

    if (!fw_dir_env || !kernel_dir_env) {
        std::puts("Usage: TT_FOIL_FW_DIR=... TT_FOIL_KERNEL_DIR=... ./add_two_numbers");
        return 1;
    }

    std::string fw_dir     = fw_dir_env;
    std::string kernel_dir = kernel_dir_env;

    // Open the first Blackhole chip.
    auto dev = tt::foil::open_device(0, fw_dir);

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
