// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Integration test: launch BRISC and NCRISC concurrently on the same Tensix
// core, each producing a different result derived from shared L1 operands.
//
// BRISC kernel : *sum  = *a + *b
// NCRISC kernel: *prod = *a * *b
//
// This verifies that:
//   - kernel_config.enables = (1<<0) | (1<<1) correctly arms both RISCs.
//   - Per-RISC RTA slots (rta_offset[0] vs rta_offset[1]) are independent.
//   - The single go_msg -> RUN_MSG_DONE transition reflects completion of
//     all enabled subordinate RISCs (BRISC fw waits for NCRISC done internally).
//
// Usage:
//   TT_FOIL_KERNEL_DIR=examples/add_two_numbers/prebuilt \
//   TT_FOIL_DEVICE=3 ./test_brisc_ncrisc

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

    const char* dev_env = std::getenv("TT_FOIL_DEVICE");
    int pcie_index = dev_env ? std::stoi(dev_env) : 0;
    auto dev = tt::foil::open_device(pcie_index);
    std::puts("test_brisc_ncrisc: device opened");

    tt::foil::CoreCoord core{0, 0};

    // Shared inputs + two distinct result slots.
    auto buf_a    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);
    auto buf_b    = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);
    auto buf_sum  = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);
    auto buf_prod = tt::foil::allocate_buffer(*dev, tt::foil::BufferLocation::L1, sizeof(uint32_t), core);

    uint32_t a = 7, b = 6;
    uint32_t zero = 0;
    tt::foil::write_buffer(*dev, *buf_a,    &a,    sizeof(a));
    tt::foil::write_buffer(*dev, *buf_b,    &b,    sizeof(b));
    tt::foil::write_buffer(*dev, *buf_sum,  &zero, sizeof(zero));
    tt::foil::write_buffer(*dev, *buf_prod, &zero, sizeof(zero));
    std::puts("test_brisc_ncrisc: buffers written (a=7, b=6)");

    // Load BOTH binaries onto the same core.
    std::string brisc_elf  = kernel_dir + "/add_brisc.elf";
    std::string ncrisc_elf = kernel_dir + "/add_ncrisc.elf";
    std::array<tt::foil::RiscBinary, 2> binaries = {{
        {tt::foil::RiscBinary::RiscId::BRISC,  brisc_elf},
        {tt::foil::RiscBinary::RiscId::NCRISC, ncrisc_elf},
    }};
    auto kernel = tt::foil::load_kernel(*dev, binaries, core);
    std::puts("test_brisc_ncrisc: BRISC + NCRISC kernels loaded");

    // BRISC writes the sum slot; NCRISC writes the product slot.
    std::vector<uint32_t> brisc_args = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        static_cast<uint32_t>(buf_sum->device_addr),
    };
    std::vector<uint32_t> ncrisc_args = {
        static_cast<uint32_t>(buf_a->device_addr),
        static_cast<uint32_t>(buf_b->device_addr),
        static_cast<uint32_t>(buf_prod->device_addr),
    };
    tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::BRISC,  brisc_args);
    tt::foil::set_runtime_args(*dev, *kernel, tt::foil::RiscBinary::RiscId::NCRISC, ncrisc_args);

    tt::foil::execute(*dev, *kernel);
    std::puts("test_brisc_ncrisc: execute returned");

    uint32_t sum = 0, prod = 0;
    tt::foil::read_buffer(*dev, *buf_sum,  &sum,  sizeof(sum));
    tt::foil::read_buffer(*dev, *buf_prod, &prod, sizeof(prod));

    std::printf("test_brisc_ncrisc: BRISC  sum  = %u (expected %u)\n", sum,  a + b);
    std::printf("test_brisc_ncrisc: NCRISC prod = %u (expected %u)\n", prod, a * b);
    assert(sum  == a + b && "BRISC kernel did not run or produced wrong result");
    assert(prod == a * b && "NCRISC kernel did not run or produced wrong result");

    tt::foil::close_device(std::move(dev));
    std::puts("test_brisc_ncrisc: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_brisc_ncrisc: FAIL — %s\n", e.what());
    return 1;
}
