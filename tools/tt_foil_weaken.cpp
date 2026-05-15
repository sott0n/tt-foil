// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tt_foil_weaken — host CLI that produces a *_weakened.elf from a firmware
// ELF.  Mirrors tt-metal's JitBuildState::weaken() (see
// tt_metal/jit_build/build.cpp:weaken): weakens all data symbols except a
// small set of "strong names" so kernel ELFs can be linked against the
// firmware via -Wl,--just-symbols without colliding on shared globals.
//
//   Usage: tt_foil_weaken <input.elf> <output_weakened.elf>
//
// Strong names match tt-metal exactly: "__fw_export_*" (any fw_export
// pointer that propagates link addresses into the kernel) and
// "__global_pointer$".

#include <cstdio>
#include <span>
#include <string>
#include <string_view>

#include "tt_elffile.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <input.elf> <output_weakened.elf>\n",
                     argv[0]);
        return 1;
    }

    try {
        tt::foil::ll_api::ElfFile elf;
        elf.ReadImage(argv[1]);
        static constexpr std::string_view strong_names[] = {
            "__fw_export_*",
            "__global_pointer$",
        };
        elf.WeakenDataSymbols(std::span<const std::string_view>{strong_names});
        // NOTE: We do NOT call ObjectifyExecutable here.  That path is for
        // kernel-as-object firmware variants (firmware_is_kernel_object_),
        // which is not used for the 5 RISCs tt-foil supports today.
        elf.WriteImage(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tt_foil_weaken: %s\n", e.what());
        return 2;
    }
    return 0;
}
