#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# 7×7 strided convolution for the ResNet stem. The device side is just
# another matmul over an im2col-style A operand — only the K dimension
# changes (Kt = Cin*49 / 32 instead of Cin*9 / 32). We delegate to
# conv_3x3_s2's builder with MM_KT overridden so we don't duplicate the
# whole compile recipe; conv_3x3_s2's kernels are bit-identical to
# matmul_dram's and any K-loop count is fine on the device.
#
# Default shape (overridable via env):
#   MM_MT = 1        Cout / 32
#   MM_KT = 49       Cin * 7 * 7 / 32   (assumes Cin = 32)
#   MM_NT = 2        H_out * W_out / 32
#
# Output:
#   prebuilt/{reader.brisc.elf, writer.ncrisc.elf, compute.trisc{0,1,2}.elf}

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT_DIR="${PREBUILT_DIR:-$HERE/prebuilt}" \
    MM_MT="${MM_MT:-1}" MM_KT="${MM_KT:-49}" MM_NT="${MM_NT:-2}" \
    bash "$HERE/../conv_3x3_s2/build_kernels.sh"

# Stale-ELF guard: the delegated conv_3x3_s2 builder writes the manifest
# into PREBUILT_DIR for us; nothing else to do here.
