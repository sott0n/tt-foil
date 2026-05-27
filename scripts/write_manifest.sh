#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone wrapper around write_kernel_manifest (defined in
# kernel_build_helpers.sh). Lets a build_kernels.sh / build.sh add a single
# trailing line to record what firmware + sources its ELFs were linked
# against, without needing to source the helper file itself.
#
# Usage:
#   bash scripts/write_manifest.sh PREBUILT FW_DIR SRC_DIR [extra_files...]
#
# See scripts/kernel_build_helpers.sh::write_kernel_manifest for details.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/kernel_build_helpers.sh"
write_kernel_manifest "$@"
