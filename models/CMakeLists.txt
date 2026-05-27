# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Model-specific integration tests, opt-in via -DTT_FOIL_MODEL_TESTS=ON.
# These tests need exported weights/fixtures that aren't in the repo
# (multi-GB binaries); see the per-model README for the export step.
# The runtime regression sweep (default `ctest`) should NOT include these.

add_subdirectory(qwen3_vl_2b/tests)
