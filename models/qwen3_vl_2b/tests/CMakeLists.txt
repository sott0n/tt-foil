# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3-VL-2B integration tests. Each test depends on real-weight binaries
# under data/qwen3_vl_2b/ (and data/qwen3_06b/ for the smaller MLP test);
# run scripts/qwen3_export_weights.sh to produce them.
#
# These are NOT part of the runtime regression sweep. They are only
# registered when -DTT_FOIL_MODEL_TESTS=ON (set in the top-level CMake).
# All tests carry the "model" label so `ctest -L model` runs only these.
#
# tt_foil_hw_test() and _tt_default() are defined in tests/CMakeLists.txt
# and are global function scope, so they're visible here.

set(OPS "${CMAKE_SOURCE_DIR}/ops")

# Real-weights smoke test: Qwen3-0.6B layer-0 MLP half.
# Needs data/qwen3_06b/layer0/ populated by models/qwen3_vl_2b/export_qwen3_layer.py +
# models/qwen3_vl_2b/golden/qwen3_mlp_golden.py.
tt_foil_hw_test(test_qwen3_mlp test_qwen3_mlp.cpp "${OPS}")
set_tests_properties(test_qwen3_mlp PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_06b/layer0"
    TIMEOUT 300
    LABELS "model;hw")

# Real-weights attention test: Qwen3-VL-2B layer-0 attention block.
# Needs data/qwen3_vl_2b/layer0/ populated by models/qwen3_vl_2b/export_qwen3_layer.py +
# models/qwen3_vl_2b/golden/qwen3_attn_golden.py.
tt_foil_hw_test(test_qwen3_attn test_qwen3_attn.cpp "${OPS}")
set_tests_properties(test_qwen3_attn PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b/layer0"
    TIMEOUT 600
    LABELS "model;hw")

# Real-weights full-layer test: Qwen3-VL-2B layer-0 attention + MLP + 2 residuals.
# Needs data/qwen3_vl_2b/layer0/ populated by models/qwen3_vl_2b/export_qwen3_layer.py +
# models/qwen3_vl_2b/golden/qwen3_layer_golden.py.
tt_foil_hw_test(test_qwen3_layer test_qwen3_layer.cpp "${OPS}")
set_tests_properties(test_qwen3_layer PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b/layer0"
    TIMEOUT 900
    LABELS "model;hw")

# Multi-layer chain: stacks N full layers (default 3). Needs
# data/qwen3_vl_2b/{layer0..layer<N-1>}/ + chain<N>/ from
# models/qwen3_vl_2b/golden/qwen3_multilayer_golden.py.
_tt_default(CHAIN_NUM_LAYERS 3)
tt_foil_hw_test(test_qwen3_multilayer test_qwen3_multilayer.cpp "${OPS}"
    "CHAIN_NUM_LAYERS=${CHAIN_NUM_LAYERS}")
set_tests_properties(test_qwen3_multilayer PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b"
    TIMEOUT 1800
    LABELS "model;hw")

# Real-weights embedding lookup test on Qwen3-VL-2B.
# Needs data/qwen3_vl_2b/model/embed_tokens.bin from
# models/qwen3_vl_2b/export_qwen3_layer.py --model-tensors.
tt_foil_hw_test(test_qwen3_embed test_qwen3_embed.cpp "${OPS}/embedding/prebuilt")
set_tests_properties(test_qwen3_embed PROPERTIES
    ENVIRONMENT "TT_FOIL_KERNEL_DIR=${OPS}/embedding/prebuilt;TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b"
    TIMEOUT 120
    LABELS "model;hw")

# lm_head matmul on real (tied) embed weights. Needs the pre-tiled
# lm_head_tiled.bin + hidden / golden binaries from
# models/qwen3_vl_2b/golden/qwen3_lm_head_golden.py. Nt=4748 is much larger than any other
# matmul in tt-foil so the timeout is generous.
tt_foil_hw_test(test_qwen3_lm_head test_qwen3_lm_head.cpp "${OPS}/matmul/prebuilt")
set_tests_properties(test_qwen3_lm_head PROPERTIES
    ENVIRONMENT "TT_FOIL_KERNEL_DIR=${OPS}/matmul/prebuilt;TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b"
    TIMEOUT 1800
    LABELS "model;hw")

# Full prefill forward pass: embed → N layers → final norm → lm_head → argmax.
# Default INF_NUM_LAYERS=3 (matches data/qwen3_vl_2b/chain3/ produced by
# models/qwen3_vl_2b/golden/qwen3_inference_golden.py).
_tt_default(INF_NUM_LAYERS 3)
tt_foil_hw_test(test_qwen3_inference test_qwen3_inference.cpp "${OPS}"
    "INF_NUM_LAYERS=${INF_NUM_LAYERS}")
set_tests_properties(test_qwen3_inference PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b"
    TIMEOUT 3600
    LABELS "model;hw")

# Prefill + 1-step decode w/ KV cache. Default DEC_NUM_LAYERS=3
# (matches chain3/, produced by models/qwen3_vl_2b/golden/qwen3_inference_golden.py which
# now also emits decode_input.bin / decode_top1.bin / decode_cos.bin /
# decode_sin.bin alongside the prefill golden).
_tt_default(DEC_NUM_LAYERS 3)
tt_foil_hw_test(test_qwen3_decode test_qwen3_decode.cpp "${OPS}"
    "DEC_NUM_LAYERS=${DEC_NUM_LAYERS}")
set_tests_properties(test_qwen3_decode PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b"
    TIMEOUT 3600
    LABELS "model;hw")
