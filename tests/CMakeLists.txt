cmake_minimum_required(VERSION 3.21)

# TT_FOIL_DEVICE picks the PCIe board index used by HW tests. Resolution
# order: CMake -DTT_FOIL_DEVICE=N > env TT_FOIL_DEVICE > 0.
if(NOT DEFINED TT_FOIL_DEVICE)
    if(DEFINED ENV{TT_FOIL_DEVICE})
        set(TT_FOIL_DEVICE "$ENV{TT_FOIL_DEVICE}")
    else()
        set(TT_FOIL_DEVICE 0)
    endif()
endif()
message(STATUS "tt-foil tests: TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")

# ---------------------------------------------------------------------------
# tt_foil_unit_test(name src ...)
#   Adds a host-side unit test (no HW). Optional remaining args are passed as
#   target_compile_definitions PRIVATE values.
# ---------------------------------------------------------------------------
function(tt_foil_unit_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE TT::Foil tt_foil_llrt)
    if(ARGN)
        target_compile_definitions(${name} PRIVATE ${ARGN})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "unit" TIMEOUT 30)
endfunction()

# ---------------------------------------------------------------------------
# tt_foil_hw_test(name src kernel_dir ...)
#   Adds a hardware integration test. The test binary gets:
#     TT_FOIL_KERNEL_DIR  pointed at `kernel_dir`
#     TT_FOIL_DEVICE      pointed at the CMake variable above
#   All HW tests share `RESOURCE_LOCK chip` so ctest serialises them
#   without running them concurrently.  Remaining args go to
#   target_compile_definitions PRIVATE.
# ---------------------------------------------------------------------------
function(tt_foil_hw_test name src kernel_dir)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE TT::Foil tt_foil_llrt)
    if(ARGN)
        target_compile_definitions(${name} PRIVATE ${ARGN})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    # set_tests_properties splits the ENVIRONMENT value on semicolons
    # for its own list parsing; pass it as a single quoted string and
    # set each property in its own call so list-ish values don't get
    # split into argument-count errors.
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "TT_FOIL_KERNEL_DIR=${kernel_dir};TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")
    set_tests_properties(${name} PROPERTIES RESOURCE_LOCK chip)
    set_tests_properties(${name} PROPERTIES LABELS "hw")
    set_tests_properties(${name} PROPERTIES TIMEOUT 90)
endfunction()

# =============================================================================
# Host-only unit tests
# =============================================================================
tt_foil_unit_test(test_elf_load        test_elf_load.cpp)
tt_foil_unit_test(test_bump_alloc      test_bump_alloc.cpp)
tt_foil_unit_test(test_risc_ids        test_risc_ids.cpp)
tt_foil_unit_test(test_hal_standalone  test_hal_standalone.cpp)
tt_foil_unit_test(test_firmware_paths  test_firmware_paths.cpp)
tt_foil_unit_test(test_bn_fold         test_bn_fold.cpp)

# =============================================================================
# Hardware integration tests
# Enable with -DTT_FOIL_HW_TESTS=ON
# =============================================================================
option(TT_FOIL_HW_TESTS "Build hardware integration tests" OFF)
if(NOT TT_FOIL_HW_TESTS)
    return()
endif()

set(EX "${CMAKE_SOURCE_DIR}/examples")
set(MO "${CMAKE_SOURCE_DIR}/models")

# NOTE: HW tests assume the board is in a healthy state at ctest
# invocation time. If you've been crashing kernels, run a one-shot
# `tt-smi -r <TT_FOIL_DEVICE>` before `ctest`. RESOURCE_LOCK chip
# serialises HW tests within one ctest run, but ctest cannot revive a
# chip that a prior test hung.

# Per-config tile-count defaults. Override via CMake -DMM_XX=N.
macro(_tt_default var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()
_tt_default(MM_MT      2)
_tt_default(MM_KT      4)
_tt_default(MM_NT      2)
_tt_default(MM_MT_2C   1)
_tt_default(MM_MT_2x2  1)
_tt_default(MM_NT_2x2  1)
_tt_default(MM_MT_2CMC 1)
_tt_default(MM_MT_4x2  1)
_tt_default(MM_NT_4x2  1)
_tt_default(MM_MT_C1   2)
_tt_default(MM_KT_C1   2)
_tt_default(MM_NT_C1   2)
_tt_default(MM_MT_C3   1)
_tt_default(MM_KT_C3   9)
_tt_default(MM_NT_C3   2)
_tt_default(MM_MT_C32  1)
_tt_default(MM_KT_C32  9)
_tt_default(MM_NT_C32  2)
_tt_default(MM_MT_CS2  1)
_tt_default(MM_KT_CS2  9)
_tt_default(MM_NT_CS2  2)

# ---- Boot / IO smoke (no kernel ELFs needed beyond firmware) ----
tt_foil_hw_test(test_umd_open         test_umd_open.cpp         "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_multi_core_boot  test_multi_core_boot.cpp  "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_dram_roundtrip   test_dram_roundtrip.cpp   "${EX}/add_two_numbers/prebuilt")

# ---- v1..v4: foundational dispatch + 5-RISC + CBs ----
tt_foil_hw_test(test_add_kernel       test_add_kernel.cpp       "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_brisc_ncrisc     test_brisc_ncrisc.cpp     "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_multi_kernel     test_multi_kernel.cpp     "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_noc_passthrough  test_noc_passthrough.cpp  "${EX}/noc_passthrough/prebuilt")
tt_foil_hw_test(test_cb_config        test_cb_config.cpp        "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_tile_copy        test_tile_copy.cpp        "${EX}/tile_copy/prebuilt")

# ---- v5: matmul + ReLU + add + bias ----
tt_foil_hw_test(test_matmul_1tile     test_matmul_1tile.cpp     "${EX}/matmul_1tile/prebuilt")
tt_foil_hw_test(test_matmul_kt        test_matmul_kt.cpp        "${EX}/matmul_kt/prebuilt"
    "MM_KT=${MM_KT}")
tt_foil_hw_test(test_matmul_mnk       test_matmul_mnk.cpp       "${EX}/matmul_mnk/prebuilt"
    "MM_MT=${MM_MT}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_dram      test_matmul_dram.cpp      "${EX}/matmul_dram/prebuilt"
    "MM_MT=${MM_MT}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_relu             test_relu.cpp             "${EX}/relu/prebuilt")
tt_foil_hw_test(test_add_tiles        test_add_tiles.cpp        "${EX}/add_tiles/prebuilt")
tt_foil_hw_test(test_bias_add         test_bias_add.cpp         "${EX}/bias_add/prebuilt")

# ---- v6: multi-core sharding + NOC mcast ----
tt_foil_hw_test(test_matmul_2core         test_matmul_2core.cpp        "${EX}/matmul_2core/prebuilt"
    "MM_MT=${MM_MT_2C}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_2x2           test_matmul_2x2.cpp          "${EX}/matmul_2x2/prebuilt"
    "MM_MT=${MM_MT_2x2}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT_2x2}")
tt_foil_hw_test(test_matmul_2core_mcast   test_matmul_2core_mcast.cpp  "${EX}/matmul_2core_mcast/prebuilt"
    "MM_MT=${MM_MT_2CMC}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_4x2           test_matmul_4x2.cpp          "${EX}/matmul_4x2/prebuilt"
    "MM_MT=${MM_MT_4x2}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT_4x2}")

# ---- v7: conv (1×1, 3×3, sharded, stride=2) ----
tt_foil_hw_test(test_conv_1x1         test_conv_1x1.cpp         "${EX}/conv_1x1/prebuilt"
    "MM_MT=${MM_MT_C1}"  "MM_KT=${MM_KT_C1}"  "MM_NT=${MM_NT_C1}")
tt_foil_hw_test(test_conv_3x3         test_conv_3x3.cpp         "${EX}/conv_3x3/prebuilt"
    "MM_MT=${MM_MT_C3}"  "MM_KT=${MM_KT_C3}"  "MM_NT=${MM_NT_C3}")
tt_foil_hw_test(test_conv_3x3_2core   test_conv_3x3_2core.cpp   "${EX}/conv_3x3_2core/prebuilt"
    "MM_MT=${MM_MT_C32}" "MM_KT=${MM_KT_C32}" "MM_NT=${MM_NT_C32}")
tt_foil_hw_test(test_conv_3x3_s2      test_conv_3x3_s2.cpp      "${EX}/conv_3x3_s2/prebuilt"
    "MM_MT=${MM_MT_CS2}" "MM_KT=${MM_KT_CS2}" "MM_NT=${MM_NT_CS2}")

# ---- v8: peripheral ops (pool, BN fold) ----
tt_foil_hw_test(test_maxpool_2x2      test_maxpool_2x2.cpp      "${EX}/maxpool_2x2/prebuilt")

# ---- v9: residual add (ResNet skip-connection building block) ----
_tt_default(RA_NT 8)
tt_foil_hw_test(test_residual_add     test_residual_add.cpp     "${EX}/residual_add/prebuilt"
    "RA_NT=${RA_NT}")

# ---- v9.1: ResNet basic block (Conv → bias+ReLU → Conv → bias → skip+ReLU) ----
# This test needs two kernel dirs: conv_3x3 prebuilt for both conv stages
# and a residual_add build compiled with RA_NT=2 (matching this test's
# Mt*Nt) under models/basic_block/prebuilt.
add_executable(test_basic_block test_basic_block.cpp)
target_include_directories(test_basic_block PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(test_basic_block PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_basic_block COMMAND test_basic_block)
set_tests_properties(test_basic_block PROPERTIES
    ENVIRONMENT "TT_FOIL_KERNEL_DIR=${EX}/conv_3x3/prebuilt;TT_FOIL_KERNEL_DIR_ADD=${MO}/basic_block/prebuilt;TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")
set_tests_properties(test_basic_block PROPERTIES RESOURCE_LOCK chip)
set_tests_properties(test_basic_block PROPERTIES LABELS "hw")
set_tests_properties(test_basic_block PROPERTIES TIMEOUT 90)

# ---- v9.2: ResNet downsample block (3×3 s=2 main + 1×1 s=2 projection skip) ----
# Single kernel root with four subdirs (conv_s2, conv, conv1x1, residual_add)
# produced by models/downsample_block/build_kernels.sh.
tt_foil_hw_test(test_downsample_block test_downsample_block.cpp
    "${MO}/downsample_block/prebuilt")

# ---- v9.3: ResNet layer (downsample_block + basic_block chained) ----
# Reuses the downsample_block kernel root — all needed binaries
# (conv_s2, conv, conv1x1, residual_add) already match this test's
# tile shapes since C stays at 32 and spatial dims stay 8×8 after
# block 1.
tt_foil_hw_test(test_layer            test_layer.cpp
    "${MO}/downsample_block/prebuilt")

# ---- v9.4: 7×7 strided conv (ResNet stem input layer) ----
_tt_default(MM_MT_C7  1)
_tt_default(MM_KT_C7  49)   # C_in * 49 / 32 ; assumes C_in = 32
_tt_default(MM_NT_C7  2)
tt_foil_hw_test(test_conv_7x7         test_conv_7x7.cpp         "${EX}/conv_7x7/prebuilt"
    "MM_MT=${MM_MT_C7}" "MM_KT=${MM_KT_C7}" "MM_NT=${MM_NT_C7}")

# ---- v9.5: 3×3 stride-2 maxpool (ResNet stem post-conv) ----
tt_foil_hw_test(test_maxpool_3x3      test_maxpool_3x3.cpp      "${EX}/maxpool_3x3/prebuilt")

# ---- v9.6: ResNet stem (Conv₇ₓ₇ → bias+ReLU → Maxpool₃ₓ₃) ----
# Conv₇ₓ₇ is rebuilt at Nt=8 (input H=W=32 → H=W=16) under
# stem/prebuilt/conv_7x7. Maxpool₃ₓ₃ matches its standalone default
# fixture and is symlinked in.
tt_foil_hw_test(test_stem             test_stem.cpp             "${MO}/stem/prebuilt")

# ---- v9.7: Mini ResNet (stem + 2× basic_block) ----
# All four kernel programs (conv_7x7, maxpool_3x3, conv, residual_add)
# are symlinked into models/mini_resnet/prebuilt by its build script;
# no fresh compiles needed for this test.
tt_foil_hw_test(test_mini_resnet      test_mini_resnet.cpp      "${MO}/mini_resnet/prebuilt")

# ---- v9.8: classifier tail (host GAP + device FC + host bias) ----
tt_foil_hw_test(test_classifier_tail  test_classifier_tail.cpp  "${MO}/classifier_tail/prebuilt")

# ---- v9.9: Full ResNet classifier — image in, logits out ----
tt_foil_hw_test(test_resnet_classifier test_resnet_classifier.cpp
    "${MO}/resnet_classifier/prebuilt")

# ---- v10.1: bias_relu_post primitive ----
tt_foil_hw_test(test_bias_relu_post   test_bias_relu_post.cpp   "${EX}/bias_relu_post/prebuilt")
tt_foil_hw_test(test_global_avg_pool  test_global_avg_pool.cpp  "${EX}/global_avg_pool/prebuilt")
tt_foil_hw_test(test_brisc_col1_boot_diag test_brisc_col1_boot_diag.cpp "${EX}/add_two_numbers/prebuilt")

# ---- op_lib: primitive op library tests ----
set(OPS "${CMAKE_SOURCE_DIR}/ops")
_tt_default(SILU_NT 8)
tt_foil_hw_test(test_silu        test_silu.cpp        "${OPS}/eltwise_unary/prebuilt"
    "SILU_NT=${SILU_NT}")
_tt_default(EMUL_NT 8)
tt_foil_hw_test(test_eltwise_mul test_eltwise_mul.cpp "${OPS}/eltwise_binary/prebuilt"
    "EMUL_NT=${EMUL_NT}")
_tt_default(RMS_NCHT 2)
_tt_default(RMS_WT 4)
tt_foil_hw_test(test_rmsnorm test_rmsnorm.cpp "${OPS}/rmsnorm/prebuilt"
    "RMS_NCHT=${RMS_NCHT}" "RMS_WT=${RMS_WT}")
_tt_default(SOFTMAX_NCHT 2)
_tt_default(SOFTMAX_WT 4)
tt_foil_hw_test(test_softmax test_softmax.cpp "${OPS}/softmax/prebuilt"
    "SOFTMAX_NCHT=${SOFTMAX_NCHT}" "SOFTMAX_WT=${SOFTMAX_WT}")
_tt_default(MHA_ST 2)
_tt_default(MHA_DT 2)
tt_foil_hw_test(test_mha test_mha.cpp "${OPS}/mha/prebuilt"
    "MHA_ST=${MHA_ST}" "MHA_DT=${MHA_DT}")
tt_foil_hw_test(test_embedding test_embedding.cpp "${OPS}/embedding/prebuilt")
# op_lib host abstraction
_tt_default(SILU_OP_NT 4)
tt_foil_hw_test(test_silu_op test_silu_op.cpp "${OPS}/eltwise_unary/prebuilt"
    "SILU_OP_NT=${SILU_OP_NT}")
_tt_default(EMUL_OP_NT 4)
tt_foil_hw_test(test_eltwise_mul_op test_eltwise_mul_op.cpp "${OPS}/eltwise_binary/prebuilt"
    "EMUL_OP_NT=${EMUL_OP_NT}")
_tt_default(EADD_OP_NT 4)
tt_foil_hw_test(test_eltwise_add_op test_eltwise_add_op.cpp "${OPS}/eltwise_binary/prebuilt"
    "EADD_OP_NT=${EADD_OP_NT}")
_tt_default(SOFTMAX_OP_NCHT 2)
_tt_default(SOFTMAX_OP_WT 4)
tt_foil_hw_test(test_softmax_op test_softmax_op.cpp "${OPS}/softmax/prebuilt"
    "SOFTMAX_OP_NCHT=${SOFTMAX_OP_NCHT}" "SOFTMAX_OP_WT=${SOFTMAX_OP_WT}")
_tt_default(RMS_OP_NCHT 2)
_tt_default(RMS_OP_WT 4)
tt_foil_hw_test(test_rmsnorm_op test_rmsnorm_op.cpp "${OPS}/rmsnorm/prebuilt"
    "RMS_OP_NCHT=${RMS_OP_NCHT}" "RMS_OP_WT=${RMS_OP_WT}")
_tt_default(MHA_OP_ST 2)
_tt_default(MHA_OP_DT 2)
tt_foil_hw_test(test_mha_op test_mha_op.cpp "${OPS}/mha/prebuilt"
    "MHA_OP_ST=${MHA_OP_ST}" "MHA_OP_DT=${MHA_OP_DT}")
tt_foil_hw_test(test_embedding_op test_embedding_op.cpp "${OPS}/embedding/prebuilt")
tt_foil_hw_test(test_rope_op test_rope_op.cpp "${OPS}/rope/prebuilt")
_tt_default(MM_OP_MT 2)
_tt_default(MM_OP_KT 4)
_tt_default(MM_OP_NT 2)
tt_foil_hw_test(test_matmul_op test_matmul_op.cpp "${OPS}/matmul/prebuilt"
    "MM_OP_MT=${MM_OP_MT}" "MM_OP_KT=${MM_OP_KT}" "MM_OP_NT=${MM_OP_NT}")
# KV cache (Phase 1-C minimum): no kernels, but needs HW for buffer R/W.
tt_foil_hw_test(test_kv_cache test_kv_cache.cpp "${OPS}/eltwise_unary/prebuilt")
# Multi-op chain — needs all op_lib kernels; TT_FOIL_OPS_DIR points at ops/
# so each make_* picks up its own prebuilt subdir.
tt_foil_hw_test(test_transformer_block test_transformer_block.cpp "${OPS}")
set_tests_properties(test_transformer_block PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")
set_tests_properties(test_transformer_block PROPERTIES TIMEOUT 180)
# Real-weights smoke test: Qwen3-0.6B layer-0 MLP half.
# Needs data/qwen3_06b/layer0/ populated by tools/export_qwen3_layer.py +
# tools/qwen3_mlp_golden.py.
tt_foil_hw_test(test_qwen3_mlp test_qwen3_mlp.cpp "${OPS}")
set_tests_properties(test_qwen3_mlp PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_06b/layer0")
set_tests_properties(test_qwen3_mlp PROPERTIES TIMEOUT 300)
# Real-weights attention test: Qwen3-VL-2B layer-0 attention block.
# Needs data/qwen3_vl_2b/layer0/ populated by tools/export_qwen3_layer.py +
# tools/qwen3_attn_golden.py.
tt_foil_hw_test(test_qwen3_attn test_qwen3_attn.cpp "${OPS}")
set_tests_properties(test_qwen3_attn PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b/layer0")
set_tests_properties(test_qwen3_attn PROPERTIES TIMEOUT 600)
# Real-weights full-layer test: Qwen3-VL-2B layer-0 attention + MLP + 2 residuals.
# Needs data/qwen3_vl_2b/layer0/ populated by tools/export_qwen3_layer.py +
# tools/qwen3_layer_golden.py.
tt_foil_hw_test(test_qwen3_layer test_qwen3_layer.cpp "${OPS}")
set_tests_properties(test_qwen3_layer PROPERTIES
    ENVIRONMENT "TT_FOIL_OPS_DIR=${OPS};TT_FOIL_DEVICE=${TT_FOIL_DEVICE};TT_FOIL_QWEN3_DATA=${CMAKE_SOURCE_DIR}/data/qwen3_vl_2b/layer0")
set_tests_properties(test_qwen3_layer PROPERTIES TIMEOUT 900)

# ---- v10.2: CIFAR-10 ResNet-20 end-to-end inference ----
# Reads pretrained weights + sample image + golden logits from
# data/cifar10_resnet20/ (produced by tools/export_cifar10_resnet20.py).
add_executable(test_cifar10_resnet20 test_cifar10_resnet20.cpp)
target_include_directories(test_cifar10_resnet20 PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(test_cifar10_resnet20 PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_cifar10_resnet20 COMMAND test_cifar10_resnet20)
set_tests_properties(test_cifar10_resnet20 PROPERTIES
    ENVIRONMENT "TT_FOIL_KERNEL_DIR=${MO}/cifar10_resnet20/prebuilt;TT_FOIL_DATA_DIR=${CMAKE_SOURCE_DIR}/data/cifar10_resnet20;TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")
set_tests_properties(test_cifar10_resnet20 PROPERTIES RESOURCE_LOCK chip)
set_tests_properties(test_cifar10_resnet20 PROPERTIES LABELS "hw")
set_tests_properties(test_cifar10_resnet20 PROPERTIES TIMEOUT 120)

# NOTE: The Phase B2 step-by-step tests (test_cold_boot, test_firmware_load,
# test_mailbox_init, test_core_info_init, test_bank_tables_init) were
# development scaffolding that linked against tt-metal's CreateDevice +
# MetalContext. They were removed when libtt_metal.so was unlinked (B3-4).
# Their production-equivalent coverage lives in test_add_kernel +
# test_brisc_ncrisc.
