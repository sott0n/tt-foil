# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# tt_foil_add_kernels — register a build_kernels.sh invocation as a CMake
# custom target so kernel ELFs are automatically (re)built when their
# sources, the build script, or the underlying firmware change.
#
# Usage:
#   tt_foil_add_kernels(
#       NAME    add_two_numbers
#       SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/build_kernels.sh
#       SOURCES kernels/add_brisc.cpp
#               kernels/add_ncrisc.cpp
#       OUTPUTS prebuilt/add_brisc.elf
#               prebuilt/add_ncrisc.elf
#               prebuilt/manifest.txt
#       [WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}]
#       [ENV VAR=value VAR2=value2 ...]
#   )
#
# Effects:
#   - Creates a custom command whose OUTPUTS depend on SOURCES, the SCRIPT
#     itself, and the tt_foil_firmware target. Touching any of those triggers
#     a rebuild on the next `cmake --build`.
#   - Creates a custom target `kernels_<NAME>` that aggregates the outputs.
#     Test or example targets should declare add_dependencies(... kernels_<NAME>)
#     so kernel ELFs are guaranteed fresh before running.
#
# Why: this catches the "I changed a kernel source but forgot to rerun
# build_kernels.sh" and "I rebuilt firmware but the kernel ELFs still link
# against the old *_weakened.elf" classes of bug at `cmake --build` time —
# the manifest check in src/kernel_manifest.cpp is the runtime safety net
# (off by default) for cases that bypass CMake (manual prebuilt/ copies,
# TT_FOIL_KERNEL_DIR pointing elsewhere, etc.).

include_guard(GLOBAL)

function(tt_foil_add_kernels)
    set(_options)
    set(_one_value NAME SCRIPT WORKING_DIRECTORY)
    set(_multi_value SOURCES OUTPUTS ENV)
    cmake_parse_arguments(TFK "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})

    if(NOT TFK_NAME)
        message(FATAL_ERROR "tt_foil_add_kernels: NAME is required")
    endif()
    if(NOT TFK_SCRIPT)
        message(FATAL_ERROR "tt_foil_add_kernels(${TFK_NAME}): SCRIPT is required")
    endif()
    if(NOT TFK_OUTPUTS)
        message(FATAL_ERROR "tt_foil_add_kernels(${TFK_NAME}): OUTPUTS is required")
    endif()
    if(NOT TFK_WORKING_DIRECTORY)
        set(TFK_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    # Normalise OUTPUTS to absolute paths under WORKING_DIRECTORY so callers
    # can pass either form.
    set(_abs_outputs)
    foreach(o IN LISTS TFK_OUTPUTS)
        if(IS_ABSOLUTE "${o}")
            list(APPEND _abs_outputs "${o}")
        else()
            list(APPEND _abs_outputs "${TFK_WORKING_DIRECTORY}/${o}")
        endif()
    endforeach()

    # SOURCES → absolute paths too. These are deps; not declared as OUTPUTS.
    set(_abs_sources)
    foreach(s IN LISTS TFK_SOURCES)
        if(IS_ABSOLUTE "${s}")
            list(APPEND _abs_sources "${s}")
        else()
            list(APPEND _abs_sources "${TFK_WORKING_DIRECTORY}/${s}")
        endif()
    endforeach()

    # Pass TT_METAL_ROOT so build_kernels.sh hits the in-tree submodule
    # (otherwise it falls back to /home/kyamaguchi/tt-metal — exactly the
    # sibling-clone trap CLAUDE.md warns about). Caller can override via ENV.
    set(_env_args
        "TT_METAL_ROOT=${TT_METAL_ROOT}"
    )
    foreach(e IN LISTS TFK_ENV)
        list(APPEND _env_args "${e}")
    endforeach()

    # tt_foil_firmware is the custom target whose outputs are the
    # *_weakened.elf files build_kernels.sh links against. Adding it as a
    # DEPENDS makes kernel ELFs rebuild whenever firmware does.
    set(_fw_dep)
    if(TARGET tt_foil_firmware)
        set(_fw_dep tt_foil_firmware)
    endif()

    # kernel_build_helpers.sh is sourced by every build_kernels.sh; changes
    # to it (e.g. the manifest format) should retrigger every kernel build.
    set(_helper "${CMAKE_SOURCE_DIR}/scripts/kernel_build_helpers.sh")

    add_custom_command(
        OUTPUT  ${_abs_outputs}
        COMMAND ${CMAKE_COMMAND} -E env ${_env_args} bash "${TFK_SCRIPT}"
        DEPENDS ${_abs_sources}
                ${TFK_SCRIPT}
                ${_helper}
                ${_fw_dep}
        WORKING_DIRECTORY ${TFK_WORKING_DIRECTORY}
        COMMENT "Building kernels: ${TFK_NAME}"
        VERBATIM
    )

    add_custom_target(kernels_${TFK_NAME} ALL DEPENDS ${_abs_outputs})
endfunction()
