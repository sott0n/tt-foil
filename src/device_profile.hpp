// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-side cycle profiler — host-side capture, parse, and CSV emit.
//
// After a slow-dispatch run, each Tensix RISC has written cycle markers
// into its slot of the profiler_msg_t mailbox region. This module reads
// the bytes back, decodes the (timer_id, cycle) pairs, and writes them
// directly to a CSV file (no in-memory buffer — keeps the implementation
// simple and avoids the libstdc++ vector quirks observed during bring-up).
//
// All entry points are no-ops when TT_FOIL_DEVICE_PROFILER_ENABLED is not
// defined at compile time.

#pragma once

#include <span>

#include "kernel.hpp"

namespace tt::foil {

struct Device;

// Read each kernel's core profiler region and append parsed events to
// the CSV at $TT_FOIL_DEVICE_ZONES_CSV (default: tt_foil_device_zones.csv
// in cwd). Called once per dispatch from dispatch_execute_multi.
void capture_device_profile(Device& dev, std::span<Kernel* const> kernels);

// Flush + close the CSV stream. Wired to atexit() automatically on the
// first capture; can also be called explicitly to ensure the file is
// fully flushed mid-process (e.g., before forking).
void flush_device_profile_csv();

}  // namespace tt::foil
