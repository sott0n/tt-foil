// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include "tt_foil/runtime.hpp"  // BufferLocation, CoreCoord

namespace tt::foil {

struct Device;

struct Buffer {
    BufferLocation location;
    uint64_t       device_addr;  // NOC-visible address on the device
    std::size_t    size_bytes;

    // For L1 buffers: which logical core owns this buffer.
    CoreCoord      core;  // ignored for DRAM
};

// Internal allocation (called by runtime.hpp wrappers).
Buffer* buffer_alloc(Device& dev, BufferLocation loc, std::size_t size_bytes, CoreCoord logical_core);

// Internal free.
void buffer_free(Buffer* buf);

}  // namespace tt::foil
