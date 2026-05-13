// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "buffer.hpp"
#include "device.hpp"

#include <stdexcept>

namespace tt::foil {

Buffer* buffer_alloc(Device& dev, BufferLocation loc, std::size_t size_bytes, CoreCoord logical_core) {
    if (size_bytes == 0) {
        throw std::runtime_error("tt-foil: buffer size must be > 0");
    }

    uint64_t dev_addr = 0;
    switch (loc) {
        case BufferLocation::L1: {
            L1Allocator& alloc = dev.l1_for_core(logical_core);
            dev_addr = alloc.alloc(size_bytes, /*alignment=*/16);
            break;
        }
        case BufferLocation::DRAM: {
            dev_addr = dev.dram_alloc.alloc(size_bytes, /*alignment=*/32);
            break;
        }
    }

    auto* buf        = new Buffer{};
    buf->location    = loc;
    buf->device_addr = dev_addr;
    buf->size_bytes  = size_bytes;
    buf->core        = logical_core;
    return buf;
}

void buffer_free(Buffer* buf) {
    // Bump allocator: freeing is a no-op.
    // Callers use reset() on the allocator to reclaim all memory at once.
    delete buf;
}

}  // namespace tt::foil
