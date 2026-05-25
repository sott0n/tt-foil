// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "buffer.hpp"
#include "device.hpp"
#include "profiling.hpp"

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

    // Per-core pool naming for L1 (separate pool per Tensix), single pool
    // for DRAM (chip-global address space). Each pool has its own address
    // space in Tracy so the raw device_addr is fine as the key.
    if (loc == BufferLocation::L1) {
        TF_ALLOC_CORE(dev_addr, size_bytes, "Device L1", logical_core.x, logical_core.y);
    } else {
        TF_ALLOC(dev_addr, size_bytes, "Device DRAM");
    }
    return buf;
}

void buffer_free(Buffer* buf) {
    // Bump allocator: freeing is a no-op.
    // Callers use reset() on the allocator to reclaim all memory at once.
    if (buf != nullptr) {
        if (buf->location == BufferLocation::L1) {
            TF_FREE_CORE(buf->device_addr, "Device L1", buf->core.x, buf->core.y);
        } else {
            TF_FREE(buf->device_addr, "Device DRAM");
        }
    }
    delete buf;
}

}  // namespace tt::foil
