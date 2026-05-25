// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "buffer.hpp"
#include "device.hpp"
#include "profiling.hpp"

#include <stdexcept>

namespace tt::foil {

namespace {
constexpr const char* kPoolL1   = "Device L1";
constexpr const char* kPoolDram = "Device DRAM";

inline const char* pool_name(BufferLocation loc) {
    return loc == BufferLocation::L1 ? kPoolL1 : kPoolDram;
}

// Tracy tracks allocations by address inside each named pool. tt-foil's L1
// is per-core, so two cores can each allocate at the same device address —
// which would look like a double-alloc to Tracy. Encode the logical core in
// the upper 16 bits to keep keys unique within the "Device L1" pool.
inline uintptr_t tracy_key(BufferLocation loc, uint64_t dev_addr, CoreCoord core) {
    if (loc == BufferLocation::DRAM) return static_cast<uintptr_t>(dev_addr);
    return (static_cast<uintptr_t>(core.x) << 56)
         | (static_cast<uintptr_t>(core.y) << 48)
         | static_cast<uintptr_t>(dev_addr);
}
}  // namespace

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

    TF_ALLOC(tracy_key(loc, dev_addr, logical_core), size_bytes, pool_name(loc));
    return buf;
}

void buffer_free(Buffer* buf) {
    // Bump allocator: freeing is a no-op.
    // Callers use reset() on the allocator to reclaim all memory at once.
    if (buf != nullptr) {
        TF_FREE(tracy_key(buf->location, buf->device_addr, buf->core),
                pool_name(buf->location));
    }
    delete buf;
}

}  // namespace tt::foil
