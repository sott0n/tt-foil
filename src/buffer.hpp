// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_foil/runtime.hpp"  // Buffer, BufferLocation, CoreCoord

namespace tt::foil {

struct Device;

// Internal allocation (called by runtime.hpp wrappers).
Buffer* buffer_alloc(Device& dev, BufferLocation loc, std::size_t size_bytes, CoreCoord logical_core);

// Internal free.
void buffer_free(Buffer* buf);

}  // namespace tt::foil
