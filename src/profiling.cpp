// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Out-of-line helpers for tt-foil profiling. Only compiled when TRACY_ENABLE
// is defined (the file's contents are #if-gated below).
//
// Pool-name interning: Tracy's TracyAllocN takes a const char* and stores
// the pointer rather than copying the string. Pool names that include
// runtime values (core coordinates) need stable backing storage; we keep a
// small process-lifetime table of interned strings here.

#include "profiling.hpp"

#if defined(TRACY_ENABLE)

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tt::foil::profiling {

const char* core_pool_name(const char* base, int x, int y) {
    static std::mutex mu;
    // Heap-allocated strings live for the lifetime of the process — that is
    // intentional: Tracy holds the pointer indefinitely.
    static std::unordered_map<std::string, std::unique_ptr<char[]>> table;

    std::string key;
    key.reserve(std::strlen(base) + 12);
    key.append(base).append(" (");
    key.append(std::to_string(x)).append(",");
    key.append(std::to_string(y)).append(")");

    std::lock_guard<std::mutex> lock(mu);
    auto it = table.find(key);
    if (it != table.end()) {
        return it->second.get();
    }
    auto buf = std::make_unique<char[]>(key.size() + 1);
    std::memcpy(buf.get(), key.data(), key.size() + 1);
    const char* result = buf.get();
    table.emplace(std::move(key), std::move(buf));
    return result;
}

}  // namespace tt::foil::profiling

#endif  // TRACY_ENABLE
