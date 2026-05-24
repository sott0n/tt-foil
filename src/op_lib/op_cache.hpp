// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Shape-keyed op cache. Memoizes the expensive parts of a make_X factory
// (5-RISC ELF parse + L1/CB allocation + DRAM constant upload) so that
// the 2nd+ call with the same (op, shape, core) becomes a pure RTA
// refresh. Built on top of pin_persistent so cached ops survive the
// surrounding release_kernels/reset_l1 of transient ops.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "tt_foil/runtime.hpp"

namespace tt::foil::op_lib {

struct ShapeKey {
    std::string_view        op_name;     // pointer-equality on string literal
    std::array<uint32_t, 8> params{};    // op-specific shape ints + bit-cast floats
    uint64_t                core_key{};  // (x << 32) | y, packed CoreCoord

    bool operator==(const ShapeKey& o) const noexcept {
        return op_name.data() == o.op_name.data()
            && op_name.size() == o.op_name.size()
            && params == o.params
            && core_key == o.core_key;
    }
};

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey& k) const noexcept {
        std::size_t h = reinterpret_cast<std::uintptr_t>(k.op_name.data());
        for (uint32_t p : k.params) h = h * 1315423911u + p;
        h = h * 1315423911u + static_cast<uint32_t>(k.core_key);
        h = h * 1315423911u + static_cast<uint32_t>(k.core_key >> 32);
        return h;
    }
};

inline uint64_t core_key_of(CoreCoord c) noexcept {
    return (uint64_t(c.x) << 32) | uint64_t(c.y);
}

// OpCache<H> stores std::unique_ptr<H> so the cached handle's address is
// stable across rehashes. Lookup is keyed by ShapeKey. On first lookup
// the factory runs (full make_X impl), the resulting H is pinned via
// pin_persistent so its kernel + L1 watermark survive
// release_kernels/reset_l1, and a reference into the map is returned.
template <typename OpHandle>
class OpCache {
public:
    template <typename Factory>
    OpHandle& get_or_create(Device&         dev,
                            CoreCoord       pin_core,
                            const ShapeKey& key,
                            Factory&&       factory) {
        auto it = map_.find(key);
        if (it != map_.end()) return *it->second;
        auto handle = std::make_unique<OpHandle>(std::forward<Factory>(factory)());
        tt::foil::pin_persistent(dev, *handle->kernel, pin_core);
        OpHandle* raw = handle.get();
        map_.emplace(key, std::move(handle));
        return *raw;
    }

    bool empty() const noexcept { return map_.empty(); }
    std::size_t size() const noexcept { return map_.size(); }

private:
    std::unordered_map<ShapeKey, std::unique_ptr<OpHandle>, ShapeKeyHash> map_;
};

}  // namespace tt::foil::op_lib
