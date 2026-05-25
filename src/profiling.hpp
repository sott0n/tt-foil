// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Tracy profiling macros for tt-foil.
//
// All macros are zero-overhead no-ops when TRACY_ENABLE is not defined at
// compile time (TT_FOIL_ENABLE_TRACY=OFF), so this header can be included
// unconditionally. Include only from .cpp files — never from public headers.
//
// Naming convention: zone names are prefixed with "TF_" so csvexport-release
// can filter to just tt-foil zones via `-p TF_` (same scheme as tt-metal's
// "TT_" prefix).

#pragma once

#if defined(TRACY_ENABLE)
#  include <tracy/Tracy.hpp>
#  include <cstdint>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <mutex>

namespace tt::foil::profiling {

// Direct CSV log of buffer alloc/free events for the Memory Report.
//
// Tracy's TracyMessage queue silently drops events under sustained load
// (verified empirically: qwen3 inference emits >5k allocs and zero made
// it into the trace), so we bypass Tracy for high-volume structured data
// and write to a dedicated CSV instead. TracyAllocN/FreeN still fires
// for the GUI's Memory tab — only the CSV-export path is replaced.
//
// File path: env var TT_FOIL_MEM_LOG, default "tt_foil_memlog.csv" in cwd.
// Format:   op,pool,addr_hex,size_bytes,ts_ns
inline void emit_mem_event(const char* op, const char* pool,
                           std::uintptr_t addr, std::size_t size) {
    static std::mutex mu;
    static std::FILE* fp = nullptr;
    static bool tried_open = false;

    std::lock_guard<std::mutex> lock(mu);
    if (!tried_open) {
        tried_open = true;
        const char* path = std::getenv("TT_FOIL_MEM_LOG");
        // Memory logging is opt-in: only enabled when TT_FOIL_MEM_LOG points
        // to a writable path. Leave it unset (or "OFF") to skip the mutex +
        // fprintf overhead — useful when measuring zone timing in isolation.
        if (path == nullptr || path[0] == '\0' ||
            std::strcmp(path, "OFF") == 0 || std::strcmp(path, "off") == 0) {
            return;
        }
        fp = std::fopen(path, "w");
        if (fp != nullptr) {
            std::fputs("op,pool,addr,size,ts_ns\n", fp);
        }
    }
    if (fp == nullptr) return;

    // Monotonic timestamp from Tracy's clock (matches zone timestamps).
    const int64_t ts = tracy::Profiler::GetTime();
    std::fprintf(fp, "%s,%s,0x%lx,%zu,%lld\n",
                 op, pool,
                 static_cast<unsigned long>(addr),
                 size,
                 static_cast<long long>(ts));
    // Flush is omitted on the hot path — atexit + line buffering keep the
    // file usable. If the process crashes mid-run, the tail is lost.
}

}  // namespace tt::foil::profiling

// CPU timeline zones (→ Performance Report).
#  define TF_ZONE_N(name)          ZoneScopedN(name)
#  define TF_FRAME_MARK()          FrameMark
// Attach a runtime-computed context string to the enclosing zone.
// Aggregator splits stats by (zone_name, zone_text) — used to tag a
// dispatch with the kernel name so per-kernel breakdowns appear in the
// Performance Report. `text` is a const char*; lifetime only needs to
// span the macro call (Tracy copies it).
#  define TF_ZONE_TEXT(text, len)  ZoneText((text), (len))
// Buffer alloc/free tracking (→ Memory Report).
// `pool` is a const char* identifying the memory pool (e.g. "Device L1").
// We emit BOTH the native Tracy alloc event (visible in Tracy GUI's
// Memory tab) and a CSV log entry (read by tt_foil_profile.py).
#  define TF_ALLOC(ptr, sz, pool) do { \
        TracyAllocN(reinterpret_cast<void*>(ptr), (sz), (pool)); \
        ::tt::foil::profiling::emit_mem_event( \
            "ALLOC", (pool), static_cast<std::uintptr_t>(ptr), (sz)); \
    } while (0)
#  define TF_FREE(ptr, pool) do { \
        TracyFreeN(reinterpret_cast<void*>(ptr), (pool)); \
        ::tt::foil::profiling::emit_mem_event( \
            "FREE", (pool), static_cast<std::uintptr_t>(ptr), 0); \
    } while (0)
#else
#  define TF_ZONE_N(name)          do {} while (0)
#  define TF_FRAME_MARK()          do {} while (0)
#  define TF_ZONE_TEXT(text, len)  do {} while (0)
#  define TF_ALLOC(ptr, sz, pool)  do {} while (0)
#  define TF_FREE(ptr, pool)       do {} while (0)
#endif
