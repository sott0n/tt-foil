// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Public profiling macros for tt-foil users.
//
// Lets test / benchmark / model code mark its own scopes so they show up in
// the Performance Report alongside tt-foil's internal TF_* zones. All
// macros are zero-overhead no-ops when tt-foil was built without
// TT_FOIL_ENABLE_TRACY (in which case TRACY_ENABLE is not propagated to
// the consumer's compile line).
//
// Usage:
//
//     #include <tt_foil/profiling.h>
//
//     void run_phase_a() {
//         TT_FOIL_ZONE("Phase A: stem + layer1");
//         // ... ops ...
//     }
//
//     for (int i = 0; i < N; ++i) {
//         TT_FOIL_ZONE("decode_step");
//         TT_FOIL_ZONE_TEXT(("token " + std::to_string(i)).c_str());
//         // ...
//     }
//
// Aggregation in the Performance Report splits stats by (zone_name,
// zone_text), so attaching text via TT_FOIL_ZONE_TEXT gives a per-context
// row (e.g., one row per token).

#pragma once

#if defined(TRACY_ENABLE)
#  include <tracy/Tracy.hpp>
#  include <cstring>
// RAII scope macro: declares a Tracy zone that ends when the surrounding
// scope exits. `name` must be a string literal (Tracy stores the pointer).
#  define TT_FOIL_ZONE(name)             ZoneScopedN(name)
// Attach a runtime-computed context string to the enclosing zone.
// `text` is a const char*; Tracy copies it.
#  define TT_FOIL_ZONE_TEXT(text)        ZoneText((text), std::strlen(text))
// Frame boundary marker — useful for marking iteration boundaries in
// inference loops so Tracy's GUI can split the timeline by frame.
#  define TT_FOIL_FRAME_MARK()           FrameMark
#else
#  define TT_FOIL_ZONE(name)             do {} while (0)
#  define TT_FOIL_ZONE_TEXT(text)        do {} while (0)
#  define TT_FOIL_FRAME_MARK()           do {} while (0)
#endif
