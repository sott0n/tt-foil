// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device_profile.hpp"

#include "device.hpp"

#include "llrt/hal.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace tt::foil {

#if defined(TT_FOIL_DEVICE_PROFILER_ENABLED)

namespace {

// Per-RISC buffer layout (mirrors hostdevcommon/profiler_common.h ::BufferIndex
// and ::ControlBuffer enums). Each marker is two uint32_t words.
constexpr uint32_t kProfilerL1VectorSize = 512;
constexpr uint32_t kControlVectorSize = 32;
constexpr uint32_t kCustomMarkersOffset = 12;
constexpr uint32_t kGuaranteedMarkersStart = 4;
constexpr uint32_t kProcessorCount = 5;
constexpr uint32_t kDeviceBufferEndIndexBase = 5;

constexpr const char* kRiscNames[] = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
constexpr const char* kPacketTypes[] = {"START", "END", "TOTAL", "TS_DATA", "TS_EVENT", "TS_DATA_16B"};

// Singleton CSV stream. Opened on first capture, flushed at atexit.
struct CsvSink {
    std::mutex mu;
    std::FILE* fp = nullptr;
    bool tried_open = false;
    bool atexit_registered = false;

    void ensure_open() {
        if (tried_open) return;
        tried_open = true;
        const char* path = std::getenv("TT_FOIL_DEVICE_ZONES_CSV");
        if (path == nullptr || path[0] == '\0') {
            path = "tt_foil_device_zones.csv";
        }
        fp = std::fopen(path, "w");
        if (fp != nullptr) {
            std::fputs(
                "dispatch_idx,host_ns,core_x,core_y,risc,packet_type,zone_hash,cycle\n",
                fp);
            std::fprintf(stderr, "[device_profile] writing to %s\n", path);
        } else {
            std::fprintf(stderr, "[device_profile] cannot open %s\n", path);
        }
        if (!atexit_registered) {
            std::atexit([] { flush_device_profile_csv(); });
            atexit_registered = true;
        }
    }
};

CsvSink& sink() {
    static CsvSink s;
    return s;
}

// Clock-sync state — populated lazily on the first marker we see.
//
// Blackhole's wall-clock runs at 1 GHz, so cycle and ns are 1:1 in
// magnitude — we only need a single offset `b` such that
//     host_aligned_ns = cycle + b
// Recording (host_ns_first, cycle_first) at the first capture and using
// b = host_ns_first - cycle_first is approximate (host_ns is sampled
// *after* the chip already ran, so b includes ~few μs of post-dispatch
// PCIe latency) but precise enough to put device zones on the same wall
// timeline as host TF_ zones. Phase 4 sticks to this simple model;
// a future iteration can fit b via a dedicated calibration kernel.
struct ClockSync {
    std::mutex mu;
    bool calibrated = false;
    uint64_t host_ns_first = 0;
    uint64_t cycle_first = 0;
};
ClockSync& clock_sync() {
    static ClockSync c;
    return c;
}

void write_clock_sync_file() {
    const char* path = std::getenv("TT_FOIL_DEVICE_CLOCK_SYNC");
    if (path == nullptr || path[0] == '\0') {
        path = "tt_foil_device_clock_sync.csv";
    }
    std::FILE* fp = std::fopen(path, "w");
    if (fp == nullptr) return;
    std::fputs("host_ns_first,cycle_first,ns_per_cycle\n", fp);
    ClockSync& c = clock_sync();
    // Lock not strictly needed (writes happen post-mortem) but keeps
    // the contract.
    std::lock_guard<std::mutex> lock(c.mu);
    std::fprintf(fp, "%llu,%llu,1.0\n",
                 static_cast<unsigned long long>(c.host_ns_first),
                 static_cast<unsigned long long>(c.cycle_first));
    std::fclose(fp);
}

bool decode_marker(uint32_t w0, uint32_t w1,
                   uint16_t& zone_hash, uint8_t& packet_type, uint64_t& cycle) {
    if ((w0 & 0x80000000u) == 0) return false;
    // init_profiler() pre-fills guaranteed marker slots with 0x80000000
    // in BOTH H and L positions — filter that uninitialized sentinel.
    if (w0 == 0x80000000u) return false;
    const uint32_t timer_id = (w0 >> 12) & 0x7FFFF;
    zone_hash   = static_cast<uint16_t>(timer_id & 0xFFFF);
    packet_type = static_cast<uint8_t>((timer_id >> 16) & 0x7);
    cycle = (static_cast<uint64_t>(w0 & 0xFFF) << 32) | w1;
    return true;
}

}  // namespace

void capture_device_profile(Device& dev, std::span<Kernel* const> kernels) {
    CsvSink& s = sink();
    {
        std::lock_guard<std::mutex> lock(s.mu);
        s.ensure_open();
        if (s.fp == nullptr) return;
    }

    const tt::tt_metal::Hal& hal = *dev.hal;
    const uint64_t profiler_addr = hal.get_dev_addr(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::PROFILER);
    const uint64_t profiler_size = hal.get_dev_size(
        tt_metal::HalProgrammableCoreType::TENSIX,
        tt_metal::HalL1MemAddrType::PROFILER);
    if (profiler_size == 0) return;

    const uint64_t expected_words =
        kControlVectorSize + kProcessorCount * kProfilerL1VectorSize;
    if (profiler_size / sizeof(uint32_t) < expected_words) return;

    static std::atomic<uint32_t> dispatch_counter{0};
    const uint32_t dispatch_idx = dispatch_counter.fetch_add(1);
    const uint64_t host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    std::vector<uint32_t> buf(profiler_size / sizeof(uint32_t));

    std::lock_guard<std::mutex> lock(s.mu);
    for (Kernel* k : kernels) {
        tt::umd::CoreCoord cc{
            k->virt_x, k->virt_y,
            tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED};
        dev.umd_driver->read_from_device(
            buf.data(), dev.chip_id, cc, profiler_addr, profiler_size);

        for (uint32_t risc_id = 0; risc_id < kProcessorCount; ++risc_id) {
            const uint32_t base = kControlVectorSize + risc_id * kProfilerL1VectorSize;
            const uint32_t w_end_idx = buf[kDeviceBufferEndIndexBase + risc_id];
            const uint32_t end_off = std::max<uint32_t>(w_end_idx, kCustomMarkersOffset);
            const uint32_t safe_end = std::min<uint32_t>(end_off, kProfilerL1VectorSize);

            for (uint32_t off = kGuaranteedMarkersStart; off + 1 < safe_end; off += 2) {
                uint16_t zone_hash;
                uint8_t  pkt;
                uint64_t cycle;
                if (!decode_marker(buf[base + off], buf[base + off + 1],
                                   zone_hash, pkt, cycle)) {
                    continue;
                }
                // First marker we ever see anchors the clock-sync mapping.
                {
                    ClockSync& cs = clock_sync();
                    std::lock_guard<std::mutex> cs_lock(cs.mu);
                    if (!cs.calibrated) {
                        cs.host_ns_first = host_ns;
                        cs.cycle_first   = cycle;
                        cs.calibrated    = true;
                    } else if (cycle < cs.cycle_first) {
                        // A later RISC may have started earlier — keep the
                        // earliest cycle as the anchor for the cleanest
                        // offset.
                        cs.cycle_first = cycle;
                    }
                }
                const char* risc_name =
                    risc_id < 5 ? kRiscNames[risc_id] : "?";
                const char* pkt_name =
                    pkt < 6 ? kPacketTypes[pkt] : "?";
                std::fprintf(s.fp,
                             "%u,%llu,%u,%u,%s,%s,0x%04x,%llu\n",
                             dispatch_idx,
                             static_cast<unsigned long long>(host_ns),
                             static_cast<unsigned>(k->core.x),
                             static_cast<unsigned>(k->core.y),
                             risc_name, pkt_name,
                             static_cast<unsigned>(zone_hash),
                             static_cast<unsigned long long>(cycle));
            }
        }
    }
}

void flush_device_profile_csv() {
    CsvSink& s = sink();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.fp != nullptr) {
        std::fflush(s.fp);
        // Don't close the file here — atexit may fire while other code
        // is still alive; closing risks SIGSEGV in the destructor chain.
        // The OS will reclaim it on process exit.
    }
    if (clock_sync().calibrated) {
        write_clock_sync_file();
    }
}

#else  // TT_FOIL_DEVICE_PROFILER_ENABLED

void capture_device_profile(Device&, std::span<Kernel* const>) {}
void flush_device_profile_csv() {}

#endif

}  // namespace tt::foil
