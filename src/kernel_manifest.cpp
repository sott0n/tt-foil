// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kernel_manifest.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::foil {
namespace {

// ---------------------------------------------------------------------------
// SHA-256 (public-domain implementation, FIPS 180-4). Sized for our use case:
// we hash at most ~1 MB across all kernel_load() calls per process, so a
// straightforward block-at-a-time implementation is fine.
// ---------------------------------------------------------------------------
class Sha256 {
    static constexpr std::array<uint32_t, 64> k{{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    }};

    std::array<uint32_t, 8> h{{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    }};
    std::array<uint8_t, 64> buf{};
    uint64_t len_bits = 0;
    size_t   buf_len = 0;

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void process_block(const uint8_t* p) {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i*4]) << 24) | (uint32_t(p[i*4+1]) << 16) |
                   (uint32_t(p[i*4+2]) << 8) |  uint32_t(p[i*4+3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

public:
    void update(const uint8_t* data, size_t n) {
        len_bits += static_cast<uint64_t>(n) * 8;
        while (n > 0) {
            size_t take = std::min<size_t>(64 - buf_len, n);
            std::memcpy(buf.data() + buf_len, data, take);
            buf_len += take; data += take; n -= take;
            if (buf_len == 64) { process_block(buf.data()); buf_len = 0; }
        }
    }

    std::string hex_digest() {
        buf[buf_len++] = 0x80;
        if (buf_len > 56) {
            while (buf_len < 64) buf[buf_len++] = 0;
            process_block(buf.data()); buf_len = 0;
        }
        while (buf_len < 56) buf[buf_len++] = 0;
        for (int i = 7; i >= 0; --i) buf[buf_len++] = (len_bits >> (i*8)) & 0xff;
        process_block(buf.data());

        std::string out; out.reserve(64);
        static const char* hex = "0123456789abcdef";
        for (uint32_t v : h) {
            for (int i = 3; i >= 0; --i) {
                uint8_t byte = (v >> (i*8)) & 0xff;
                out.push_back(hex[byte >> 4]);
                out.push_back(hex[byte & 0xf]);
            }
        }
        return out;
    }
};

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("kernel_manifest: cannot open " + path);
    Sha256 h;
    std::array<uint8_t, 64 * 1024> buf{};
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        std::streamsize n = f.gcount();
        if (n > 0) h.update(buf.data(), static_cast<size_t>(n));
    }
    return h.hex_digest();
}

// Per-process firmware-hash cache. The firmware ELFs don't change within a
// process, so we hash each one once.
std::string cached_fw_hash(const std::string& path) {
    static std::mutex                                  mu;
    static std::unordered_map<std::string, std::string> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    std::string h = sha256_file(path);
    cache.emplace(path, h);
    return h;
}

// Manifest parsing — flat key=value, prefixes "fw:", "src:", "elf:", "env:"
// for the namespaced sections (see write_kernel_manifest in
// scripts/kernel_build_helpers.sh).
struct Manifest {
    std::string firmware_dir;
    std::unordered_map<std::string, std::string> fw;   // basename → sha256
    std::unordered_map<std::string, std::string> elf;  // basename → sha256
};

Manifest parse_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("kernel_manifest: cannot open " + path);
    Manifest m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "firmware_dir") {
            m.firmware_dir = val;
        } else if (key.rfind("fw:", 0) == 0) {
            m.fw.emplace(key.substr(3), val);
        } else if (key.rfind("elf:", 0) == 0) {
            m.elf.emplace(key.substr(4), val);
        }
        // src: and env: entries are recorded for audit but not checked at
        // runtime — the firmware + ELF hashes are sufficient to catch the
        // "ELF doesn't match the firmware loaded onto the chip" case, which
        // is the bug class we actually hit in practice. Source-newer-than-
        // ELF is caught by the CMake dependency graph at build time.
    }
    return m;
}

// Map manifest fw:<basename> entries to absolute paths under the runtime's
// resolved firmware root. The manifest always records *_weakened.elf since
// that's what build_kernels.sh links against (--just-symbols=…). Note that
// FirmwarePaths::trisc{0,1,2} points to the *non-weakened* trisc?.elf used
// for chip-side firmware load, so we can't just return those fields.
std::string firmware_path_for(const FirmwarePaths& fw, const std::string& basename) {
    static const std::array<const char*, 5> kRiscs = {
        "brisc", "ncrisc", "trisc0", "trisc1", "trisc2"
    };
    for (const char* risc : kRiscs) {
        if (basename == std::string{risc} + "_weakened.elf") {
            return fw.root + "/" + risc + "/" + basename;
        }
    }
    return {};
}

bool verify_enabled() {
    const char* e = std::getenv("TT_FOIL_VERIFY_MANIFEST");
    return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
}

bool bypass_enabled() {
    const char* e = std::getenv("TT_FOIL_SKIP_MANIFEST_CHECK");
    return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
}

}  // namespace

void check_kernel_manifest(
    std::span<const RiscBinary> binaries,
    const FirmwarePaths& fw_paths)
{
    if (!verify_enabled() || bypass_enabled() || binaries.empty()) return;

    // Memoise per kernel directory — within one process we only need to
    // verify a given prebuilt/ once.
    static std::mutex                       mu;
    static std::unordered_set<std::string>  verified;

    namespace fs = std::filesystem;
    fs::path first_elf(binaries[0].elf_path);
    fs::path kernel_dir = first_elf.parent_path();
    std::string kernel_dir_str = kernel_dir.string();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (verified.count(kernel_dir_str)) return;
    }

    fs::path manifest_path = kernel_dir / "manifest.txt";
    if (!fs::exists(manifest_path)) {
        std::cerr << "[tt-foil] WARNING: " << kernel_dir_str
                  << " has no manifest.txt — cannot verify ELF freshness. "
                     "Rebuild kernels with the current build_kernels.sh "
                     "to generate one.\n";
        std::lock_guard<std::mutex> lk(mu);
        verified.insert(kernel_dir_str);
        return;
    }

    Manifest m = parse_manifest(manifest_path.string());

    // 1. Firmware *_weakened.elf hashes must match the firmware actually
    //    resolved for this process — otherwise the ELFs were linked against
    //    a different firmware tree than the one running on the chip.
    for (const auto& [basename, expected] : m.fw) {
        std::string fw_elf = firmware_path_for(fw_paths, basename);
        if (fw_elf.empty() || !std::filesystem::exists(fw_elf)) continue;
        std::string actual = cached_fw_hash(fw_elf);
        if (actual != expected) {
            std::ostringstream oss;
            oss << "tt-foil: kernel ELFs in " << kernel_dir_str
                << " were linked against a different firmware than the one "
                << "loaded on the chip.\n"
                << "  firmware file:     " << fw_elf << "\n"
                << "  manifest expected: " << expected << "\n"
                << "  actual on disk:    " << actual << "\n"
                << "Rebuild the kernels (e.g. rerun build_kernels.sh) or set\n"
                << "TT_FOIL_SKIP_MANIFEST_CHECK=1 to bypass.";
            throw std::runtime_error(oss.str());
        }
    }

    // 2. ELF files on disk must match the manifest hashes (catches a
    //    half-finished rebuild or someone overwriting a single ELF by hand).
    for (const auto& rb : binaries) {
        std::string basename = fs::path(rb.elf_path).filename().string();
        auto it = m.elf.find(basename);
        if (it == m.elf.end()) continue;  // not tracked; ignore
        std::string actual = sha256_file(rb.elf_path);
        if (actual != it->second) {
            std::ostringstream oss;
            oss << "tt-foil: kernel ELF " << rb.elf_path
                << " does not match manifest.\n"
                << "  manifest expected: " << it->second << "\n"
                << "  actual on disk:    " << actual << "\n"
                << "Rebuild the kernels or set TT_FOIL_SKIP_MANIFEST_CHECK=1.";
            throw std::runtime_error(oss.str());
        }
    }

    std::lock_guard<std::mutex> lk(mu);
    verified.insert(kernel_dir_str);
}

}  // namespace tt::foil
