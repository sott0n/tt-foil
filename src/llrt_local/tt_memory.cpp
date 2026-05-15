// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_memory.h"

#include <tt_stl/assert.hpp>
#include <algorithm>
#include <cstdint>
#include <span>

#include "tt_elffile.hpp"
// MetalContext include removed — XIP dump path patched below to skip the
// MetalContext::instance() call that would otherwise pull in tt-metal's full
// device-init machinery on first use.
#include <cstdlib>

namespace tt::foil::ll_api {

memory::memory() {
    constexpr uint32_t initial_data_space_ = 0x400;
    constexpr uint32_t initial_span_space_ = 4;

    data_.reserve(initial_data_space_);
    link_spans_.reserve(initial_span_space_);
}

memory::memory(const std::string& path, Loading loading) : loading_(loading) {
    ElfFile elf;

    elf.ReadImage(path);
    if (loading == Loading::CONTIGUOUS_XIP) {
        elf.MakeExecuteInPlace();

        // debug: dump disassembly after XIP transform (tt-triage helper).
        // Upstream tt_memory.cpp gates this with MetalContext::instance()
        // .rtoptions().get_disable_xip_dump(); we replaced that with a
        // plain env-var read so this TU stays free of MetalContext, whose
        // instance() call implicitly opens a UMD cluster and conflicts
        // with the one device_open() owns. Set TT_FOIL_XIP_DUMP=1 to
        // re-enable the dump (default: off, matching production runs).
        if (const char* env = std::getenv("TT_FOIL_XIP_DUMP"); env && *env != '0') {
            std::string out_elf_path = std::string(path) + ".xip.elf";
            try {
                elf.WriteImage(out_elf_path);
            } catch (const std::exception &e) {
                log_warning(tt::LogLLRuntime, "Failed to write XIP ELF for disassembly ({}): {}", out_elf_path, e.what());
            } catch (...) {
                log_warning(tt::LogLLRuntime, "Failed to write XIP ELF for disassembly: {}", out_elf_path);
            }
        }
    }

    auto const& segments = elf.GetSegments();

    // The ELF file puts the text segment first, but one set of
    // binaries (ncrisc) places data a lower address, and at least one
    // consumer (unknown) requires spans in address order, so generate
    // a mapping table.
    // TODO: Perhaps we can relax this?
    std::vector<unsigned> map;
    map.reserve(segments.size());
    for (unsigned ix = 0; ix != segments.size(); ix++) {
        map.push_back(ix);
    }
    if (loading == Loading::DISCRETE) {
        std::sort(
            map.begin(), map.end(), [&](unsigned a, unsigned b) { return segments[a].address < segments[b].address; });
    }

    link_spans_.reserve(segments.size());
    text_addr_ = segments[0].address;
    text_size_ = segments[0].contents.size() * sizeof(word_t);
    auto lma = segments[0].lma;

    for (unsigned ix : map) {
        auto const& segment = segments[map[ix]];
        if (not segment.relocs.empty()) {
            TT_THROW("{}: unexpected dynamic relocations", path);
        }
        if (loading != Loading::DISCRETE) {
            if (segment.lma != lma) {
                TT_THROW("{}: inconsistent load addresses for packing", path);
            }
            lma += segment.contents.size() * sizeof(word_t);
        }
        if (loading == Loading::DISCRETE ? !segment.contents.empty() : link_spans_.empty()) {
            link_spans_.emplace_back(segment.address, 0);
        }
        link_spans_.back().len += segment.contents.size();
        data_.insert(data_.end(), segment.contents.begin(), segment.contents.end());
    }
}

bool memory::operator==(const memory& other) const { return data_ == other.data_ && link_spans_ == other.link_spans_; }

void memory::fill_from_mem_template(
    const memory& mem_template,
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    link_spans_ = mem_template.link_spans_;
    data_.resize(mem_template.data_.size());
    process_spans(callback);
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::const_iterator cit = data_.cbegin() + offset;
        callback(cit, span.addr, span.len);
        offset += span.len;
    }
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::iterator it = data_.begin() + offset;
        callback(it, span.addr, span.len);
        offset += span.len;
    }
}

void memory::update_spans(std::function<void(uint64_t& addr)>& callback) {
    for (auto& span : link_spans_) {
        callback(span.addr);
    }
}

}  // namespace tt::foil::ll_api
