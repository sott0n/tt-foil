#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate tt-foil Performance Report and Memory Report via Tracy capture.

Usage:
    python tools/tt_foil_profile.py [-o <out_dir>] [--] <binary> [args...]

Outputs (under <out_dir>, default: generated/profiler/):
    .logs/tracy_profile_log.tracy   -- raw Tracy capture (binary, Tracy GUI)
    .logs/tracy_ops_times.csv       -- per-zone timing (TF_ prefix filtered)
    .logs/tt_foil_memlog.csv        -- raw buffer alloc/free event log
    .logs/tt_foil_device_zones.csv  -- raw device-side zone events (if profiler-on)
    reports/tt_foil_perf_results.csv    -- host Performance Report (zone aggregate)
    reports/tt_foil_perf_per_call.csv   -- per-call timeline (seq, ts, zone, dur, ctx)
    reports/tt_foil_memory_report.csv   -- Memory Report (per-pool stats)
    reports/tt_foil_device_perf.csv     -- device-side Perf Report (per RISC × zone)

Requires:
    * tt-foil built with -DTT_FOIL_ENABLE_TRACY=ON
    * tt-metal's capture-release + csvexport-release binaries
      (auto-discovered under ${TT_METAL_BUILD_DIR}/tools/profiler/bin/)
"""

import argparse
import collections
import csv
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Tracy tool discovery
# ---------------------------------------------------------------------------

def find_tt_metal_build() -> pathlib.Path:
    env = os.environ.get("TT_METAL_BUILD_DIR")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    submodule = repo_root / "third_party" / "tt-metal" / "build_Release"
    if submodule.exists():
        return submodule
    sys.exit(
        "ERROR: TT_METAL_BUILD_DIR not set and submodule build not found.\n"
        "Build tt-metal with ENABLE_TRACY=ON or set TT_METAL_BUILD_DIR."
    )


def find_tools(build: pathlib.Path):
    bin_dir = build / "tools" / "profiler" / "bin"
    capture = bin_dir / "capture-release"
    csvexport = bin_dir / "csvexport-release"
    for b in (capture, csvexport):
        if not b.exists():
            sys.exit(
                f"ERROR: {b} not found.\n"
                "Rebuild tt-metal with ENABLE_TRACY=ON."
            )
    return capture, csvexport


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def detect_phases(ops_times_csv: pathlib.Path,
                  phases_csv: pathlib.Path,
                  gap_us: int = 1000) -> None:
    """Cluster TF_kernel_load events into Phase_N groups by inter-event gap.

    Many tt-foil workloads (e.g. ResNet-20 inference) load a fresh set of
    kernels at each phase boundary, then dispatch many times against them
    before loading the next set. Detecting these bursts gives us virtual
    phase zones for free without requiring the user to wrap their code in
    TT_FOIL_ZONE().

    Output CSV columns:
        phase, kernel_load_count, t_start_ns, t_end_ns, duration_ns,
        first_kernel, last_kernel

    `duration_ns` is the wall-time from the first kernel_load of the
    phase to the last dispatch_execute that runs against those kernels
    (i.e., end = next phase's first kernel_load, or end-of-trace).
    """
    headers = ["phase", "kernel_load_count", "t_start_ns", "t_end_ns",
               "duration_ns", "first_kernel", "last_kernel"]
    if not ops_times_csv.exists() or ops_times_csv.stat().st_size == 0:
        with phases_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(headers)
        return

    kls = []  # (ts_ns, context/kernel_name)
    last_ts = 0
    with ops_times_csv.open() as fh:
        for row in csv.DictReader(fh):
            try:
                ts = int(row.get("ns_since_start", 0))
            except ValueError:
                continue
            if ts > last_ts:
                last_ts = ts
            if row.get("name") == "TF_kernel_load":
                kls.append((ts, row.get("zone_text") or ""))
    kls.sort()

    # Cluster: gap > gap_us microseconds → new phase.
    gap_ns = gap_us * 1000
    clusters: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for ts, name in kls:
        if cur and ts - cur[-1][0] > gap_ns:
            clusters.append(cur)
            cur = []
        cur.append((ts, name))
    if cur:
        clusters.append(cur)

    with phases_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i, c in enumerate(clusters):
            t_start = c[0][0]
            # Phase ends when the next phase begins, or at the last
            # event seen in the trace.
            t_end = clusters[i + 1][0][0] if i + 1 < len(clusters) else last_ts
            first_k = c[0][1]
            last_k = c[-1][1]
            w.writerow([f"Phase_{i}", len(c), t_start, t_end,
                        t_end - t_start, first_k, last_k])


def generate_per_call_csv(ops_times_csv: pathlib.Path,
                          per_call_csv: pathlib.Path) -> None:
    """Emit a cleaned per-event CSV preserving temporal order.

    csvexport-release -u writes one row per zone instance with many columns
    (src_file, src_line, thread, ...). We project to the useful subset and
    sort by time so users can scan the timeline directly:

        seq, ts_ns, zone, duration_ns, context

    `context` is whatever `ZoneText(...)` attached at runtime (e.g., the
    kernel name in TF_dispatch_execute_multi after improvement (2)). Empty
    when the zone has no per-call context.
    """
    headers = ["seq", "ts_ns", "zone", "duration_ns", "context"]
    if not ops_times_csv.exists() or ops_times_csv.stat().st_size == 0:
        with per_call_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(headers)
        return

    rows = []
    with ops_times_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name") or row.get("zone_name") or ""
            if not name:
                continue
            try:
                ts = int(row.get("ns_since_start", 0))
                dur = int(row.get("exec_time_ns", 0))
            except ValueError:
                continue
            # zone_text is set via ZoneText() at runtime (e.g., kernel name).
            ctx = row.get("zone_text") or ""
            rows.append((ts, name, dur, ctx))

    rows.sort()
    with per_call_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i, (ts, name, dur, ctx) in enumerate(rows):
            w.writerow([i, ts, name, dur, ctx])


def generate_perf_report(ops_times_csv: pathlib.Path,
                         perf_report: pathlib.Path) -> None:
    """Convert csvexport-release -u output to a Performance Report.

    csvexport -u emits one row per zone instance with at least these columns:
        name, src_file, src_line, zone_text, ns_since_start, exec_time_ns, ...

    Aggregates by (zone_name, zone_text) tuple. When zone_text is non-empty
    (set via ZoneText() at runtime — e.g., the kernel name attached in
    dispatch_execute_multi), each context gets its own row so users can
    see "TF_dispatch_execute_multi(conv_3x3_l1): 8 calls, mean 530μs"
    instead of just the un-split aggregate.
    """
    headers = ["ZONE_NAME", "CONTEXT", "CALL_COUNT", "TOTAL_NS",
               "MEAN_NS", "MIN_NS", "MAX_NS"]
    if not ops_times_csv.exists() or ops_times_csv.stat().st_size == 0:
        print(f"  (no zone data in {ops_times_csv})")
        with perf_report.open("w", newline="") as fh:
            csv.writer(fh).writerow(headers)
        return

    agg: dict[tuple[str, str], list] = {}

    with ops_times_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name") or row.get("zone_name") or ""
            if not name:
                continue
            ctx = row.get("zone_text") or ""
            try:
                t = int(row.get("exec_time_ns", 0))
            except ValueError:
                t = 0
            key = (name, ctx)
            cur = agg.get(key)
            if cur is None:
                agg[key] = [1, t, t, t]
            else:
                cur[0] += 1
                cur[1] += t
                if t < cur[2]:
                    cur[2] = t
                if t > cur[3]:
                    cur[3] = t

    # Sort by zone, then by descending total_ns inside zone — hot context first.
    sorted_keys = sorted(agg.keys(), key=lambda k: (k[0], -agg[k][1]))
    with perf_report.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for key in sorted_keys:
            name, ctx = key
            count, total, mn, mx = agg[key]
            mean = total // count if count else 0
            w.writerow([name, ctx, count, total, mean, mn, mx])


def generate_memory_report(memlog_csv: pathlib.Path,
                           mem_report: pathlib.Path) -> None:
    """Aggregate the per-event memory log into a Memory Report.

    Input CSV (written by src/profiling.hpp):
        op,pool,addr,size,ts_ns
        ALLOC,Device L1,0x1b200,4,231779020
        FREE,Device L1,0x1b200,0,234029429
        ...

    Per-pool output: ALLOC_COUNT, FREE_COUNT, TOTAL_BYTES_ALLOCATED,
                     PEAK_LIVE_BYTES, LEAKED_BYTES.
    """
    headers = ["POOL", "ALLOC_COUNT", "FREE_COUNT",
               "TOTAL_BYTES_ALLOCATED", "PEAK_LIVE_BYTES", "LEAKED_BYTES"]

    if not memlog_csv.exists() or memlog_csv.stat().st_size == 0:
        with mem_report.open("w", newline="") as fh:
            csv.writer(fh).writerow(headers)
        return

    live: dict[str, dict[int, int]] = collections.defaultdict(dict)
    alloc_count: dict[str, int] = collections.defaultdict(int)
    free_count: dict[str, int] = collections.defaultdict(int)
    total_bytes: dict[str, int] = collections.defaultdict(int)
    peak_live: dict[str, int] = collections.defaultdict(int)
    live_bytes: dict[str, int] = collections.defaultdict(int)

    with memlog_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            op = row.get("op", "")
            pool = row.get("pool", "")
            addr_s = row.get("addr", "0")
            size_s = row.get("size", "0")
            try:
                addr = int(addr_s, 16) if addr_s.startswith("0x") else int(addr_s)
                size = int(size_s)
            except ValueError:
                continue

            if op == "ALLOC":
                live[pool][addr] = size
                alloc_count[pool] += 1
                total_bytes[pool] += size
                live_bytes[pool] += size
                if live_bytes[pool] > peak_live[pool]:
                    peak_live[pool] = live_bytes[pool]
            elif op == "FREE":
                free_count[pool] += 1
                sz = live[pool].pop(addr, 0)
                live_bytes[pool] -= sz
            elif op == "RESET":
                # Pool-wide reset: the underlying bump allocator rewound,
                # so anything still considered live for this pool is gone.
                # Count each forgotten entry as a free so the alloc/free
                # totals stay balanced.
                if pool in live:
                    free_count[pool] += len(live[pool])
                    live[pool].clear()
                live_bytes[pool] = 0

    with mem_report.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for pool in sorted(set(list(alloc_count) + list(free_count))):
            leaked = sum(live[pool].values())
            w.writerow([pool, alloc_count[pool], free_count[pool],
                        total_bytes[pool], peak_live[pool], leaked])


# ---------------------------------------------------------------------------
# Device-side zone aggregation
# ---------------------------------------------------------------------------
#
# src/device_profile.cpp writes a raw per-event CSV at
#   $TT_FOIL_DEVICE_ZONES_CSV  (default: tt_foil_device_zones.csv in cwd)
# with columns:
#   dispatch_idx,host_ns,core_x,core_y,risc,packet_type,zone_hash,cycle
#
# Each kernel-side `DeviceZoneScopedN("name")` emits one START + one END
# row. We pair them per (core, risc) using a stack (zones nest), compute
# the cycle duration, and aggregate by (risc, zone_hash).
#
# zone_hash is `kernel_profiler::Hash16_CT(name "," __FILE__ "," __LINE__
# ",KERNEL_PROFILER")` — a 16-bit folded FNV-1a. To put readable names in
# the report we scan kernel sources for `DeviceZoneScopedN("...")` calls
# and rehash with the matching (abs_path, line_no), building a hash → name
# table. Best-effort: if the file path used at compile time differs from
# what rglob() finds (e.g. via a symlink), the entry stays unnamed and
# shows as "0xABCD".


def compute_device_zone_hash(name: str, src_file: str, line: int) -> int:
    """Replicate kernel_profiler::Hash16_CT(name "," file "," line ",KERNEL_PROFILER")."""
    s = f"{name},{src_file},{line},KERNEL_PROFILER"
    h = 2166136261
    for c in s.encode("utf-8"):
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ ((h >> 16) & 0xFFFF)) & 0xFFFF


# Matches every DeviceZoneScopedN-family macro. Many of the events seen
# in a typical capture come from kernel_profiler's auto-zones in tt-metal
# firmware (DeviceZoneScopedMainN("BRISC-FW"), DeviceZoneScopedMainChildN
# ("TRISC-KERNEL"), etc.), so scan firmware sources as well as our own
# kernels.
_ZONE_PAT = re.compile(
    r'DeviceZoneScoped(?:MainN|MainChildN|SumN1|SumN2|N)\s*\(\s*"([^"]+)"\s*\)')


def discover_device_zone_names(search_dirs: list[pathlib.Path]) -> dict[int, str]:
    """Scan kernel + firmware sources for DeviceZoneScoped*("...") and pre-compute hashes."""
    table: dict[int, str] = {}
    seen_files: set[pathlib.Path] = set()
    exts = (".cpp", ".cc", ".c", ".h", ".hpp")
    for root in search_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in exts:
                continue
            rp = path.resolve()
            if rp in seen_files:
                continue
            seen_files.add(rp)
            try:
                lines = rp.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, text in enumerate(lines, 1):
                m = _ZONE_PAT.search(text)
                if not m:
                    continue
                name = m.group(1)
                h = compute_device_zone_hash(name, str(rp), line_no)
                table[h] = name
    return table


def generate_device_perf_report(zones_csv: pathlib.Path,
                                perf_csv: pathlib.Path,
                                name_table: dict[int, str]) -> None:
    """Pair START/END device events and aggregate by (risc, zone)."""
    headers = ["RISC", "ZONE", "ZONE_HASH", "CALL_COUNT",
               "TOTAL_CYCLES", "MEAN_CYCLES", "MIN_CYCLES", "MAX_CYCLES"]
    if not zones_csv.exists() or zones_csv.stat().st_size == 0:
        with perf_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(headers)
        return

    # (risc, hash) -> [durations]
    agg: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    # (core_x, core_y, risc) -> stack of (hash, start_cycle)
    stacks: dict[tuple[int, int, str], list[tuple[int, int]]] = collections.defaultdict(list)
    orphan_starts = 0
    orphan_ends = 0

    with zones_csv.open() as fh:
        for row in csv.DictReader(fh):
            try:
                cx = int(row["core_x"])
                cy = int(row["core_y"])
                risc = row["risc"]
                pkt = row["packet_type"]
                zh_s = row["zone_hash"]
                zh = int(zh_s, 16) if zh_s.startswith("0x") else int(zh_s)
                cyc = int(row["cycle"])
            except (KeyError, ValueError):
                continue
            key = (cx, cy, risc)
            if pkt == "START":
                stacks[key].append((zh, cyc))
            elif pkt == "END":
                stk = stacks[key]
                # Pop the matching same-hash entry (top of stack) — naive
                # but correct for cleanly-nested zones, which is all we
                # generate today.
                if stk and stk[-1][0] == zh:
                    start_cyc = stk.pop()[1]
                    agg[(risc, zh)].append(cyc - start_cyc)
                else:
                    orphan_ends += 1
            # TOTAL / TS_DATA / TS_EVENT etc. — not zone pairs, ignored.

    for stk in stacks.values():
        orphan_starts += len(stk)

    # Build the sortable rows.
    out_rows = []
    for (risc, zh), durs in agg.items():
        n = len(durs)
        total = sum(durs)
        mean = total // n if n else 0
        out_rows.append((risc, name_table.get(zh, ""), zh, n,
                         total, mean, min(durs), max(durs)))
    # Sort by total cycles desc — hot zones first.
    out_rows.sort(key=lambda r: -r[4])

    with perf_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for risc, name, zh, n, total, mean, mn, mx in out_rows:
            w.writerow([risc, name, f"0x{zh:04x}", n, total, mean, mn, mx])

    if orphan_starts or orphan_ends:
        print(f"[tt_foil_profile] device_perf: {orphan_starts} unmatched START, "
              f"{orphan_ends} unmatched END (truncated trace?)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a tt-foil binary under Tracy and emit Performance + Memory reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-o", "--output-folder", default="generated/profiler",
                    help="output directory (default: %(default)s)")
    ap.add_argument("--zone-filter", default="",
                    help="substring filter for zone names; empty = include all "
                         "(default: empty, capturing both tt-foil's TF_* zones "
                         "and any user-set TT_FOIL_ZONE() zones)")
    ap.add_argument("--capture-port", type=int, default=8086,
                    help="Tracy capture server port (default: %(default)s)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="show capture-release output")
    ap.add_argument("--kernel-source-dirs", default="examples,models,src",
                    help="comma-separated dirs (relative to repo root) to scan for "
                         "DeviceZoneScopedN(\"...\") so the device perf report can "
                         "resolve hashes to names (default: %(default)s)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="binary and args to profile (use -- to separate)")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command provided")

    build = find_tt_metal_build()
    capture, csvexport = find_tools(build)

    out = pathlib.Path(args.output_folder).resolve()
    logs = out / ".logs"
    reports = out / "reports"
    logs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    tracy_file = logs / "tracy_profile_log.tracy"
    ops_times_csv = logs / "tracy_ops_times.csv"
    memlog_csv = logs / "tt_foil_memlog.csv"
    device_zones_csv = logs / "tt_foil_device_zones.csv"
    device_clock_sync_csv = logs / "tt_foil_device_clock_sync.csv"
    perf_report = reports / "tt_foil_perf_results.csv"
    per_call_csv = reports / "tt_foil_perf_per_call.csv"
    phases_csv = reports / "tt_foil_phases.csv"
    mem_report = reports / "tt_foil_memory_report.csv"
    device_perf_report = reports / "tt_foil_device_perf.csv"

    # Remove stale capture; capture-release will not overwrite by default.
    if tracy_file.exists():
        tracy_file.unlink()

    # 1. Start Tracy capture server in the background.
    print(f"[tt_foil_profile] starting capture → {tracy_file}")
    cap_args = [str(capture), "-o", str(tracy_file),
                "-p", str(args.capture_port), "-f"]
    cap = subprocess.Popen(
        cap_args,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=None if args.verbose else subprocess.DEVNULL,
    )
    # Give the server a moment to bind its socket before the client connects.
    time.sleep(0.5)

    # 2. Run target binary. Tracy client connects automatically on first zone.
    # Point the in-process memory logger (src/profiling.hpp) at our log dir.
    env = os.environ.copy()
    env["TT_FOIL_MEM_LOG"] = str(memlog_csv)
    env["TT_FOIL_DEVICE_ZONES_CSV"] = str(device_zones_csv)
    env["TT_FOIL_DEVICE_CLOCK_SYNC"] = str(device_clock_sync_csv)
    print(f"[tt_foil_profile] running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, env=env)
    finally:
        # 3. Stop capture server. capture-release flushes the .tracy file
        # to disk on SIGINT (its "Save & Quit" path); SIGTERM kills it
        # before the save completes. Give it a moment after the client
        # exits so the final events propagate over the socket.
        time.sleep(0.5)
        cap.send_signal(signal.SIGINT)
        try:
            cap.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cap.kill()

    # 4. Wait for the .tracy file to materialise (capture flush is async).
    for _ in range(30):
        if tracy_file.exists() and tracy_file.stat().st_size > 0:
            break
        time.sleep(0.5)
    else:
        print("ERROR: Tracy capture file not produced — "
              "is the binary built with TT_FOIL_ENABLE_TRACY=ON?",
              file=sys.stderr)
        return proc.returncode or 1

    # 5. Export CSVs from the capture (zones only — memory comes from the
    # direct CSV log written by src/profiling.hpp::emit_mem_event).
    print(f"[tt_foil_profile] exporting CSVs")
    cmd_export = [str(csvexport), "-u", str(tracy_file)]
    if args.zone_filter:
        cmd_export[2:2] = ["-f", args.zone_filter]
    with ops_times_csv.open("w") as fh:
        subprocess.run(cmd_export, stdout=fh, check=False)

    # 6. Post-process into final reports.
    generate_perf_report(ops_times_csv, perf_report)
    generate_per_call_csv(ops_times_csv, per_call_csv)
    detect_phases(ops_times_csv, phases_csv)
    generate_memory_report(memlog_csv, mem_report)

    # 7. Device-side aggregation (only when the C++ runtime was built
    # with TT_FOIL_DEVICE_PROFILER=ON and produced a zones CSV).
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    kdirs = [repo_root / d.strip()
             for d in args.kernel_source_dirs.split(",") if d.strip()]
    # Auto-include tt-metal firmware sources — kernel_profiler.hpp's
    # auto-zones (BRISC-FW, NCRISC-KERNEL, TRISC-KERNEL, ...) come from
    # there. Hashes use the absolute path that the kernel-build script
    # passes via -c, so we resolve TT_METAL_ROOT the same way.
    tt_metal_root = pathlib.Path(
        os.environ.get("TT_METAL_ROOT")
        or (repo_root / "third_party" / "tt-metal"))
    if tt_metal_root.exists():
        kdirs.append(tt_metal_root / "tt_metal" / "hw" / "firmware" / "src" / "tt-1xx")
    name_table = discover_device_zone_names(kdirs)
    generate_device_perf_report(device_zones_csv, device_perf_report, name_table)

    print(f"[tt_foil_profile] Performance Report:  {perf_report}")
    print(f"[tt_foil_profile] Per-call timeline:   {per_call_csv}")
    print(f"[tt_foil_profile] Phases (auto):       {phases_csv}")
    print(f"[tt_foil_profile] Memory Report:       {mem_report}")
    if device_zones_csv.exists():
        print(f"[tt_foil_profile] Device Perf Report:  {device_perf_report}  "
              f"({len(name_table)} zone names known)")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
