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
    reports/tt_foil_perf_results.csv    -- Performance Report (zone aggregate)
    reports/tt_foil_perf_per_call.csv   -- per-call timeline (seq, ts, zone, dur, ctx)
    reports/tt_foil_memory_report.csv   -- Memory Report (per-pool stats)

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

    with mem_report.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for pool in sorted(set(list(alloc_count) + list(free_count))):
            leaked = sum(live[pool].values())
            w.writerow([pool, alloc_count[pool], free_count[pool],
                        total_bytes[pool], peak_live[pool], leaked])


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
    perf_report = reports / "tt_foil_perf_results.csv"
    per_call_csv = reports / "tt_foil_perf_per_call.csv"
    mem_report = reports / "tt_foil_memory_report.csv"

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
    generate_memory_report(memlog_csv, mem_report)

    print(f"[tt_foil_profile] Performance Report: {perf_report}")
    print(f"[tt_foil_profile] Per-call timeline:  {per_call_csv}")
    print(f"[tt_foil_profile] Memory Report:      {mem_report}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
