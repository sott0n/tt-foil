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
    reports/tt_foil_perf_results.csv    -- Performance Report (zones)
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

def generate_perf_report(ops_times_csv: pathlib.Path,
                         perf_report: pathlib.Path) -> None:
    """Convert csvexport-release -u output to a Performance Report.

    csvexport -u emits one row per zone instance with at least these columns:
        name, src_file, src_line, total_ns, ...

    We aggregate by name: call_count, total_ns, mean_ns, min_ns, max_ns.
    """
    if not ops_times_csv.exists() or ops_times_csv.stat().st_size == 0:
        print(f"  (no zone data in {ops_times_csv})")
        # Still create an empty report so downstream tooling sees the file.
        with perf_report.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["ZONE_NAME", "CALL_COUNT", "TOTAL_NS",
                 "MEAN_NS", "MIN_NS", "MAX_NS"])
        return

    Stats = collections.namedtuple("Stats", "count total mn mx")
    agg: dict[str, list] = {}

    with ops_times_csv.open() as fh:
        reader = csv.DictReader(fh)
        time_field = None
        if reader.fieldnames:
            # csvexport -u column for per-event time is typically "ns_since_start"
            # paired with "exec_time_ns" (the duration). Fall back to common
            # alternatives so this works across Tracy versions.
            for candidate in ("exec_time_ns", "ns", "time_ns", "duration"):
                if candidate in reader.fieldnames:
                    time_field = candidate
                    break
        for row in reader:
            name = row.get("name") or row.get("zone_name") or ""
            if not name:
                continue
            try:
                t = int(row.get(time_field, 0)) if time_field else 0
            except ValueError:
                t = 0
            cur = agg.get(name)
            if cur is None:
                agg[name] = [1, t, t, t]
            else:
                cur[0] += 1
                cur[1] += t
                if t < cur[2]:
                    cur[2] = t
                if t > cur[3]:
                    cur[3] = t

    with perf_report.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ZONE_NAME", "CALL_COUNT", "TOTAL_NS",
                    "MEAN_NS", "MIN_NS", "MAX_NS"])
        for name in sorted(agg.keys()):
            count, total, mn, mx = agg[name]
            mean = total // count if count else 0
            w.writerow([name, count, total, mean, mn, mx])


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
    ap.add_argument("--zone-prefix", default="TF_",
                    help="zone-name prefix to filter (default: %(default)s)")
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
    with ops_times_csv.open("w") as fh:
        subprocess.run(
            [str(csvexport), "-u", "-f", args.zone_prefix, str(tracy_file)],
            stdout=fh, check=False)

    # 6. Post-process into final reports.
    generate_perf_report(ops_times_csv, perf_report)
    generate_memory_report(memlog_csv, mem_report)

    print(f"[tt_foil_profile] Performance Report: {perf_report}")
    print(f"[tt_foil_profile] Memory Report:      {mem_report}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
