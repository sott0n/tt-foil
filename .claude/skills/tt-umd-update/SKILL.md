---
name: tt-umd-update
description: Check tt-umd releases against what tt-foil currently ships, decide whether anything is worth pulling in, and update tt-foil — by bumping the tt-metal submodule (which transitively bumps tt-umd), then adapting tt-foil's call sites. Use when the user asks to "check tt-umd releases", "update tt-umd", "bump umd / tt-metal", "新しい tt-umd を取り込む", "tt-umd を update", or wants to know if a newer UMD has anything tt-foil should adopt. Produces an impact report + a submodule bump + code adaptation; it does NOT run the ~30 min tt-metal rebuild or the HW ctest sweep (it hands those off with exact commands).
---

# tt-umd update for tt-foil

Check the tt-umd releases tt-foil transitively depends on, decide what is worth
pulling in, and apply the update — **without** taking on the maintenance burden
of an independently-pinned UMD.

## Mental model (read first — it determines the whole flow)

tt-foil does **not** pin tt-umd directly. The dependency chain is:

```
tt-foil  ──submodule──▶  third_party/tt-metal  ──submodule──▶  tt_metal/third_party/umd
   (this repo)              (pinned SHA)                          (tt-umd, pinned SHA)
```

At build time tt-foil does `find_package(umd)` from
`${TT_METAL_BUILD_DIR}/lib/cmake/umd` (see `cmake/tt_metal_deps.cmake`) — i.e.
it links whatever UMD **tt-metal built**.

**The deliberate design decision (do not relitigate without the user):**
the unit of update is the **tt-metal source SHA**, not tt-umd on its own.
tt-umd releases are only the *signal* ("is there a newer UMD worth getting?").
The *mechanism* is always: bump `third_party/tt-metal` to a SHA whose UMD
submodule pins the desired tt-umd, so the tt-metal ↔ tt-umd pairing that
tt-metal CI already validated comes along for free.

Rationale (full version in the design dialogue that created this skill):
- tt-foil compiles tt-metal's **HAL** (`hal.cpp` / `bh_hal*.cpp`) and vendors
  its **llrt** (`tt_memory` / `tt_elffile`) from the tt-metal *source* tree, and
  builds **firmware** from that same source. Those define a host↔device layout
  contract (mailbox offsets, `launch_msg_t`, `relocate_dev_addr`, RiscId map,
  noc_parameters) that must stay in lockstep with the firmware.
- tt-metal's UMD submodule pin is the version tt-metal's HAL was validated
  against. Bumping UMD *independently* would make tt-foil own that
  compatibility audit forever — exactly the `cb_reserve_back`-hang / stale-ELF
  drift class CLAUDE.md warns about, just moved to the umd↔HAL boundary.

So: **never bump `tt_metal/third_party/umd` in isolation.** Always drive the
update from a tt-metal SHA bump.

## What this skill does / does not do

Does:
1. Report current vs available tt-umd versions and what changed.
2. Filter the umd diff to tt-foil's actual API surface + tt-metal's layout
   contract, and assess impact.
3. Pick a candidate tt-metal SHA and bump `third_party/tt-metal` to it.
4. Adapt tt-foil's call sites / contract touch-points to compile against the new
   UMD/HAL.
5. Write an impact report.

Does **not** (hands off to the user with exact commands):
- Run the ~30 min tt-metal rebuild.
- Run the HW `ctest` sweep (chip-locking; needs `tt-smi -r` on failure).

## Procedure

### 1. Establish the current baseline

```bash
cd /home/kyamaguchi/tt-foil
git submodule status third_party/tt-metal
git -C third_party/tt-metal submodule status tt_metal/third_party/umd
```

Record the tt-metal SHA + tag and the **current umd pin** (commit + tag).
(At the time this skill was written: tt-metal `v0.71.0-dev20260513-3` pinning
umd `v0.9.5-dev.260424-43-g2cf326e3e52`. These drift — always re-read.)

### 2. Collect the signal: tt-umd releases

```bash
gh api repos/tenstorrent/tt-umd/releases --jq '.[] | "\(.tag_name)\t\(.published_at)"' | head -20
```

List releases newer than the current pin. For each candidate of interest, pull
its notes to judge whether it's worth chasing:

```bash
gh api repos/tenstorrent/tt-umd/releases/tags/<TAG> --jq '.body'
```

If nothing newer is relevant to tt-foil's surface (§ "Impact analysis"), stop
and report "no actionable UMD changes" — do not bump for the sake of bumping.

### 3. Resolve the candidate tt-metal SHA

The update is driven by tt-metal, so map "desired umd" → "tt-metal SHA that
pins it". Two approaches:

**Forward (preferred, simplest):** pick a concrete tt-metal ref (latest `main`,
or a release tag), read what umd it pins, and check that umd is at-or-past what
you want:

```bash
# latest main commit
gh api repos/tenstorrent/tt-metal/commits/main --jq '.sha'
# umd pin at that ref (gitlink tree entry under tt_metal/third_party/)
gh api "repos/tenstorrent/tt-metal/contents/tt_metal/third_party/umd?ref=<TTM_SHA>" \
    --jq '.sha'   # <- this is the tt-umd commit SHA
```

Then name that umd commit. Note: tt-metal usually pins a **dev commit**, not a
clean tag (current pin `…-43-g2cf326e` is 43 commits past tag `v0.9.5-dev.260424`),
so an exact-SHA tag lookup will normally return nothing. Use `git describe`:

```bash
# against the locally checked-out umd (most reliable, gives the -N-g<sha> form):
git -C third_party/tt-metal/tt_metal/third_party/umd describe --tags <UMD_SHA>
# or, for a remote SHA you haven't fetched, fall back to the commit date:
gh api "repos/tenstorrent/tt-umd/commits/<UMD_SHA>" --jq '.commit.committer.date'
```

`git submodule status tt_metal/third_party/umd` already prints this
`describe`-form string for the current pin — use it directly for the baseline.

**Reverse (optional, when you need the *earliest* tt-metal pinning umd ≥ vX):**
walk tt-metal commits that touched the gitlink and inspect each one's umd SHA:

```bash
gh api "repos/tenstorrent/tt-metal/commits?path=tt_metal/third_party/umd&per_page=30" \
    --jq '.[] | "\(.sha)\t\(.commit.committer.date)"'
# for each candidate <TTM_SHA>, re-read its umd pin with the contents call above
```

Pick the candidate `<TTM_SHA>` and note the umd delta `<UMD_OLD>..<UMD_NEW>`.

### 4. Impact analysis (two axes)

#### Axis 1 — tt-umd API surface tt-foil actually calls

Diff the umd source between the old and new pins, then check it against the
**exact surface tt-foil uses** (re-grep to stay current — this list is the
ground truth, not memory):

```bash
grep -rn "tt::umd::\|umd/device\|ClusterOptions\|set_barrier_address_params\|\
l1_membar\|deassert_risc_reset\|assert_risc_reset\|SocDescriptor\|RiscType\|\
BarrierAddress" src/ include/ tools/ tests/
```

**Include `tools/` and `tests/` in the grep, not just `src/`/`include/`.**
`tools/tt_foil_inspect.cpp` calls the umd reset API directly and was missed in
the v0.9.6 pass when only `src/` was searched — a fourth call site that broke
the build after the first three were fixed.

Known touch-points (verify, don't trust blindly):
- `src/umd_boot.cpp`, `src/device.cpp` — `tt::umd::ClusterOptions` (`.target_devices`,
  `.chip_type`), `std::make_unique<tt::umd::Cluster>(std::move(opts))`,
  `set_barrier_address_params(bp)`.
- `src/reset.cpp` — `driver.l1_membar(chip)`, `driver.deassert_risc_reset(chip,
  core, tt::umd::RiscType::BRISC, /*staggered_start=*/true)`, assert variant.
- `src/dispatch.cpp` — `l1_membar(chip, cores)` (set-of-cores overload),
  `tt::umd::CoreCoord` construction.
- `src/device.cpp` — `tt::umd::SocDescriptor`, logical→translated coord
  (`soc_logical_to_translated`), `cluster_descriptor_types.hpp`.
- `src/noc_addr.{hpp,cpp}`, `src/core_info_init.*`, `src/mailbox_init.*`,
  `src/cb_config.cpp`, `src/firmware_load.cpp` — `tt::umd::CoreCoord`,
  read/write L1/DRAM.

Diff commands:

```bash
cd third_party/tt-metal/tt_metal/third_party/umd
git fetch --tags origin
git log --oneline <UMD_OLD>..<UMD_NEW> -- device/api include
git diff <UMD_OLD>..<UMD_NEW> -- device/api/umd/device/cluster.hpp \
    device/api/umd/device/types
```

For every changed signature in that surface, decide: no-op / mechanical rename /
semantic change. **Semantic changes to `CoreCoord`/`SocDescriptor` coordinate
translation are the dangerous silent ones** — flag them loudly (NOC writes to
the wrong core resolve as "systematically scaled/wrong" output, per CLAUDE.md).

**A "mechanical rename" can hide a semantic regression — check the
implementation, not just the signature.** The v0.9.6 bump removed
`Cluster::assert_risc_reset_at_core(chip, core)` and the obvious migration is
`assert_risc_reset(chip, core, RiscType::ALL_TENSIX)` (mask is even equivalent).
But the *old* call mapped to `send_tensix_risc_reset` → a single **absolute**
`set_risc_reset_state(core, mask)` (no read), while the *new* `assert_risc_reset`
is **read-modify-write** (it reads the soft-reset reg from the — possibly
mid-NOC-transaction — core, then ORs in bits). That read is unreliable for
exactly the busy cores the per-core unicast reset exists to force down, so the
"mechanical" swap introduced **flaky, run-to-run-varying cross-test hangs**
(boot-time `0x40` on a later test, or kernel-time `0x80`), while each test
passed in isolation. Fix: restore the absolute write via
`Cluster::get_tt_device(chip)->set_risc_reset_state(core, <abs bits>)` (both
are public). Lesson: for reset/barrier/membar/NOC primitives, read the umd
*impl* (`device/tt_device/tt_device.cpp`, arch `get_soft_reset_reg_value`) to
confirm RMW-vs-absolute and read-vs-write-only behavior didn't change.

#### Axis 2 — tt-metal layout contract (because a tt-metal SHA also moved)

A tt-metal bump moves HAL/dev_msgs/firmware too. Check the contract surfaces
CLAUDE.md depends on:

```bash
cd third_party/tt-metal
git fetch --tags origin
git log --oneline <TTM_OLD>..<TTM_NEW> -- \
    tt_metal/hw/inc tt_metal/llrt/hal tt_metal/hal \
    tt_metal/llrt/hal/generated/dev_msgs.hpp tt_metal/hw/firmware
```

Specifically check for changes to:
- Mailbox offsets (`LAUNCH`=0x70, `GO_MSG`=0x3F0, `GO_MSG_INDEX`=0x420,
  `KERNEL_CONFIG`=0x9E00, `BANK_TO_NOC_SCRATCH`=0x12E00, …) — see CLAUDE.md table.
- `launch_msg_t` / `go_msg_t` layout (`tt_metal/llrt/hal/generated/dev_msgs.hpp`).
- `relocate_dev_addr` semantics (XIP relocation).
- RiscId → (class, type, idx) mapping (`src/risc_id*`, `test_risc_ids`).
- `noc_parameters.h` (`NOC_ADDR_NODE_ID_BITS`=6, `NOC_ADDR_LOCAL_BITS`=36) —
  `make_noc_unicast_addr` depends on these.
- The firmware-build entry points (`brisc.cc`/`ncrisc.cc`/`trisc.cc`,
  `noc_get_cfg_reg` & friends) — renames here break `scripts/build_firmware.sh`.

Any change here means tt-foil's firmware + HAL + manifest must be rebuilt from
the new source and re-validated; the kernel `*_weakened.elf` ↔ firmware pairing
is the thing that silently breaks (`cb_reserve_back` hang).

**The diff-stat is NOT enough for Axis 2 — the firmware build is the real
contract check.** A tt-metal bump can *add* a header that shadows one of
tt-foil's firmware low-level includes via `-I` order, with zero change to the
files you'd think to inspect. This bit the v0.9.6 bump: tt-metal added
`tt_metal/hw/inc/api/dataflow/noc.h` (kernel-only; it pulls `dataflow_api.h`
which `#error`s under `#if !defined(KERNEL_BUILD)`). Because
`scripts/build_firmware.sh` listed `-I .../hw/inc/api/dataflow` *before*
`-I .../hw/inc/internal/tt-1xx/blackhole/noc`, `brisc.cc`'s `#include "noc.h"`
suddenly resolved to the kernel header instead of the firmware low-level one,
and the whole firmware build collapsed (`'noc_wr_ack_received' was not declared`
and ~hundreds of downstream errors). Fix: keep the `internal/` low-level include
dirs *ahead of* `api/`/`api/dataflow`.

**Rebuild EVERY kernel set against the new firmware — `build_ops.sh` only
covers `ops/`.** Stale `*_weakened.elf` linked against the old firmware run to
completion but produce garbage or hang `cb_reserve_back` (CLAUDE.md). The v0.9.6
bump needed three separate rebuilds: `scripts/build_ops.sh` (ops/), each
`examples/*/build_kernels.sh`, AND `models/<m>/build_kernels.sh` (e.g.
resnet20's 48 ELFs, which were missed at first and made `test_resnet20` the lone
sweep failure — a kernel-time `0x80` hang). The CMake auto-build stamps usually
catch these on `cmake --build`, but a standalone `build_ops.sh` run does not.
After bumping, find every kernel build script
(`find . -name build_kernels.sh -o -name build.sh | grep -v third_party`) and
confirm its outputs are newer than `build/firmware/`.

**HW validation gotchas (these cost hours on the v0.9.6 run):**
- `tt-smi` may not be on the non-interactive `PATH` — use the full path
  (`/home/kyamaguchi/tt-venv/bin/tt-smi`). A silently-failing `tt-smi -r`
  (redirected to /dev/null) means the chip is never reset; accumulated hangs
  then wedge it past `tt-smi -r` recovery, requiring a `tt-flash` reflash
  (see the chip-recovery memory). Never redirect the reset's output away.
- A genuine hang wedges the chip, so **every later test in the same `ctest`
  invocation also fails** (boot `0x40`) — the failing-test list is meaningless
  after the first hang. Find the *first* genuine failure with
  `ctest --stop-on-failure --timeout 8` from a fresh reset; isolate true hangs
  (`0x80` kernel / `0x40` boot on a clean chip) from contamination by running
  the suspect alone after a reset.
- "Which test fails varies run to run" ⇒ a reliability/reset regression, not a
  kernel-logic bug — look at the teardown/reset path (see Axis 1 above), not the
  kernel that happened to fail this time.

**The same `-I` reorder is needed in EVERY kernel build script, not just
`build_firmware.sh`.** `c_tensix_core.h` (pulled into kernels via the LLK /
compute path) also does a bare `#include "noc.h"` and needs the low-level
symbols (`noc_copy_word_be`, `noc_atomic_increment`, …). The v0.9.6 bump
therefore broke all 18 `ops/*` and all the `examples/*` kernel builds with the
identical error until the include block was reordered in each of
`ops/*/build.sh` and `examples/*/build_kernels.sh` (they each carry a copy of
the block — ~42 files; a scripted reorder is the sane way). Reorder rule: move
the `-I".../hw/inc/api" -I".../hw/inc/api/dataflow"` line to *after* the
`-I".../hw/inc/internal/tt-1xx/blackhole/noc"` line.

So, as part of Axis 2, **build the firmware target AND the op kernels** before
declaring the bump clean (both are pure compile, no chip):
```bash
cmake --build build --target tt_foil_firmware
TT_METAL_ROOT=$PWD/third_party/tt-metal bash scripts/build_ops.sh
```
and diff the device-side surfaces that feed it:
```bash
git -C third_party/tt-metal diff --stat <TTM_OLD> <TTM_NEW> -- \
    tt_metal/hw/inc/api tt_metal/hw/firmware/src tt_metal/hw/inc/internal/tt-1xx
```
Watch for *new* headers under `hw/inc/api/**` whose basename collides with a
firmware low-level header (`noc.h`, etc.) — those are the silent `-I`-shadowing
traps. (A new header is invisible to a "did file X change" check; you have to
look for additions, or just let the firmware compile surface it.)

### 5. Apply the bump

```bash
cd /home/kyamaguchi/tt-foil/third_party/tt-metal
git fetch origin
git checkout <TTM_SHA>
git submodule update --init --recursive --depth 1 tt_metal/third_party/umd
cd /home/kyamaguchi/tt-foil
git add third_party/tt-metal
# confirm the new umd pin came along:
git -C third_party/tt-metal submodule status tt_metal/third_party/umd
```

Then adapt tt-foil to compile:
- Apply the mechanical renames / signature changes found in Axis 1 to the call
  sites in `src/`.
- If `cmake/tt_metal_deps.cmake` references moved (header paths, generated
  include dirs, fmt/tracy/umd cmake package locations), update them.
- Touch nothing speculatively — only what the diff forces.

### 6. Hand off build + test (do NOT run these yourself)

Present the user the exact commands. Per CLAUDE.md / "in-tree submodule":

```bash
# 1. rebuild tt-metal against the new SHA (~30 min)
( cd third_party/tt-metal && ./build_metal.sh --release )

# 2. reconfigure + rebuild tt-foil (libtt_foil.a + binaries + firmware)
TT_METAL_BUILD_DIR=$PWD/third_party/tt-metal/build_Release \
    cmake -B build -DTT_FOIL_HW_TESTS=ON -DTT_FOIL_DEVICE=0
cmake --build build -j$(nproc)

# 3. rebuild prebuilt op kernels against the same source
TT_METAL_ROOT=$PWD/third_party/tt-metal bash scripts/build_ops.sh

# 4. regression sweep (must be 100% green)
tt-smi -r 0
ctest --test-dir build
```

Call out the UMD/contract-sensitive tests to watch first: `test_umd_open`,
`test_multi_core_boot`, `test_risc_ids`, `test_cb_config`, `test_tile_copy`,
and the dispatch/noc tests. If firmware-contract changes were flagged in Axis 2,
the `cb_reserve_back` hang is the expected failure mode — rebuild kernels +
firmware from the new source before suspecting anything else.

### 7. Report

Emit a concise summary:

```
## tt-umd update report
- tt-metal:  <TTM_OLD tag/sha>  →  <TTM_NEW tag/sha>
- tt-umd:    <UMD_OLD tag>      →  <UMD_NEW tag>   (transitive via tt-metal)

### tt-umd changes relevant to tt-foil (Axis 1)
- <commit/PR>: <what> → impact on <call site> = none|mechanical|semantic
- ...

### tt-metal layout-contract changes (Axis 2)
- mailbox offsets / launch_msg_t / relocate / RiscId / noc_params: <none | details>
- firmware build impact: <none | details>

### Code adapted in tt-foil
- <file:line> — <change>

### Required follow-up (user runs)
- tt-metal rebuild + cmake reconfigure + build_ops + ctest (commands above)
- watch: test_umd_open, test_multi_core_boot, test_risc_ids, ...

### Verdict
<worth landing | hold — no actionable change | blocked on contract break X>
```

If the analysis concludes nothing actionable, **do not bump** — report the
no-op and stop. Bumping for its own sake just risks a contract break for zero
gain.

## Guardrails

- Never edit `tt_metal/third_party/umd`'s pin directly; only move it via a
  tt-metal `git checkout` (§5). An independently-pinned UMD reintroduces the
  audit burden this skill exists to avoid.
- Never link `libtt_metal.so` or call into `MetalContext` / `tt::Cluster` to
  "make it build" — that is the dual-UMD wall (CLAUDE.md). If a UMD change seems
  to require it, stop and surface it; the fix is elsewhere.
- Stay inside the in-tree submodule. Do not point tt-foil at a sibling
  `~/tt-metal` to dodge a build break — header/`*_weakened.elf` skew silently
  hangs `cb_reserve_back` at runtime.
- Keep the diff minimal and one logical unit. Commit message: no "Phase/Step"
  language; `Co-Authored-By: Claude <noreply@anthropic.com>` trailer.
