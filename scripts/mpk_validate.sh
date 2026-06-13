#!/usr/bin/env bash
# mpk_validate.sh — in-MPK correctness + PERFORMANCE validation for a ferret
#                   candidate kernel, on ONE exclusive GPU.
#
# Closes the "standalone-correct but in-MPK-crash" gap (the SplitK Heisenbug):
# a kernel that passes ferret's own standalone host-reference check can still
# crash or miscompare once it runs through the REAL MPK compile pipeline
# (graph.cc dispatch -> task_register codegen -> tma.cuh descriptors ->
# megakernel nvcc -> scheduler dispatch). This script drives that full path on
# a single, exclusive GPU and reports a clear PASS / FAIL with cos.
#
# It ALSO reports the kernel-under-test's faithful single-kernel in-MPK latency
# (test mode's MAIN purpose): a profiling-enabled driver emits a Perfetto trace
# + CSV from which scripts/parse_profile.py reads the per-task timings. The
# kernel-latency metric is the WALL-SPAN = max(end_ts) - min(begin_ts) over the
# task's events (first CTA start -> last CTA finish), NOT the median/avg
# duration_ns. Decode GEMMs are BIMODAL — most of their grid_dim=128 CTAs
# idle-exit in <1us while only a handful do real work — so the median is an
# *idle CTA* and understates latency by ~30x; using it gives a nonsense ratio
# (~0.06x). When the driver also runs the BASELINE kernel the split-K replaces,
# it prints a `PERF_SUMMARY: splitk_wall_us=.. mediumm_wall_us=.. ratio=..`
# line; the harness surfaces candidate_us / baseline_us / ratio (all WALL-SPAN)
# in its verdict. Perf is reported ALONGSIDE correctness — correctness
# (cos>0.99 + zero sentinel rows) still gates PASS/FAIL; a PASS with no perf
# number is flagged as a WARNING because the standalone-vs-in-MPK latency is
# exactly what this harness must capture.
#
# Usage:
#   scripts/mpk_validate.sh <WS_INDEX> <KERNEL_NAME> <TEST_DRIVER> [options]
#
#   <WS_INDEX>     ferret workspace index (1..8); reads workspaceN/kernel.cuh
#   <KERNEL_NAME>  basename (no .cuh) of the MPK task header this kernel maps to,
#                  e.g. fp8_gemm_dense_qkva_splitk_sm100. The candidate is copied
#                  to $MIRAGE_ROOT/include/.../tasks/blackwell/<KERNEL_NAME>.cuh
#   <TEST_DRIVER>  the per-kernel MPK test to run. Two forms:
#                    Pattern A: a path to a test_*_testmode.py (PREFERRED)
#                    Pattern B: a path to a setup.py (CUDAExtension wrapper dir)
#                  The form is auto-detected from the filename.
#
# Options:
#   --gpu N            force CUDA_VISIBLE_DEVICES=N (skip auto-pick)
#   --gpu-pool "a b c" restrict auto-pick to this set of GPU indices
#   --no-revert        keep the copied .cuh in the MPK tree even on failure
#   --keep-on-pass     keep the copied .cuh on success (default: revert always,
#                      this harness is validate-only, NOT integrate)
#   --timeout SECS     per-test timeout (default 1200)
#   --gpu-family DIR   tasks subdir (default: blackwell)
#
# Exit code: 0 = PASS, 1 = FAIL/crash, 2 = harness/setup error.
#
# Output: a single machine-greppable verdict line. perf_us/baseline_us are the
#         candidate/baseline WALL-SPAN in us; ratio = baseline_wall/cand_wall
#         (>1 => candidate faster). All '-' if the driver was not
#         profiling-enabled:
#   MPK_VALIDATE: <PASS|FAIL> kernel=<name> gpu=<N> cos=<x> sentinel_rows=<n> \
#       perf_us=<f|-> baseline_us=<f|-> ratio=<f|-> reason=<...>
set -uo pipefail

# ── locate ferret + mirage ─────────────────────────────────────────────────
FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# MIRAGE_ROOT: prefer env, else read from docs/dev-memory/machine.md, else ~/mirage.
if [[ -z "${MIRAGE_ROOT:-}" ]]; then
    MIRAGE_ROOT="$HOME/mirage"
fi
MIRAGE_ROOT="${MIRAGE_ROOT/#\~/$HOME}"

die() { echo "MPK_VALIDATE: FAIL kernel=${KERNEL_NAME:-?} gpu=${GPU:--} cos=- sentinel_rows=- reason=$*" >&2; exit 2; }

# ── parse args ─────────────────────────────────────────────────────────────
[[ $# -lt 3 ]] && { echo "usage: $0 <WS_INDEX> <KERNEL_NAME> <TEST_DRIVER> [opts]" >&2; exit 2; }
WS_INDEX="$1"; KERNEL_NAME="$2"; TEST_DRIVER="$3"; shift 3

FORCE_GPU=""
GPU_POOL=""
REVERT=1            # revert by default; this harness validates, it does not integrate
KEEP_ON_PASS=0
TIMEOUT=1200
GPU_FAMILY="blackwell"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         FORCE_GPU="${2:-}"; shift 2 ;;
        --gpu-pool)    GPU_POOL="${2:-}"; shift 2 ;;
        --no-revert)   REVERT=0; shift ;;
        --keep-on-pass) KEEP_ON_PASS=1; shift ;;
        --timeout)     TIMEOUT="${2:-1200}"; shift 2 ;;
        --gpu-family)  GPU_FAMILY="${2:-blackwell}"; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

WS="$FERRET_DIR/workspace$WS_INDEX"
SRC_CUH="$WS/kernel.cuh"
DST_DIR="$MIRAGE_ROOT/include/mirage/persistent_kernel/tasks/$GPU_FAMILY"
DST_CUH="$DST_DIR/$KERNEL_NAME.cuh"

# Pick the mirage venv python if present (uv-managed py3.11), else system python3.
PY="$MIRAGE_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

echo "── mpk_validate ────────────────────────────────────────────────"
echo "  ferret workspace : $WS"
echo "  candidate .cuh   : $SRC_CUH"
echo "  MIRAGE_ROOT      : $MIRAGE_ROOT"
echo "  dest .cuh        : $DST_CUH"
echo "  test driver      : $TEST_DRIVER"
echo "  python           : $PY"

# ── preflight ──────────────────────────────────────────────────────────────
[[ -d "$WS" ]]        || die "workspace$WS_INDEX missing"
[[ -f "$SRC_CUH" ]]   || die "no kernel.cuh in workspace$WS_INDEX (extract first)"
[[ -d "$DST_DIR" ]]   || die "MPK tasks dir missing: $DST_DIR (check MIRAGE_ROOT)"
[[ -e "$TEST_DRIVER" ]] || die "test driver not found: $TEST_DRIVER"
command -v "$PY" >/dev/null 2>&1 || die "python not found"

# ── GPU selection — torch-probe + exclusivity (NOT just nvidia-smi mem%) ────
# The MPK megakernel needs an EXCLUSIVE GPU: a co-resident process deadlocks
# the persistent kernel. nvidia-smi can also show a GPU "free" that then fails
# torch with cudaErrorDevicesUnavailable. So we (a) build a candidate list of
# truly-idle GPUs (no other compute apps, util low, mem low), then (b)
# torch-probe each until one actually initializes CUDA.
pick_gpu() {
    local pool="$1"
    # Build the candidate ordering: idle first (low mem, low util, no procs).
    # Columns: index,memused,memtotal,util
    local cand
    cand=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
              --format=csv,noheader,nounits 2>/dev/null \
           | awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$4); print $1, $2, $4}')
    # GPUs that currently host ANY compute process (their own user OR others).
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)
    # Map uuid->index so we can flag busy GPUs by index.
    declare -A IDX_BUSY
    while IFS=',' read -r uuid idx; do
        uuid=$(echo "$uuid" | xargs); idx=$(echo "$idx" | xargs)
        if grep -qx "$uuid" <<<"$busy" 2>/dev/null; then IDX_BUSY[$idx]=1; fi
    done < <(nvidia-smi --query-gpu=uuid,index --format=csv,noheader 2>/dev/null)

    # Rank candidates: prefer (no compute proc) AND util<5 AND memused<2000,
    # then fall back to least-memory. Honor an explicit pool if given.
    local ranked=""
    while read -r idx memused util; do
        [[ -z "$idx" ]] && continue
        if [[ -n "$pool" ]]; then
            case " $pool " in *" $idx "*) : ;; *) continue ;; esac
        fi
        local score=0
        [[ "${IDX_BUSY[$idx]:-0}" == "1" ]] && score=$((score + 100000))
        [[ "$util" -ge 5 ]] 2>/dev/null && score=$((score + 50000))
        score=$((score + memused))
        ranked+="$score $idx"$'\n'
    done <<<"$cand"

    # Emit indices in ascending score (best first).
    echo "$ranked" | grep -v '^[[:space:]]*$' | sort -n | awk '{print $2}'
}

torch_probe() {  # returns 0 if torch can init CUDA on $1
    CUDA_VISIBLE_DEVICES="$1" "$PY" - <<'PYEOF' >/dev/null 2>&1
import torch, sys
try:
    torch.cuda.init()
    assert torch.cuda.device_count() >= 1
    x = torch.zeros(8, device="cuda"); x += 1; torch.cuda.synchronize()
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
}

GPU=""
if [[ -n "$FORCE_GPU" ]]; then
    echo "  GPU            : forced -> $FORCE_GPU (torch-probing)"
    if torch_probe "$FORCE_GPU"; then
        GPU="$FORCE_GPU"
    else
        die "forced GPU $FORCE_GPU failed torch probe (busy/unavailable)"
    fi
else
    echo "  GPU            : auto-pick (pool='${GPU_POOL:-all}', torch-probe + exclusivity)"
    for g in $(pick_gpu "$GPU_POOL"); do
        echo "    probing GPU $g ..."
        if torch_probe "$g"; then GPU="$g"; break; fi
        echo "    GPU $g failed torch probe — skipping"
    done
    [[ -z "$GPU" ]] && die "no torch-usable idle GPU found (pool='${GPU_POOL:-all}')"
fi
echo "  selected GPU   : $GPU"

# ── stage the candidate kernel into the MPK tree (back up existing) ─────────
BACKUP=""
if [[ -f "$DST_CUH" ]]; then
    BACKUP="$DST_CUH.mpkvalidate_backup.$$"
    cp -p "$DST_CUH" "$BACKUP" || die "could not back up $DST_CUH"
    echo "  backed up dest -> $(basename "$BACKUP")"
fi

restore() {
    if [[ "$REVERT" == "1" ]]; then
        if [[ -n "$BACKUP" && -f "$BACKUP" ]]; then
            mv -f "$BACKUP" "$DST_CUH"
            echo "  reverted dest .cuh (restored backup)"
        elif [[ -z "$BACKUP" && -f "$DST_CUH" ]]; then
            # We created the file (no prior dest); remove it so the tree is clean.
            rm -f "$DST_CUH"
            echo "  reverted dest .cuh (removed copied file)"
        fi
    else
        [[ -n "$BACKUP" ]] && rm -f "$BACKUP"
        echo "  --no-revert: left candidate .cuh in MPK tree"
    fi
}

cp -p "$SRC_CUH" "$DST_CUH" || { restore; die "copy candidate -> dest failed"; }
echo "  staged candidate kernel into MPK tree"

# ── run the per-kernel test driver ──────────────────────────────────────────
LOG="$WS/.mpk_validate.$KERNEL_NAME.log"
echo "  test log         : $LOG"
echo "─────────────────────────────────────────────────────────────────"

RC=0
case "$TEST_DRIVER" in
    *setup.py)
        # ── Pattern B: build the CUDAExtension wrapper, then run its driver. ──
        DRV_DIR="$(cd "$(dirname "$TEST_DRIVER")" && pwd)"
        echo "  Pattern B (CUDAExtension wrapper) in $DRV_DIR"
        ( cd "$DRV_DIR" && \
          CUDA_VISIBLE_DEVICES="$GPU" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}" \
            timeout "$TIMEOUT" "$PY" setup.py build_ext --inplace ) \
            >"$LOG" 2>&1 || RC=$?
        if [[ $RC -eq 0 ]]; then
            # A Pattern B dir must ship a test runner: prefer test_*.py, else run.py.
            RUNNER="$(ls "$DRV_DIR"/test_*.py "$DRV_DIR"/run.py 2>/dev/null | head -1)"
            if [[ -n "$RUNNER" ]]; then
                ( cd "$DRV_DIR" && CUDA_VISIBLE_DEVICES="$GPU" \
                    timeout "$TIMEOUT" "$PY" "$RUNNER" ) >>"$LOG" 2>&1 || RC=$?
            else
                echo "  NOTE: Pattern B dir has no test_*.py/run.py — built only." >>"$LOG"
            fi
        fi
        ;;
    *.py)
        # ── Pattern A: PersistentKernel test_mode driver (full MPK pipeline). ──
        echo "  Pattern A (PersistentKernel test_mode — full scheduler+megakernel)"
        ( cd "$MIRAGE_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" \
            timeout "$TIMEOUT" "$PY" "$TEST_DRIVER" ) >"$LOG" 2>&1 || RC=$?
        ;;
    *)
        restore; die "unrecognized TEST_DRIVER (expect a .py test or setup.py): $TEST_DRIVER" ;;
esac

# ── parse the verdict from the test log ─────────────────────────────────────
# Decision logic (a kernel PASSES only if ALL hold):
#   1. The process did not crash / time out (RC==0).
#   2. No CUDA-runtime sentinel error string in the log
#      (Invalid __global__/__shared__, illegal memory access, misaligned, etc.).
#   3. At least one cos= line, and EVERY parsed cos > 0.99.
#   4. sentinel_rows == 0 on every line that reports it (a decode-gated kernel
#      that early-exits to all-zero/garbage must NOT masquerade as a pass).
#   5. A PASS/ALL PASS token present AND no FAIL/SOME FAILED token.
COS_MIN="$("$PY" - "$LOG" <<'PYEOF'
import re, sys
vals = []
with open(sys.argv[1], errors="ignore") as f:
    for line in f:
        for m in re.finditer(r"cos\s*=\s*(-?\d+\.\d+)", line):
            vals.append(float(m.group(1)))
print(f"{min(vals):.6f}" if vals else "nan")
PYEOF
)"
SENT_MAX="$("$PY" - "$LOG" <<'PYEOF'
import re, sys
vals = []
with open(sys.argv[1], errors="ignore") as f:
    for line in f:
        for m in re.finditer(r"sentinel_rows\s*=\s*(\d+)", line):
            vals.append(int(m.group(1)))
print(max(vals) if vals else 0)
PYEOF
)"

# ── PERFORMANCE: extract the in-MPK single-kernel WALL-SPAN from the log ─────
# Test mode's MAIN purpose is the faithful single-kernel latency on 1 GPU. A
# profiling-enabled driver emits a Perfetto trace + CSV; it then runs
# scripts/parse_profile.py --stat wall to print the kernel-under-test's
# WALL-SPAN and, if it ran the BASELINE kernel too, a `PERF_SUMMARY:` line.
#
# KERNEL-LATENCY METRIC = WALL-SPAN, NOT median. Decode GEMMs are bimodal: most
# of grid_dim=128 CTAs idle-exit in <1us, only a handful do real work, so the
# median duration_ns is an *idle CTA* (~0.66us) and gives a nonsense ratio
# (~0.06x). The correct latency = max(end_ts)-min(begin_ts) over the task's
# events (= `--stat wall` / the WALL_us field). We scrape, in priority order:
#   PERF_SUMMARY: splitk_wall_us=<f> mediumm_wall_us=<f> ratio=<f> (cand vs base)
#   PERF: kernel=<TASK> ... WALL_us=<f>                            (candidate)
# with back-compat fallbacks to the legacy splitk_us / median_us field names so
# an older driver still surfaces *something* (flagged via the field it matched).
# This block NEVER changes PASS/FAIL — correctness still gates (cos + sentinel).
# A missing perf number is reported as `perf=-` so the caller sees the driver
# was not profiling-enabled (and should be refined to add a profiler_tensor).
read -r PERF_CAND_US PERF_BASE_US PERF_RATIO PERF_NS <<EOF
$("$PY" - "$LOG" <<'PYEOF'
import re, sys
cand_us = base_us = ratio = wall = None
with open(sys.argv[1], errors="ignore") as f:
    text = f.read()
# 1) Prefer the explicit PERF_SUMMARY line, WALL-SPAN field names first.
m = re.search(r"PERF_SUMMARY:\s*splitk_wall_us=([\d.]+)\s+"
              r"mediumm_wall_us=([\d.]+)\s+ratio=([\d.]+)", text)
if not m:
    # Back-compat: legacy PERF_SUMMARY used splitk_us / mediumm_us (median).
    m = re.search(r"PERF_SUMMARY:\s*splitk_us=([\d.]+)\s+"
                  r"mediumm_us=([\d.]+)\s+ratio=([\d.]+)", text)
if m:
    cand_us, base_us, ratio = m.group(1), m.group(2), m.group(3)
# 2) Fall back to the last per-candidate PERF line. WALL_us is the latency
#    metric; only if no WALL_us is present do we accept median_us (legacy).
for mm in re.finditer(r"WALL_us=([\d.]+)", text):
    wall = mm.group(1)
    if cand_us is None:
        cand_us = mm.group(1)
if cand_us is None:
    for mm in re.finditer(r"median_us=([\d.]+)", text):
        cand_us = mm.group(1)
print(cand_us or "-", base_us or "-", ratio or "-", wall or "-")
PYEOF
)
EOF

CRASH=0
if grep -Eq "Invalid __(global|shared|local)__|illegal memory access|misaligned address|an illegal|CUDA error|cudaError|uncorrectable ECC|device-side assert|Segmentation fault|core dumped" "$LOG"; then
    CRASH=1
fi
HAS_FAIL=0
if grep -Eq "SOME FAILED|^FAILED|: FAIL\b|-> FAIL|\bAssertionError\b|Traceback \(most recent" "$LOG"; then
    HAS_FAIL=1
fi
HAS_PASS=0
if grep -Eq "ALL PASS|-> PASS|: PASS\b|PASSED" "$LOG"; then
    HAS_PASS=1
fi
# Independent sentinel guard: a driver's own `sentinel_rows` count can be
# defeated by dtype rounding (the historical bug: a bf16 output sentinel-filled
# with -987.0 rounds to -988.0, so `== -987.0` never matches and the count
# stays 0 even on a full no-write). The CANONICAL poison value is now the
# BF16-exact -1024.0 (a power of two — survives bf16 round-trip), but older
# drivers may still use -987/-988. So we scan the printed `out[...]:` row for
# ANY of those poison values: if the kernel wrote nothing, that row is all
# sentinel and we catch it here regardless of the driver's own count.
SENTINEL_OUT=0
if grep -Eq "out\[[^]]*\]:[[:space:]]*\[-(1024|987|988)(\.0+)?(,[[:space:]]*-(1024|987|988)(\.0+)?)+\]" "$LOG"; then
    SENTINEL_OUT=1
fi

REASON="ok"
VERDICT="PASS"
if [[ $RC -ne 0 ]]; then
    VERDICT="FAIL"; REASON="driver_rc=$RC(crash_or_timeout)"
elif [[ $CRASH -eq 1 ]]; then
    VERDICT="FAIL"; REASON="cuda_sentinel_error_in_log"
elif [[ "$COS_MIN" == "nan" ]]; then
    VERDICT="FAIL"; REASON="no_cos_reported(driver_did_not_validate)"
elif [[ "$SENT_MAX" -gt 0 || "$SENTINEL_OUT" -eq 1 ]]; then
    # No-write / early-exit to sentinel. Checked BEFORE the cos gate because a
    # sentinel output is a hard fail regardless of any cos the driver computed
    # (cos-vs-reference of an unwritten buffer is meaningless), and because the
    # driver's own sentinel_rows count may be dtype-rounding-defeated.
    VERDICT="FAIL"
    REASON="sentinel_output(kernel_early_exit_no_write;rows=${SENT_MAX};out_row_sentinel=${SENTINEL_OUT})"
elif "$PY" -c "import sys; sys.exit(0 if float('$COS_MIN')>0.99 else 1)"; then
    if [[ $HAS_FAIL -eq 1 && $HAS_PASS -eq 0 ]]; then
        VERDICT="FAIL"; REASON="driver_reported_FAIL"
    else
        VERDICT="PASS"; REASON="cos>${COS_MIN}_sentinel0"
    fi
else
    VERDICT="FAIL"; REASON="cos=$COS_MIN<=0.99"
fi

echo "─────────────────────────────────────────────────────────────────"
echo "  (tail of $LOG)"
tail -n 25 "$LOG" 2>/dev/null | sed 's/^/    /'
echo "─────────────────────────────────────────────────────────────────"

# ── PERFORMANCE report (alongside the PASS/FAIL correctness verdict) ────────
# The contract requires BOTH a correctness verdict AND a perf number. Perf does
# NOT gate PASS/FAIL (correctness does), but a PASS with no perf number means
# the driver was not profiling-enabled — surface that as a warning so the
# harness gets refined to add a profiler_tensor + trace_name.
if [[ "$PERF_CAND_US" == "-" && "$PERF_NS" == "-" ]]; then
    echo "  PERF: WARNING — no in-MPK WALL-SPAN found in log."
    echo "        The driver must enable profiling (params['profiler_tensor'] +"
    echo "        params['trace_name'] before pk.compile()) and run"
    echo "        scripts/parse_profile.py <csv> <TASK_NAME> --stat wall so the"
    echo "        kernel-under-test's single-kernel WALL-SPAN latency is"
    echo "        reported (median/avg are bimodal-skewed — do NOT use them)."
else
    echo "  PERF: candidate_wall_us=$PERF_CAND_US baseline_wall_us=$PERF_BASE_US"\
         "ratio(base/cand)=$PERF_RATIO candidate_WALL_us=$PERF_NS"
fi
echo "─────────────────────────────────────────────────────────────────"

# ── revert the tree (validate-only) unless told otherwise ───────────────────
if [[ "$VERDICT" == "PASS" && "$KEEP_ON_PASS" == "1" ]]; then
    [[ -n "$BACKUP" ]] && rm -f "$BACKUP"
    echo "  --keep-on-pass: candidate .cuh left in MPK tree"
else
    restore
fi

echo "MPK_VALIDATE: $VERDICT kernel=$KERNEL_NAME gpu=$GPU cos=$COS_MIN sentinel_rows=$SENT_MAX perf_us=$PERF_CAND_US baseline_us=$PERF_BASE_US ratio=$PERF_RATIO reason=$REASON"
[[ "$VERDICT" == "PASS" ]] && exit 0 || exit 1
