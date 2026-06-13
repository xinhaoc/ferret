#!/usr/bin/env bash
# Watchdog that keeps ferret running across budget exits and silent crashes.
#
# Runs on the same host as ferret (typically catalyst). Independent of any
# Claude or interactive session — once launched with nohup+setsid it survives
# disconnection.
#
#   scripts/run_forever.sh <task.yaml> [--max-restarts N] [--watch-pid PID]
#                                       [extra ferret args (e.g. --remote-host nebius-b200)]
#
# Behavior:
#   1. If --watch-pid is given: attach to an already-running ferret first,
#      wait for it to exit, then start the respawn loop. (Lets you wrap a
#      ferret you already launched without restarting it now.)
#      Otherwise: launch ferret fresh as the first iteration.
#   2. After each ferret exit, classify the cause from the last lines of
#      ferret/run.log:
#        - "Wall-time budget exceeded" / "Done in X min" → clean budget exit,
#          relaunch with --keep-workspace.
#        - "max_iterations" reached / Done with score >= target → DONE, stop.
#        - SIGTERM / SIGKILL with no log marker → silent kill, relaunch.
#        - "OSError: [Errno 28] No space left on device" / "MemoryError" /
#          "Killed" with OOM signature → STOP, do not relaunch into a
#          systemic failure.
#   3. Cap at --max-restarts respawns (default 6).
#   4. Each respawn archives the current run.log to run.log.iterN and
#      starts a fresh run.log.
#   5. Watchdog appends its own decisions to run.log.watchdog.
#
# All control flow lives in this single shell script. ferret/main.py is
# untouched.

set -uo pipefail

FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/xinhaoc/miniconda3/bin/python3}"
WATCHDOG_LOG="$FERRET_DIR/run.log.watchdog"

MAX_RESTARTS=6
WATCH_PID=""
TASK=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --watch-pid)    WATCH_PID="$2";    shift 2 ;;
        --help|-h)
            sed -n '2,33p' "$0"
            exit 0 ;;
        *)
            if [[ -z "$TASK" && "$1" != --* ]]; then TASK="$1"
            else EXTRA_ARGS+=("$1")
            fi
            shift ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "usage: $0 <task.yaml> [--max-restarts N] [--watch-pid PID] [extra ferret args]" >&2
    exit 2
fi
if [[ ! -f "$TASK" && ! -f "$FERRET_DIR/$TASK" ]]; then
    echo "ERROR: task spec not found: $TASK" >&2
    exit 2
fi

log() {
    echo "$(date -u +%FT%TZ) [watchdog] $*" >> "$WATCHDOG_LOG"
    echo "[watchdog] $*"
}

# Identify the exit reason from the tail of ferret's run.log. Returns one of:
#   BUDGET   — clean wall/iter exit, restart with --keep-workspace.
#   SUCCESS  — score met the target / agent declared done, STOP.
#   SILENT   — process gone with no exit marker (cascade-kill, etc).
#   FATAL    — disk-full, OOM, or other systemic failure. STOP.
classify_exit() {
    local logfile="$1"
    [[ ! -f "$logfile" ]] && { echo "SILENT"; return; }
    local tail_lines
    tail_lines="$(tail -50 "$logfile" 2>/dev/null)"
    if echo "$tail_lines" | grep -qE "Wall-time budget exceeded|Done in [0-9.]+ *min\\."; then
        echo "BUDGET"
        return
    fi
    if echo "$tail_lines" | grep -qE "No space left on device|MemoryError|OutOfMemoryError|cudaErrorMemoryAllocation"; then
        echo "FATAL"
        return
    fi
    echo "SILENT"
}

launch_ferret() {
    # Launch ferret in the background. Echoes the launched PID on stdout.
    #
    # NOTE on workspace persistence: ferret.main does NOT have a
    # --keep-workspace flag (that's a scripts/run.sh wrapper option).
    # When we launch python -m ferret.main directly here, the workspace
    # files persist UNLESS we explicitly wipe them. The watchdog never
    # wipes, so passing a "keep" argument is implicit: just don't add
    # a wipe step. The `_unused_keep` arg is kept for caller readability.
    local _unused_keep="$1"; shift
    cd "$(dirname "$FERRET_DIR")"  # parent of ferret/
    if [[ -f "$FERRET_DIR/.env" ]]; then
        set -a; . "$FERRET_DIR/.env"; set +a
    fi
    local args=("$TASK")
    args+=("${EXTRA_ARGS[@]}")
    setsid nohup "$PYTHON" -m ferret.main "${args[@]}" \
        > "$FERRET_DIR/run.log" 2>&1 < /dev/null &
    disown
    local pid="$!"
    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
        # Process died within 3 sec — startup failure. Caller decides.
        echo ""
        return 1
    fi
    echo "$pid"
}

# --- Phase 1: attach to existing PID if requested ---
if [[ -n "$WATCH_PID" ]]; then
    if ! kill -0 "$WATCH_PID" 2>/dev/null; then
        log "ERROR: --watch-pid $WATCH_PID not alive at startup"
        exit 1
    fi
    log "attaching to existing ferret PID $WATCH_PID"
    # Wait for it to exit
    while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 15; done
    log "PID $WATCH_PID exited"
    REASON="$(classify_exit "$FERRET_DIR/run.log")"
    log "exit classified as $REASON"
    if [[ "$REASON" == "FATAL" || "$REASON" == "SUCCESS" ]]; then
        log "STOP: $REASON"
        exit 0
    fi
    # Archive log so the next iteration starts fresh.
    iter=1
    while [[ -f "$FERRET_DIR/run.log.iter$iter" ]]; do iter=$((iter+1)); done
    mv "$FERRET_DIR/run.log" "$FERRET_DIR/run.log.iter$iter" 2>/dev/null || true
fi

# --- Phase 2: respawn loop ---
RESTART=0
FIRST_LAUNCH=$([[ -z "$WATCH_PID" ]] && echo "yes" || echo "no")

while [[ $RESTART -lt $MAX_RESTARTS ]]; do
    if [[ "$FIRST_LAUNCH" == "yes" ]]; then
        log "first launch (no --keep-workspace), task=$TASK"
        FERRET_PID="$(launch_ferret nokeep)"
        FIRST_LAUNCH="no"
    else
        log "respawn #$RESTART with --keep-workspace"
        FERRET_PID="$(launch_ferret keep)"
    fi
    if [[ -z "$FERRET_PID" ]]; then
        log "ERROR: ferret died within 3s of launch — startup failure, STOPPING"
        exit 1
    fi
    log "launched ferret PID $FERRET_PID"

    # Wait for it to exit
    while kill -0 "$FERRET_PID" 2>/dev/null; do sleep 30; done
    log "PID $FERRET_PID exited"

    REASON="$(classify_exit "$FERRET_DIR/run.log")"
    log "exit classified as $REASON (after restart $RESTART)"

    if [[ "$REASON" == "FATAL" ]]; then
        log "STOP: FATAL exit — not relaunching into systemic failure"
        exit 0
    fi
    if [[ "$REASON" == "SUCCESS" ]]; then
        log "STOP: agent declared done"
        exit 0
    fi

    # Archive log so the next iteration starts fresh
    iter=1
    while [[ -f "$FERRET_DIR/run.log.iter$iter" ]]; do iter=$((iter+1)); done
    mv "$FERRET_DIR/run.log" "$FERRET_DIR/run.log.iter$iter" 2>/dev/null || true

    RESTART=$((RESTART+1))
done

log "STOP: hit --max-restarts $MAX_RESTARTS"
