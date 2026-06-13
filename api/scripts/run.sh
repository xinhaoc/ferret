#!/usr/bin/env bash
# Launch a ferret run with a verified-clean workspace.
#
#   scripts/run.sh <task.yaml> [--max-iterations N] [extra ferret args...]
#
# What it does:
#   1. Verifies parent repo HEAD is real ferret code (not v###/a### agent commits).
#   2. Resets workspace: removes it, recreates it, `git init` inside so
#      workspace has its OWN .git (otherwise agent's `cd workspace && git log`
#      walks up into the parent ferret repo and reads old v### tags).
#   3. Launches `python -m ferret.main` in nohup background, logs to run.log.
#
# Flags:
#   --keep-workspace   skip the workspace wipe (use when resuming a run)
#   --no-detach        run in foreground (don't nohup+background)

set -euo pipefail

# api/scripts/run.sh → ferret root is TWO levels up (api/ moved under ferret/).
FERRET_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"   # override with PYTHON=/path/to/python if needed

KEEP_WS=0
DETACH=1
TASK=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-workspace) KEEP_WS=1; shift ;;
        --no-detach)      DETACH=0; shift ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *)
            if [[ -z "$TASK" ]]; then TASK="$1"
            else EXTRA_ARGS+=("$1")
            fi
            shift ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "usage: $0 <task.yaml> [--max-iterations N] [--keep-workspace] [--no-detach]" >&2
    exit 2
fi

# Resolve task path (allow either absolute or relative-to-ferret)
if [[ "$TASK" != /* && ! -f "$TASK" ]]; then
    TASK="$FERRET_DIR/$TASK"
fi
if [[ ! -f "$TASK" ]]; then
    echo "task file not found: $TASK" >&2
    exit 2
fi

# ── 1. Sanity-check parent repo HEAD ─────────────────────────────────────────
cd "$FERRET_DIR"
HEAD_SUBJ=$(git log -1 --format=%s 2>/dev/null || echo "")
if [[ "$HEAD_SUBJ" =~ ^(v|a)[0-9]{3}: ]]; then
    echo "ERROR: ferret HEAD looks like an agent commit ($HEAD_SUBJ)." >&2
    echo "Run: git fetch origin && git reset --hard origin/main" >&2
    exit 3
fi

# ── 2. Reset workspace (unless --keep-workspace) ─────────────────────────────
if [[ "$KEEP_WS" -eq 0 ]]; then
    rm -rf "$FERRET_DIR/workspace"
    mkdir "$FERRET_DIR/workspace"
    git -C "$FERRET_DIR/workspace" init -q
fi

# Verify workspace has its own .git and no commits
if [[ ! -d "$FERRET_DIR/workspace/.git" ]]; then
    echo "ERROR: workspace has no .git — agent's git commands will leak into parent repo." >&2
    exit 3
fi
if git -C "$FERRET_DIR/workspace" log --oneline 2>/dev/null | grep -q .; then
    echo "ERROR: workspace git repo is non-empty — old v### commits would leak to agent prompt." >&2
    git -C "$FERRET_DIR/workspace" log --oneline | head -3 >&2
    exit 3
fi

# ── 3. Launch ────────────────────────────────────────────────────────────────
# `python -m ferret.api.main` requires cwd to be the parent of ferret/.
RUN_DIR="$(dirname "$FERRET_DIR")"
LOG="$FERRET_DIR/run.log"
REL_TASK="$(realpath --relative-to="$RUN_DIR" "$TASK")"

echo "ferret root : $FERRET_DIR"
echo "HEAD        : $(git -C "$FERRET_DIR" log -1 --oneline)"
echo "task        : $TASK"
echo "log         : $LOG"

cd "$RUN_DIR"
if [[ "$DETACH" -eq 1 ]]; then
    nohup "$PYTHON" -m ferret.api.main "$REL_TASK" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1 &
    echo "launched PID $! — tail -f $LOG"
else
    exec "$PYTHON" -m ferret.api.main "$REL_TASK" "${EXTRA_ARGS[@]}"
fi
