#!/usr/bin/env bash
# Initialize one ferret workspace for the Claude-Code mainthread path.
#
#   scripts/cc-init.sh <N> <task.yaml>
#
# What it does:
#   1. Sanity-check the parent ferret repo (HEAD not an agent v###/a###
#      commit — same guard as scripts/run.sh).
#   2. Create workspaceN/ (refuses if it already exists and is non-empty).
#   3. `git init` inside workspaceN/ so its history is independent of
#      the parent ferret repo AND of sibling workspaces.
#   4. Copy the task.yaml into workspaceN/task.yaml (the spec).
#   5. Write a fresh progress.md skeleton (Plan / Tried / Untried /
#      Current Best). The planner subagent later fills in details.
#
# This script does NOT launch claude; use scripts/cc-run.sh for that.

set -euo pipefail

FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <N> <task.yaml>" >&2
    echo "       N is the workspace index (1..8 by convention)." >&2
    exit 2
fi

N="$1"
TASK="$2"

case "$N" in
    ''|*[!0-9]*)
        echo "ERROR: N must be a positive integer (got: $N)" >&2
        exit 2
        ;;
esac

# Resolve task path: absolute, or relative to ferret root.
if [[ "$TASK" != /* && ! -f "$TASK" ]]; then
    TASK="$FERRET_DIR/$TASK"
fi
if [[ ! -f "$TASK" ]]; then
    echo "ERROR: task file not found: $TASK" >&2
    exit 2
fi

# ── Sanity-check parent repo HEAD (same logic as scripts/run.sh) ───────────
cd "$FERRET_DIR"
HEAD_SUBJ=$(git log -1 --format=%s 2>/dev/null || echo "")
if [[ "$HEAD_SUBJ" =~ ^(v|a)[0-9]{3}: ]]; then
    echo "ERROR: ferret HEAD looks like an agent commit ($HEAD_SUBJ)." >&2
    echo "       Run: git fetch origin && git reset --hard origin/main" >&2
    exit 3
fi

# ── Bootstrap docs/dev-memory/ from the tracked seed dir if absent ─────────
# dev-memory holds shared, host-specific knowledge appended by the
# memory-keeper subagent. It is gitignored (so runtime appends don't
# pollute the repo), but the initial template lives in docs/dev-memory-seed/.
# Copy seed -> live on first init when the live dir is missing.
if [[ ! -d "$FERRET_DIR/docs/dev-memory" || -z "$(ls -A "$FERRET_DIR/docs/dev-memory" 2>/dev/null || true)" ]]; then
    if [[ -d "$FERRET_DIR/docs/dev-memory-seed" ]]; then
        mkdir -p "$FERRET_DIR/docs/dev-memory"
        cp -n "$FERRET_DIR/docs/dev-memory-seed/"*.md "$FERRET_DIR/docs/dev-memory/" 2>/dev/null || true
        echo "Bootstrapped docs/dev-memory/ from docs/dev-memory-seed/"
    else
        echo "WARN: no docs/dev-memory-seed/ template found — memory-keeper will start from empty files." >&2
    fi
fi

WS="$FERRET_DIR/workspace$N"

# ── Create workspaceN/ ─────────────────────────────────────────────────────
if [[ -d "$WS" ]]; then
    if [[ -n "$(ls -A "$WS" 2>/dev/null || true)" ]]; then
        echo "ERROR: $WS already exists and is non-empty. Refusing to clobber." >&2
        echo "       To re-init: rm -rf $WS && $0 $N $TASK" >&2
        exit 3
    fi
fi
mkdir -p "$WS"

# ── Independent git history inside workspaceN/ ─────────────────────────────
git -C "$WS" init -q
if [[ ! -d "$WS/.git" ]]; then
    echo "ERROR: failed to create $WS/.git" >&2
    exit 3
fi

# ── Copy task.yaml verbatim (the spec) ─────────────────────────────────────
cp "$TASK" "$WS/task.yaml"

# ── progress.md skeleton ───────────────────────────────────────────────────
TASK_NAME=$(grep -E '^name:' "$TASK" | head -1 | sed 's/^name:[[:space:]]*//')
cat > "$WS/progress.md" <<EOF
# progress.md — ${TASK_NAME:-<task>}

## Mirage interface
(populated by planner — see \$MIRAGE_ROOT/include/mirage/)

## Plan
(populated by planner on cold-start)

## Tried
(append-only — every iteration that didn't improve goes here with a one-line summary)

## Untried (Hard)
(stretch ideas the agent has considered but not yet attempted)

## Current Best
(updated after every \`git tag v###\`: <tag>: <TFLOPS by config>, <technique>)
EOF

# ── Verify the workspace looks right ───────────────────────────────────────
if ! git -C "$WS" log --oneline 2>/dev/null | grep -q .; then
    : # expected — no commits yet
fi
if git -C "$WS" tag | grep -q .; then
    echo "WARN: $WS already has tags. The git init may have been seeded "
    echo "      from a stale dir. Inspect manually." >&2
fi

echo "Initialized ferret workspace: $WS"
echo "  task.yaml   : $TASK_NAME"
echo "  progress.md : skeleton written"
echo "  .git        : new, no commits, no tags"
echo
echo "Next: scripts/cc-run.sh $N"
