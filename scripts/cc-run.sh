#!/usr/bin/env bash
# Launch a Claude-Code mainthread bound to one ferret workspace.
#
#   scripts/cc-run.sh <N> [task.yaml]                        # interactive
#   scripts/cc-run.sh <N> [task.yaml] --prompt "<seed>"      # non-interactive
#   scripts/cc-run.sh <N> [task.yaml] --print-only           # print env + cmd, don't exec
#
# What it does:
#   1. If workspaceN/ doesn't exist or has no task.yaml, calls cc-init.sh
#      first (requires the task.yaml arg in that case).
#   2. Picks a GPU once via pick_gpu.sh and exports CUDA_VISIBLE_DEVICES
#      so every benchmark in this Claude session sees the same GPU.
#   3. Sets FERRET_WORKSPACE=workspaceN and TMPDIR=/tmp/$USER (the ncu
#      workaround documented in docs/dev-memory/machine.md).
#   4. `cd ferret/` and exec `claude` (or `claude -p "<seed>"`).
#
# Notes:
#   - The mainthread reads CLAUDE.md from the ferret root, which says
#     "your workspace is $FERRET_WORKSPACE — first thing, cat its
#     task.yaml". So all routing is via that env var.
#   - Each invocation picks a GPU once. To re-pick (e.g. previous GPU
#     got noisy), exit and re-launch.

set -euo pipefail

FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"

PRINT_ONLY=0
SEED_PROMPT=""
GOAL_TEXT=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)      SEED_PROMPT="${2:-}"; shift 2 ;;
        --goal)        GOAL_TEXT="${2:-}"; shift 2 ;;
        --print-only)  PRINT_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *)             POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
    echo "usage: $0 <N> [task.yaml] [--prompt '<seed>'] [--print-only]" >&2
    exit 2
fi

N="${POSITIONAL[0]}"
TASK="${POSITIONAL[1]:-}"

case "$N" in
    ''|*[!0-9]*)
        echo "ERROR: N must be a positive integer (got: $N)" >&2
        exit 2
        ;;
esac

WS="$FERRET_DIR/workspace$N"

# ── Init if needed ─────────────────────────────────────────────────────────
if [[ ! -d "$WS" || ! -f "$WS/task.yaml" ]]; then
    if [[ -z "$TASK" ]]; then
        echo "ERROR: $WS not initialized and no task.yaml given." >&2
        echo "       usage: $0 $N <task.yaml>" >&2
        exit 2
    fi
    "$FERRET_DIR/scripts/cc-init.sh" "$N" "$TASK"
fi

# ── Bootstrap dev-memory if a fresh clone missed it (cc-init also does
#    this, but resume paths skip cc-init so we re-check here). ──────────────
if [[ ! -d "$FERRET_DIR/docs/dev-memory" || -z "$(ls -A "$FERRET_DIR/docs/dev-memory" 2>/dev/null || true)" ]]; then
    if [[ -d "$FERRET_DIR/docs/dev-memory-seed" ]]; then
        mkdir -p "$FERRET_DIR/docs/dev-memory"
        cp -n "$FERRET_DIR/docs/dev-memory-seed/"*.md "$FERRET_DIR/docs/dev-memory/" 2>/dev/null || true
    fi
fi

# ── Pick GPU ──────────────────────────────────────────────────────────────
PICKGPU="$FERRET_DIR/pick_gpu.sh"
if [[ -x "$PICKGPU" ]]; then
    # pick_gpu.sh prints `export CUDA_VISIBLE_DEVICES=...` — eval so it
    # becomes part of OUR env, then claude inherits it.
    eval "$("$PICKGPU")" || true
fi

# ── Required env for the mainthread ───────────────────────────────────────
export FERRET_WORKSPACE="workspace$N"
export FERRET_ROOT="$FERRET_DIR"
# `python -m ferret.state` / `ferret.profile` need ferret's parent dir on
# PYTHONPATH so the `ferret` package is importable, even though the
# mainthread's cwd is `ferret/` itself. Without this, every subagent
# Bash that calls a ferret CLI fails with ModuleNotFoundError.
export PYTHONPATH="$(dirname "$FERRET_DIR"):${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp/$USER}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# ── Sanity-check claude binary ────────────────────────────────────────────
if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: 'claude' CLI not found on PATH." >&2
    echo "       Install Claude Code first." >&2
    exit 127
fi

echo "ferret root         : $FERRET_DIR"
echo "FERRET_WORKSPACE    : $FERRET_WORKSPACE"
echo "FERRET_ROOT         : $FERRET_ROOT"
echo "PYTHONPATH          : $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(unset — pick_gpu did not run)}"
echo "TMPDIR              : $TMPDIR"
echo "claude binary       : $(command -v claude)"
echo

cd "$FERRET_DIR"

# ── Default standing goal (overridable via --goal "<text>") ───────────────
# This is injected via --append-system-prompt so the mainthread sees it on
# every turn — not just the first one. The goal also gets echoed as the
# leading `/goal` line in the seed prompt for the slash-command channel.
#
# The default goal is rendered from task.yaml by ferret.cc_goal so it
# contains the concrete SOTA name + per-config target ratios — not an
# abstract "iterate until done". This gives the session-scoped Stop hook
# (installed by /goal) a numerical success condition.
if [[ -z "$GOAL_TEXT" ]]; then
    if GOAL_TEXT=$(python3 -m ferret.cc_goal "$WS" --label "$FERRET_WORKSPACE" 2>/dev/null); then
        : # rendered cleanly
    else
        # Fallback if cc_goal.py crashes for any reason — never block launch.
        GOAL_TEXT="Drive $FERRET_WORKSPACE's task.yaml to completion. Iterate write → compile → benchmark → commit + tag → reviewer continuously until \`python3 -m ferret.state $FERRET_WORKSPACE $FERRET_WORKSPACE/task.yaml\` reports advance? True AND every per-config row shows the ✓ marker. See CLAUDE.md §6.5 for stop conditions."
    fi
fi
EFFECTIVE_GOAL="$GOAL_TEXT"

echo "goal                : $EFFECTIVE_GOAL"
echo

# Assemble the claude command. --append-system-prompt makes the goal
# system-level (survives every iteration); we also prefix the seed with
# /goal so the slash-command skill registers it where applicable.
CLAUDE_ARGS=(
    --append-system-prompt "STANDING GOAL: $EFFECTIVE_GOAL"
)

if [[ "$PRINT_ONLY" -eq 1 ]]; then
    if [[ -n "$SEED_PROMPT" ]]; then
        echo "WOULD EXEC: claude ${CLAUDE_ARGS[*]} -p \"/goal $EFFECTIVE_GOAL"$'\n\n'"$SEED_PROMPT\""
    else
        echo "WOULD EXEC: claude ${CLAUDE_ARGS[*]}  (interactive)"
        echo "  (will prompt mainthread to read CLAUDE.md and act on goal)"
    fi
    exit 0
fi

if [[ -n "$SEED_PROMPT" ]]; then
    exec claude "${CLAUDE_ARGS[@]}" -p "/goal $EFFECTIVE_GOAL"$'\n\n'"$SEED_PROMPT"
else
    exec claude "${CLAUDE_ARGS[@]}"
fi
