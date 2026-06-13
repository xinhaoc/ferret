#!/usr/bin/env bash
# remote_run.sh — run a compile/benchmark command on a REMOTE GPU host over
# ssh+rsync, so the CC-mode ferret mainthread can do its OWN GPU work without
# bouncing back to a Mirage session. Transparent LOCAL fallback when
# FERRET_REMOTE_HOST is unset. (Same per-call rsync idea as the API-mode
# api/remote.py, but a bash CLI the Claude-Code mainthread invokes directly.)
#
#   bash "$FERRET_ROOT/scripts/remote_run.sh" 'cd "$FERRET_WORKSPACE" && nvcc … -o kernel && ./kernel'
#
# Combine compile+run in ONE call → one rsync round-trip. The remote command's
# STDOUT (the KERNEL_RESULT / KERNEL_RESULT_REFERENCE lines) is forwarded to this
# script's stdout; all rsync/diagnostic chatter goes to STDERR, so the
# mainthread's KERNEL_RESULT parsing stays clean. Exit code = the remote
# command's exit code.
#
# Env (exported by scripts/cc-run.sh / the launcher):
#   FERRET_REMOTE_HOST          ssh alias w/ ControlMaster. UNSET ⇒ run locally.
#   FERRET_ROOT                 ferret dir — MUST be the SAME absolute path on the remote.
#   FERRET_WORKSPACE            workspace name (e.g. workspace3); rsync'd every call.
#   FERRET_REMOTE_CUDA_DEVICES  GPU index on the REMOTE (default 0; the local
#                               pick_gpu.sh choice is meaningless on the remote).
#   FERRET_REMOTE_ENV           file sourced on the remote before the cmd
#                               (default: $FERRET_ROOT/.env).
# Remote prereqs: ferret repo at the SAME abs path with resources/ staged, nvcc +
# a working GPU, and passwordless ssh (ControlMaster) to FERRET_REMOTE_HOST.
set -u -o pipefail
CMD="${1:?usage: remote_run.sh \"<shell command>\"}"
HOST="${FERRET_REMOTE_HOST:-}"
ROOT="${FERRET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WS="${FERRET_WORKSPACE:-workspace}"

# ── transparent LOCAL fallback (no remote host configured) ──
[ -z "$HOST" ] && exec bash -c "$CMD"

WS_ABS="$ROOT/$WS"
CM="$HOME/.ssh/cm_%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=600 "$HOST")
RE="ssh -o ControlPath=$CM"
DEV="${FERRET_REMOTE_CUDA_DEVICES:-0}"
ENVF="${FERRET_REMOTE_ENV:-$ROOT/.env}"
TO="${FERRET_REMOTE_TIMEOUT:-300}"   # remote hard timeout (s) on the compile+benchmark

# 1. push the fresh workspace (kernel.cu) → remote at the SAME abs path. The
#    workspace's own .git stays local (the mainthread tags versions locally).
"${SSH[@]}" "mkdir -p '$WS_ABS'" >/dev/null 2>&1
if ! rsync -az --delete --exclude='.git/' -e "$RE" "$WS_ABS/" "$HOST:$WS_ABS/" >&2; then
  echo "[remote_run] rsync-push FAILED → $HOST:$WS_ABS" >&2; exit 3
fi

# 2. run on the remote. Piped via `bash -s` so quotes/`$FERRET_WORKSPACE` inside
#    CMD survive verbatim to the remote (single-pass heredoc expansion: $CMD is
#    inserted as text; its inner $vars expand on the REMOTE, where we export them).
# The inner `timeout … bash -s <<'FERRET_REMOTE_CMD'` runs $CMD under a REMOTE
# hard timeout (so a hung nvcc/benchmark self-kills, not just on ssh teardown);
# the QUOTED inner heredoc feeds $CMD verbatim so its `$FERRET_WORKSPACE`/quotes
# expand on the remote (where we exported the env). source redirects stdout too
# so a chatty .env can't pollute the KERNEL_RESULT lines (Codex).
REMOTE_SCRIPT=$(cat <<EOF
cd '$ROOT' 2>/dev/null || { echo "[remote_run] ferret root '$ROOT' missing on remote" >&2; exit 4; }
export FERRET_ROOT='$ROOT' FERRET_WORKSPACE='$WS' CUDA_VISIBLE_DEVICES='$DEV'
[ -f '$ENVF' ] && source '$ENVF' >/dev/null 2>&1
timeout --signal=TERM --kill-after=5s '$TO' bash -s <<'FERRET_REMOTE_CMD'
$CMD
FERRET_REMOTE_CMD
EOF
)
printf '%s' "$REMOTE_SCRIPT" | "${SSH[@]}" "bash -s"; rc=${PIPESTATUS[1]}

# 3. pull the workspace back (compiled binary + any output artifacts).
rsync -az --exclude='.git/' -e "$RE" "$HOST:$WS_ABS/" "$WS_ABS/" >/dev/null 2>&1 \
  || echo "[remote_run] rsync-pull warn (run may have failed before output)" >&2
exit "$rc"
