#!/usr/bin/env bash
# update_kernelwiki.sh — keep the KernelWiki submodule (resources/kernelwiki) fresh.
#
# Two update channels (run both by default):
#   (A) UPSTREAM SYNC  — fast-forward the submodule to mit-han-lab/KernelWiki's
#       latest master. Clean + shareable: the new commit is a real upstream commit,
#       so the parent-ferret gitlink can be advanced and pushed. Needs network.
#   (B) LOCAL REFRESH  — run KernelWiki's own ingest pipeline (gh-search newly
#       merged kernel PRs since the cutoff → regenerate pages/indices). Grows the
#       corpus between upstream releases. Needs `gh` (authed). Produces LOCAL-ONLY
#       submodule commits (origin is mit-han-lab = read-only for us) — to SHARE
#       them, push the submodule to a fork and repoint the URL (see NOTE at end).
#
# The OFFLINE READ path (scripts/query.py / get_page.py that ferret uses at
# planner/iterator time) is never touched by this script and keeps working even
# if both channels fail. Safe to run from cron.
#
# Usage:
#   scripts/update_kernelwiki.sh                  # A + B (B auto-skips if gh absent)
#   scripts/update_kernelwiki.sh --upstream-only  # A only (zero local-dep)
#   scripts/update_kernelwiki.sh --refresh-only    # B only
#   scripts/update_kernelwiki.sh --refresh-only --repos vllm,sglang   # scope B
#   scripts/update_kernelwiki.sh --commit-pointer  # also `git add`+commit the
#                                                  #   advanced gitlink in ferret
set -uo pipefail

FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KW="$FERRET_DIR/resources/kernelwiki"
LOG_DIR="$FERRET_DIR/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/kernelwiki-update.log"
LOCK="$LOG_DIR/.kernelwiki-update.lock"
UPSTREAM_BRANCH="master"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- arg parse ---
DO_UPSTREAM=1; DO_REFRESH=1; COMMIT_POINTER=0; REPOS=""
while [ $# -gt 0 ]; do case "$1" in
  --upstream-only) DO_REFRESH=0;;
  --refresh-only)  DO_UPSTREAM=0;;
  --commit-pointer) COMMIT_POINTER=1;;
  --repos) REPOS="${2:?--repos requires a value, e.g. --repos vllm,sglang}"; shift;;
  -h|--help) sed -n '2,30p' "$0"; exit 0;;
  *) log "WARN: unknown arg '$1' (ignored)";;
esac; shift; done

# --- preconditions ---
[ -d "$KW/.git" ] || [ -f "$KW/.git" ] || { log "FATAL: $KW is not a submodule checkout. Run: git -C '$FERRET_DIR' submodule update --init resources/kernelwiki"; exit 1; }
[ -f "$KW/scripts/query.py" ] || { log "FATAL: $KW/scripts/query.py missing — submodule not populated."; exit 1; }

# --- single-flight lock (cron-safe) ---
exec 9>"$LOCK"
if ! flock -n 9; then log "another update_kernelwiki.sh is running; exiting."; exit 0; fi

log "=== KernelWiki update start (upstream=$DO_UPSTREAM refresh=$DO_REFRESH repos='${REPOS:-all}') ==="
before_head="$(git -C "$KW" rev-parse --short HEAD 2>/dev/null)"
log "submodule HEAD before: $before_head"

# ============================ (A) UPSTREAM SYNC ============================
if [ "$DO_UPSTREAM" = 1 ]; then
  log "[A] fetching upstream origin/$UPSTREAM_BRANCH ..."
  if timeout 120 git -C "$KW" fetch --quiet origin "$UPSTREAM_BRANCH" 2>>"$LOG"; then
    # fast-forward only — never create a merge/diverge here.
    if git -C "$KW" merge --ff-only "origin/$UPSTREAM_BRANCH" >>"$LOG" 2>&1; then
      after="$(git -C "$KW" rev-parse --short HEAD)"
      if [ "$after" != "$before_head" ]; then
        log "[A] fast-forwarded $before_head -> $after"
      else
        log "[A] already up to date ($after)"
      fi
    else
      log "[A] WARN: not fast-forwardable (local commits diverge from upstream — likely from a prior --refresh). Skipping FF. Reconcile manually or push to a fork."
    fi
  else
    log "[A] WARN: upstream fetch failed (network?). Offline read path unaffected; continuing."
  fi
fi

# ============================ (B) LOCAL REFRESH ============================
if [ "$DO_REFRESH" = 1 ]; then
  if ! command -v gh >/dev/null 2>&1; then
    log "[B] SKIP: 'gh' (GitHub CLI) not installed — the refresh pipeline needs it for 'gh search'."
    log "[B]       install: see https://github.com/cli/cli#installation ; then 'gh auth login'."
  elif ! gh auth status >/dev/null 2>&1; then
    log "[B] SKIP: 'gh' present but not authed — run 'gh auth login' first."
  else
    cutoff="$(date +%F)"
    repo_arg=(); [ -n "$REPOS" ] && repo_arg=(--repos "$REPOS")
    log "[B] refresh ledger (cutoff=$cutoff ${REPOS:+repos=$REPOS}) ..."
    ( cd "$KW" \
      && python3 scripts/refresh_candidate_ledger.py --cutoff "$cutoff" ${repo_arg[@]+"${repo_arg[@]}"} \
      && python3 scripts/generate-pr-pages.py --all \
      && python3 scripts/fetch_pr_diff.py --all \
      && python3 scripts/generate-indices.py ) >>"$LOG" 2>&1 \
      && log "[B] refresh pipeline OK" \
      || log "[B] WARN: refresh pipeline returned non-zero (see $LOG)"
  fi
fi

# ============================ VALIDATE (offline) ===========================
log "[V] validating corpus (offline) ..."
( cd "$KW" && python3 scripts/validate.py ) >>"$LOG" 2>&1 \
  && log "[V] validate OK" \
  || log "[V] WARN: validate.py reported issues (see $LOG)"

# ============================ REPORT =======================================
after_head="$(git -C "$KW" rev-parse --short HEAD 2>/dev/null)"
kpages="$(ls "$KW"/wiki/kernels/ 2>/dev/null | wc -l | tr -d ' ')"
dirty="$(git -C "$KW" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
log "submodule HEAD after: $after_head  (kernel pages: $kpages, uncommitted submodule changes: $dirty)"

# parent-ferret gitlink state
pointer_state="$(git -C "$FERRET_DIR" status --porcelain resources/kernelwiki 2>/dev/null)"
if [ -n "$pointer_state" ]; then
  log "NOTE: ferret gitlink for resources/kernelwiki changed."
  if [ "$COMMIT_POINTER" = 1 ] && [ "$dirty" = 0 ]; then
    git -C "$FERRET_DIR" add resources/kernelwiki \
      && git -C "$FERRET_DIR" commit -m "chore(kernelwiki): bump submodule to $after_head" >>"$LOG" 2>&1 \
      && log "committed gitlink bump ($before_head -> $after_head)" \
      || log "WARN: gitlink commit failed (see $LOG)"
  else
    log "      to record it:  git -C '$FERRET_DIR' add resources/kernelwiki && git -C '$FERRET_DIR' commit -m 'chore(kernelwiki): bump submodule'"
  fi
fi
if [ "$dirty" != 0 ]; then
  log "NOTE: submodule has $dirty uncommitted changes (from --refresh). origin is mit-han-lab (read-only)."
  log "      to SHARE refreshed content: fork KernelWiki, 'git -C $KW remote set-url origin <fork>',"
  log "      commit+push in the submodule, then bump the ferret gitlink + update .gitmodules url."
fi
log "=== KernelWiki update done ==="
