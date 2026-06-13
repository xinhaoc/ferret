# Keeping the KernelWiki submodule current for ferret

KernelWiki is vendored as the **`resources/kernelwiki` git submodule** (upstream
`https://github.com/mit-han-lab/KernelWiki`). Ferret queries the live submodule
working tree at planner/iterator time (see `.claude/skills/kernelwiki/SKILL.md`),
so a refreshed wiki is consumed on the **very next dispatch** — no ferret-side
cache, no rebuild.

## First-time / fresh clone

A fresh `git clone` of ferret leaves the submodule empty. Populate it once:
```bash
git -C "$FERRET_ROOT" submodule update --init resources/kernelwiki
```
The READ path (`scripts/query.py` / `get_page.py`) is fully OFFLINE after that.

## The automation: `scripts/update_kernelwiki.sh`

One script drives both update channels (safe to run from cron; the offline read
path keeps working even if both channels fail):

```bash
scripts/update_kernelwiki.sh                  # (A) upstream sync + (B) gh-ingest
scripts/update_kernelwiki.sh --upstream-only  # (A) only — zero local deps
scripts/update_kernelwiki.sh --refresh-only --repos vllm,sglang   # (B) scoped
scripts/update_kernelwiki.sh --commit-pointer # also commit the gitlink bump
```

- **(A) Upstream sync** — fast-forwards the submodule to mit-han-lab's latest
  `master`. Clean + shareable: the new commit is a real upstream commit, so the
  parent-ferret gitlink can be advanced and pushed. Needs network; degrades to a
  warning if offline.
- **(B) Local refresh** — runs KernelWiki's own ingest pipeline to pull
  newly-merged kernel PRs since the cutoff and regenerate pages/indices. Needs
  the GitHub CLI (`gh`, authed via `gh auth login`); auto-skips with an install
  hint if `gh` is missing. Produces **local-only** submodule commits (origin is
  read-only for us) — see "Sharing refreshed content" below.
- Always finishes with an offline `validate.py` and a page-count report, and
  tells you if the ferret gitlink moved.

Logs to `logs/kernelwiki-update.log`; single-flight `flock` makes it cron-safe.

## What (B) runs under the hood (manual equivalent)

```bash
cd "$FERRET_ROOT/resources/kernelwiki"
python3 scripts/refresh_candidate_ledger.py --cutoff $(date +%F)   # [--repos vllm,sglang]
python3 scripts/generate-pr-pages.py --all \
  && python3 scripts/fetch_pr_diff.py --all \
  && python3 scripts/generate-indices.py \
  && python3 scripts/validate.py        # expect "<N> files / 0 errors"
```
Add a **brand-new repo**: add a `slug -> owner/repo` entry to
`scripts/refresh_candidate_ledger.py::REPO_SLUG_TO_FULL`, create
`candidates/<slug>.yaml` (`repo:` / `keywords_used:` / `prs: []`), then run (B)
with `--repos <slug>`.

## Sharing refreshed content (channel B → pushable)

Channel-A commits are real upstream commits (pushable as-is). Channel-B commits
exist only locally because `origin` is mit-han-lab (read-only). To share them:
1. Fork KernelWiki, then in the submodule:
   `git -C resources/kernelwiki remote set-url origin <your-fork>`.
2. Commit + push the refreshed content inside the submodule.
3. Bump the ferret gitlink (`git add resources/kernelwiki`) **and** update the
   `.gitmodules` url to the fork, then commit in ferret.

## Cadence

Recommended **weekly cron** (`Sun 03:00`) of `update_kernelwiki.sh` — DSv3 SOTA
(DeepGEMM/CUTLASS/vLLM FP8) moves on a multi-week timescale and ferret only READS
the wiki, so a stale-by-days corpus breaks nothing. Don't run channel B before
the read/query path is actually in use (premature `gh search` burns rate-limit
for no consumer).
