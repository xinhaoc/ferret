# `api/` — Ferret API-form (motus) invocation path (PRESERVED, not active)

This directory holds the **older programmatic / API form** of ferret — the
`motus`-based orchestrator/agents pipeline that predates the current Claude-Code
mainthread + subagents design. It was removed from the active tree in `da135d5`
("Delete API-mode files") and is **restored here so it isn't lost**, kept out of
the main tree so it doesn't clutter the Claude-Code agent's context.

> **The Claude-Code agent ignores this directory** (see top-level `CLAUDE.md`).
> The agent uses `cc-run.sh` + the `.claude/agents/` subagents. `api/` is purely
> for anyone who wants to drive ferret via the programmatic API form.

## Contents
- `main.py` — CLI entry (`python -m ferret.api.main <task.yaml>`).
- `orchestrator.py` — the 3-stage (build → optimize) run loop.
- `agents.py` — the `motus` ReAct agent wrapper.
- `prompts.py`, `cost_tracker.py` — prompts + token/cost accounting.
- `tools/compiler.py`, `tools/doc_loader.py` — API-mode tools.
- `scripts/run.sh` — workspace-reset + launch wrapper.

## Launch (requires the external `motus` package installed)
```bash
# from the PARENT directory of the ferret repo:
python -m ferret.api.main path/to/task.yaml [--max-iterations N]
# or:
api/scripts/run.sh path/to/task.yaml [--no-detach]
```

## Notes
- **Shared root modules** (`task_spec`, `state`, `profile`, `tools/profiler`)
  live at the ferret root and are imported via `..` (reused by the cc path).
  API-internal modules (`prompts`, `cost_tracker`, `orchestrator`, `agents`,
  `tools/compiler`, `tools/doc_loader`) are imported via `.`.
- This path has **not** been re-validated end-to-end after the move (it depends
  on `motus` + a GPU); the relative-import + `run.sh` path fixes are mechanical.
  If you revive it, expect to adjust the `motus` integration first.
