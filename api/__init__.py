"""Ferret API-mode (motus) invocation path — PRESERVED for reference.

This is the older programmatic/API form of ferret (orchestrator/agents/main on
top of the external `motus` ReAct framework). It was removed from the active tree
in da135d5 ("Delete API-mode files") when ferret moved to the Claude-Code
mainthread + subagents design, and is restored HERE so it isn't lost.

NOT used by the Claude-Code agent. The agent (mainthread + subagents) should
IGNORE everything under api/ — it only uses cc-run.sh + the .claude/agents/
subagents. See CLAUDE.md.

To launch the API form (requires the external `motus` package installed):
    cd <parent of the ferret dir> && python -m ferret.api.main <task.yaml>
  or:
    api/scripts/run.sh <task.yaml> [--max-iterations N] [--no-detach]

Shared root modules (task_spec, state, profile, tools/profiler) are imported via
`..` (they live at the ferret root and are reused by the cc path); api-internal
modules (prompts, cost_tracker, orchestrator, agents, tools/compiler,
tools/doc_loader) are imported via `.`.
"""
