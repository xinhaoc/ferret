# dev-memory — shared machine knowledge across workspaces

This directory is **gitignored**. It holds operational knowledge that
survives across `workspaceN/` resets but isn't part of the ferret source
tree. The initial content is bootstrapped from the tracked
`docs/dev-memory-seed/` directory by `scripts/cc-init.sh` /
`scripts/cc-run.sh` on first launch; the `memory-keeper` subagent then
appends to it at runtime. Every mainthread session reads this `INDEX.md`
first; the individual notes are loaded on demand.

## Files

| File | Purpose | Read by |
|------|---------|---------|
| `machine.md` | Static facts about the host: `MIRAGE_ROOT`, Python paths, `pick_gpu.sh` usage, the `TMPDIR=/tmp/$USER` ncu workaround, conda envs. | mainthread (always), iterator, planner, profiler, reviewer, codex-dispatcher, memory-keeper |
| `quirks.md` | Cluster / library version footguns discovered during runs (e.g. flashinfer 0.6.7 not wiring tcgen05 for MLA prefill, B200 wgmma broken). | mainthread (on demand), iterator, planner, profiler, reviewer, memory-keeper |
| `tips.md` | Agent-discovered dev tricks: SASS shortcut, useful nvcc flags, helpful one-liners. | mainthread (on demand), iterator, memory-keeper |

## Who can write here

**Only the `memory-keeper` subagent.** Everyone else reads but doesn't edit.
This rule is enforced by the subagent's `tools:` allowlist in
`.claude/agents/memory-keeper.md` — its filesystem write is scoped to
`docs/dev-memory/**`.

If a subagent or the mainthread finds a new machine-level fact that should
live here, it must `Task(memory-keeper, ...)` rather than editing directly.

## Editorial rules (for memory-keeper)

1. **Append, don't overwrite.** Conflicts get a new `Updated YYYY-MM-DD:`
   block underneath the existing one; the reader keeps both and prefers the
   newer one. This preserves history.
2. **Each entry is one factual paragraph.** No prose explanations of why
   ferret exists — that's in `CLAUDE.md`.
3. **Categorise correctly**:
   - `machine.md` — true across all tasks, host-specific.
   - `quirks.md` — true for a specific lib/cluster version, may need
     revisiting when versions change.
   - `tips.md` — agent-discovered hacks; lowest authority, freely prunable.
4. **Date every entry.** Format `<date> — <fact>`. Use ISO dates.
5. **Keep `INDEX.md` (this file) under ~80 lines.** Long content goes in
   the target file, not here.
