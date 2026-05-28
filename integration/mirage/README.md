# Mirage integration

This directory holds the **Mirage-side artifacts** ferret expects to be
deployed into the user's `~/mirage` checkout. Mirage's `.gitignore`
excludes `.claude/` (see `~/mirage/.gitignore` line 85), so we can't ship
the subagent file inside Mirage itself — ferret holds the canonical
template here, and teammates copy it locally.

## What's here

- `ferret-kernel-agent.md` — dispatcher subagent. When Mirage's claude
  thread needs a new or optimized CUDA kernel, it invokes this subagent;
  the subagent synthesizes a ferret `task.yaml`, picks a free workspace,
  launches `~/ferret/scripts/cc-run.sh`, monitors the run, and returns
  the delivered `workspace<N>/kernel.cuh` (Mirage-ready, written by
  ferret's `kernel-extractor`).

## Install

```bash
mkdir -p ~/mirage/.claude/agents
cp ~/ferret/integration/mirage/ferret-kernel-agent.md \
   ~/mirage/.claude/agents/ferret-kernel-agent.md
```

Mirage's claude thread picks the subagent up on next session start
because `.claude/agents/<name>.md` is one of the conventional locations.

Do **not** edit the file inside `~/mirage/.claude/agents/` — keep it as
a verbatim copy and edit the source here in ferret instead. Re-`cp`
after every ferret `git pull --ff-only`.

## When to update the source

If ferret's CLI changes (new flag in `cc-run.sh`, new env var, new
deliverable shape, etc.), update `ferret-kernel-agent.md` here in the
same commit. The README + the subagent file together are the contract
between Mirage and ferret.
