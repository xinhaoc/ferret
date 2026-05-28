# machine.md — host-specific facts

Edited only by the `memory-keeper` subagent. Append-only; updates go below
the original entry as `Updated YYYY-MM-DD:` blocks.

## Paths

- 2026-05-28 — `MIRAGE_ROOT=~/mirage` (i.e. `/home/$USER/mirage`). Mirage's
  public C++ kernel-launch API is under `$MIRAGE_ROOT/include/mirage/` —
  notably `kernel/`, `persistent_kernel/`, and `transpiler/`. The
  `codex-dispatcher` subagent reads these to verify ferret-generated
  `kernel.cu` exposes a compatible `extern "C"` entry.
- 2026-05-28 — ferret root: `~/ferret` (`/home/$USER/ferret`). The
  mainthread is launched with cwd at this directory; `$FERRET_WORKSPACE`
  is a relative path under it (`workspace1`..`workspace8`).
- 2026-05-28 — Python: `scripts/run.sh` references `/home/xinhaoc/miniconda3/bin/python3`
  for the legacy motus path. For the Claude-Code path use the host's
  `python3` (3.12). All Python helpers (`ferret.state`, `ferret.task_spec`,
  `ferret.profile`) are runnable as modules from the ferret root.

## GPU selection (catalyst-fleet1 shared cluster)

- 2026-05-28 — `eval $(./pick_gpu.sh)` before every benchmark / profile.
  It writes `export CUDA_VISIBLE_DEVICES=...` to stdout. Different
  invocations pick different GPUs, so measure your kernel AND its
  baseline in the same `pick_gpu` invocation (i.e. the same shell
  session) to keep the comparison honest.

## ncu / profiling

- 2026-05-28 — On this cluster ncu fails with "Unknown error on device 0"
  when its default `/tmp` is read-only or full. Workaround:
  `export TMPDIR=/tmp/$USER` (the `ferret.profile` CLI sets this for you).
  Source: AGENT_EVOLUTION.md item 10.
- 2026-05-28 — Always use `python3 -m ferret.profile <workspace>` instead
  of hand-crafting ncu commands. The wrapper picks the GPU, sets TMPDIR,
  runs the standard 7-metric ncu invocation, parses CSV, and persists a
  `.profile_last.json` snapshot so the next run prints a delta line.

## Codex sub-agent

- 2026-05-28 — `codex` CLI present at `~/.nvm/versions/node/v25.9.0/bin/codex`.
  The `codex-dispatcher` subagent shells out to it in
  `--sandbox read-only --ask-for-approval on-request` mode for Mirage-API
  verification work. Use `-C $MIRAGE_ROOT` so Codex reads the right repo.
  If the CLI is missing or login expired, the dispatcher must degrade to
  `{status: "codex_unavailable", reason: ...}` and let the reviewer
  record that — never crash the mainthread.

## Compute / binaries

- 2026-05-28 — B200 nvcc default flags expected by ferret tasks:
  `-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lcuda -lcudart`.
  Do NOT use `-arch=sm_100a` (loses the `a` tier of tcgen05 instructions).
