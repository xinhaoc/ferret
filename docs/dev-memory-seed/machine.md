# machine.md — host-specific facts (committed generic template)

> This is the generic committed seed. The live runtime copy at
> `docs/dev-memory/machine.md` is gitignored and is bootstrapped from this
> file on first launch, then appended to by the `memory-keeper` subagent
> with host-specific facts. Keep this template machine-agnostic: use
> `$MIRAGE_ROOT` / `$FERRET_ROOT` / `$USER` / `python3` placeholders, never
> a real username, host, absolute home path, or device UUID.

Edited only by the `memory-keeper` subagent. Append-only; updates go below
the original entry as `Updated YYYY-MM-DD:` blocks.

## Paths

- `MIRAGE_ROOT` points at the Mirage checkout (export it; defaults to
  `$HOME/mirage`). Mirage's public C++ kernel-launch API is under
  `$MIRAGE_ROOT/include/mirage/` — notably `kernel/`, `persistent_kernel/`,
  and `transpiler/`. The `codex-dispatcher` subagent reads these to verify
  a ferret-generated `kernel.cu` exposes a compatible `extern "C"` entry.
- `FERRET_ROOT` is the ferret checkout (defaults to `$HOME/ferret`). The
  mainthread is launched with cwd at this directory; `$FERRET_WORKSPACE`
  is a relative path under it (`workspace1`..`workspace8`).
- Python: use the host's `python3`. All Python helpers (`ferret.state`,
  `ferret.task_spec`, `ferret.profile`) are runnable as modules from the
  ferret root. To run them outside `$FERRET_ROOT`, put the parent of
  `$FERRET_ROOT` on `PYTHONPATH` (e.g.
  `PYTHONPATH=$(dirname $FERRET_ROOT) python3 -m ferret.state ...`).

## GPU selection (shared cluster)

- `eval $(./pick_gpu.sh)` before every benchmark / profile. It writes
  `export CUDA_VISIBLE_DEVICES=...` to stdout. Different invocations pick
  different GPUs, so measure your kernel AND its baseline in the same
  `pick_gpu` invocation (i.e. the same shell session) to keep the
  comparison honest.

## ncu / profiling

- On a shared cluster, ncu can fail with "Unknown error on device 0" when
  its default `/tmp` is read-only or full. Workaround:
  `export TMPDIR=/tmp/$USER` (the `ferret.profile` CLI sets this for you).
- Always use `python3 -m ferret.profile <workspace>` instead of
  hand-crafting ncu commands. The wrapper picks the GPU, sets TMPDIR, runs
  the standard 7-metric ncu invocation, parses CSV, and persists a
  `.profile_last.json` snapshot so the next run prints a delta line.

## Codex sub-agent (read-only, MCP)

- Codex is reached over the **MCP protocol**, not a CLI. The `codex` MCP
  server is configured in `$FERRET_ROOT/.mcp.json` (`{"command": "codex",
  "args": ["mcp-server"]}`); restart Claude Code to load it. The
  `codex-dispatcher` subagent calls the `mcp__codex__codex` /
  `mcp__codex__codex-reply` tools — never the retired `codex exec` CLI.
- Always dispatch with `sandbox: "read-only"` and
  `approval-policy: "never"`, and `cwd: $MIRAGE_ROOT` for grounding. Codex
  is used only for read-only Mirage-API / ABI verification of a generated
  `kernel.cu` / `kernel.cuh`.
- Codex on this MCP server has **no non-shell file-read path**, and we
  forbid shell exec — so it cannot read files from `cwd` on its own.
  Pre-feeding is mandatory: the dispatcher must `Read` and paste every
  file Codex needs (the kernel source AND the relevant Mirage-ABI header
  snippets) directly into the prompt. Citing a path alone gets Codex
  nothing.
- If the `mcp__codex__*` tools are absent (the MCP server didn't connect),
  the dispatcher degrades to `{"status": "codex_unavailable", "reason":
  ...}` and lets the reviewer record it — never crash the mainthread.

## Compute / binaries

- B200 (SM100a) nvcc default flags expected by ferret tasks:
  `-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lcuda -lcudart`.
  Do NOT use `-arch=sm_100a` (loses the `a` tier of tcgen05 instructions).
