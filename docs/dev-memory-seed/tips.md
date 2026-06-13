# tips.md — agent-discovered dev tricks (committed generic template)

> This is the generic committed seed. The live runtime copy at
> `docs/dev-memory/tips.md` is gitignored, bootstrapped from this file, and
> freely pruned/appended by the `memory-keeper` subagent. Keep tips here
> machine-agnostic (`$MIRAGE_ROOT` / `$FERRET_ROOT` placeholders, no
> usernames/hosts).

Lowest-authority knowledge: handy one-liners, useful nvcc flags, navigation
shortcuts. The `memory-keeper` subagent prunes freely here — if a tip is
unused for many sessions or contradicted by `quirks.md`, drop it.

- `ferret.state` commit-body parsing: the `python3 -m ferret.state` CLI
  scores a tagged commit only when the commit body contains the literal
  lines `KERNEL_RESULT {...}` and `KERNEL_RESULT_REFERENCE {...}` as valid
  JSON objects. The shorthand `TFLOPS: M1=.. M4=..` in the commit body is
  NOT parsed for non-`Q` config keys; only `Q<N>=` keys on a line starting
  with `TFLOPS:` are recognized. A commit using only the shorthand shows
  score 0.0 and never advances the stage gate. Always copy the harness's
  full `KERNEL_RESULT` / `KERNEL_RESULT_REFERENCE` JSON lines verbatim into
  the commit body.

- `nvcc` link flags when the calibration harness includes an in-process
  cuBLASLt reference: the harness `#include`s `<cublasLt.h>`, so it needs
  `-lcublasLt -lcublas` in addition to the §5 template's `-lcuda -lcudart`.
  The split-K device function itself has no cuBLASLt dependency — the
  requirement comes only from the standalone harness's reference path.
  Working line: `nvcc -gencode arch=compute_100a,code=sm_100a -O3
  -std=c++17 -lcuda -lcudart -lcublasLt -lcublas kernel.cu -o kernel`.
  (A reusable calibration seed lives under `$FERRET_ROOT/calib_scratch/`.)
