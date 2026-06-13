#!/bin/bash
# Pick the GPU with lowest memory usage (proxy for lowest occupancy).
# Usage:
#   source pick_gpu.sh        # sets CUDA_VISIBLE_DEVICES
#   eval $(./pick_gpu.sh)     # same effect
#   GPU=$(./pick_gpu.sh -q)   # just print the GPU index
#
# Skips GPUs using >50% memory (configurable via MAX_MEM_PCT).

MAX_MEM_PCT=${MAX_MEM_PCT:-50}

# Avoid GPUs already pinned by another running ferret cc-run / workspace session.
# Their CUDA_VISIBLE_DEVICES is baked at launch and NOT re-picked, so without this
# two concurrent dispatches both grab the lowest-mem GPU (all ~0 MiB at launch) and
# collide on benchmarks, corrupting each other's perf numbers. Self-maintaining:
# reflects live procs, no lease files. Set FERRET_NO_EXCLUDE=1 to disable.
EXCLUDE_GPUS=" "
if [ "${FERRET_NO_EXCLUDE:-0}" != "1" ]; then
    for _p in $(pgrep -u "$(whoami)" -f "cc-run|workspace[0-9]" 2>/dev/null); do
        _cvd=$(tr '\0' '\n' < "/proc/$_p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
        [ -n "$_cvd" ] && EXCLUDE_GPUS="$EXCLUDE_GPUS${_cvd//,/ } "
    done
fi

best_gpu=""
best_mem=999999

while IFS=, read -r idx used total; do
    idx=$(echo "$idx" | xargs)
    used=$(echo "$used" | xargs | sed 's/ MiB//')
    total=$(echo "$total" | xargs | sed 's/ MiB//')

    if [ "$total" -eq 0 ] 2>/dev/null; then continue; fi

    # Skip GPUs pinned by another running ferret session
    case "$EXCLUDE_GPUS" in *" $idx "*) continue;; esac

    pct=$((used * 100 / total))

    # Skip GPUs over threshold
    if [ "$pct" -gt "$MAX_MEM_PCT" ]; then continue; fi

    if [ "$used" -lt "$best_mem" ]; then
        best_mem=$used
        best_gpu=$idx
    fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

if [ -z "$best_gpu" ]; then
    echo "# WARNING: No GPU with <${MAX_MEM_PCT}% memory usage found" >&2
    # Fall back to least-used GPU regardless of threshold
    best_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | xargs)
fi

if [ "$1" = "-q" ]; then
    echo "$best_gpu"
else
    echo "export CUDA_VISIBLE_DEVICES=$best_gpu"
    echo "# Selected GPU $best_gpu (${best_mem} MiB used)" >&2
fi
