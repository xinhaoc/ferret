"""Build hero bar chart: ferret-produced paged-GQA kernel (v024) vs FlashInfer
on Qwen3-30B-A3B (TP=4 per-rank), 16 (Q, seq) configurations.

Numbers measured on B200, GPU 5, 2026-06-03, same physical run.
Methodology: 2-sec warmup + L2 flush + 300-iter median (kernel.cu and
baselines/paged-gqa-fused-qwen3/baseline.py use the exact same harness shape).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Microseconds median (lower = faster)
CONFIGS = [
    ("q1_seq128", 12.3, 15.5),
    ("q1_seq512", 11.3, 19.5),
    ("q1_seq4k",  17.4, 53.3),
    ("q1_seq32k", 37.9, 57.4),
    ("q2_seq128", 12.3, 20.3),
    ("q2_seq512", 13.3, 23.6),
    ("q2_seq4k",  23.6, 59.5),
    ("q2_seq32k", 52.3, 63.5),
    ("q3_seq128", 12.3, 34.2),
    ("q3_seq512", 16.4, 42.0),
    ("q3_seq4k",  25.6, 91.1),
    ("q3_seq32k", 62.5, 93.2),
    ("q4_seq128", 12.3, 20.1),
    ("q4_seq512", 16.4, 27.6),
    ("q4_seq4k",  29.7, 90.1),
    ("q4_seq32k", 72.7, 93.2),
]

names = [c[0] for c in CONFIGS]
kernel_us = np.array([c[1] for c in CONFIGS])
fi_us = np.array([c[2] for c in CONFIGS])
speedup = fi_us / kernel_us

# Colour by Q value
q_colors = {"q1": "#4C72B0", "q2": "#55A868", "q3": "#C44E52", "q4": "#8172B2"}
colors = [q_colors[n.split("_")[0]] for n in names]

fig, ax = plt.subplots(figsize=(11, 4.3))
xs = np.arange(len(names))
bars = ax.bar(xs, speedup, color=colors, edgecolor="black", linewidth=0.4)

ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6, zorder=0)
ax.text(len(names) - 0.5, 1.04, "FlashInfer parity",
        ha="right", va="bottom", fontsize=8, alpha=0.7)

# Annotate each bar with the ratio
for x, s in zip(xs, speedup):
    ax.text(x, s + 0.05, f"{s:.2f}x", ha="center", va="bottom", fontsize=8)

ax.set_xticks(xs)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Speedup over FlashInfer", fontsize=10)
ax.set_title("ferret-produced paged-GQA kernel vs FlashInfer\n"
             "Qwen3-30B-A3B (NUM_QO_HEADS=8, NUM_KV_HEADS=1, HEAD_DIM=128) on B200, bf16",
             fontsize=11)
ax.set_ylim(0, max(speedup) * 1.15)

# Legend for Q value
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=q_colors[k], label=f"Q={k[1]}") for k in ["q1", "q2", "q3", "q4"]]
ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, ncols=4)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "paged_gqa_vs_flashinfer.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"min speedup: {speedup.min():.2f}x ({names[speedup.argmin()]})")
print(f"max speedup: {speedup.max():.2f}x ({names[speedup.argmax()]})")
print(f"geomean:     {np.exp(np.mean(np.log(speedup))):.2f}x")
