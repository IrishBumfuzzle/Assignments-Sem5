#!/usr/bin/env python
"""Generate the two report figures from saved run artifacts.

Data sources:
  outputs/C{1..5}/results.json          per-epoch history (loss, val metrics,
                                        peak_mem_mb, throughput)
  outputs/anlp_*.log                    per-epoch wall-clock (final "NNNs" field)

Outputs (300 dpi PNG, sized for a 6.3-inch single-column LaTeX layout):
  report/figures/training_curves.png    Fig 1: val loss / bit acc / seq acc
  report/figures/blt_tradeoff.png       Fig 2: peak GPU memory, epoch time,
                                        test accuracy (BLT tradeoff figure)

Usage:  .venv/bin/python scripts/make_report_figures.py
"""

import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {"C1": "C1 Base", "C2": "C2 RoPE", "C3": "C3 GQA",
          "C4": "C4 RMSNorm", "C5": "C5 BLT"}
COLORS = {"C1": "#1f77b4", "C2": "#d62728", "C3": "#2ca02c",
          "C4": "#ff7f0e", "C5": "#7b3294"}
# micro-batch size actually resident on the GPU (C1-C4: batch 8 x 2 grad-accum)
MICROBATCH = {"C1": 8, "C2": 8, "C3": 8, "C4": 8, "C5": 16}
LOGS = {
    "C1": "outputs/anlp_C1_v512_2686499.log",
    "C2": "outputs/anlp_C2_v512_2686500.log",
    "C3": "outputs/anlp_C3_v512_2686501.log",
    "C4": "outputs/anlp_C4_v512_2686502.log",
    "C5": "outputs/anlp_C5_fixed_2686327.log",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
})


def load_history(cfg):
    with open(os.path.join(ROOT, "outputs", cfg, "results.json")) as f:
        return json.load(f)


def epoch_seconds(cfg):
    pat = re.compile(r"\[C\d\] epoch \d+ \|.*peak \d+ MB \| (\d+)s")
    with open(os.path.join(ROOT, LOGS[cfg])) as f:
        return [int(m.group(1)) for m in (pat.search(line) for line in f) if m]


full = {c: load_history(c) for c in CONFIGS}
hist = {c: full[c]["history"] for c in CONFIGS}
med_sec = {c: statistics.median(epoch_seconds(c)) for c in CONFIGS}
peak_mb = {c: max(h["peak_mem_mb"]) for c, h in hist.items()}
per_line_mb = {c: peak_mb[c] / MICROBATCH[c] for c in CONFIGS}
sps = {c: float(sum(h["train_samples_per_sec"]) / len(h["epoch"]))
       for c, h in hist.items()}

# ---------------------------------------------------------------- Fig 1
fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.95))
for ax, key, title, scale in [
    (axes[0], "val_loss", "(a) Validation loss", "log"),
    (axes[1], "val_bit_accuracy", "(b) Validation bit accuracy", "linear"),
    (axes[2], "val_sequence_accuracy", "(c) Validation sequence accuracy", "linear"),
]:
    for c in CONFIGS:
        h = hist[c]
        lw = 2.0 if c == "C5" else 1.2
        ax.plot(h["epoch"], h[key], color=COLORS[c], lw=lw,
                label=LABELS[c], marker="none")
    if scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
axes[0].set_ylim(1e-4, 30)
axes[1].set_ylim(0.75, 1.0)
axes[2].set_ylim(0.0, 1.0)
for ax in axes:
    ax.grid(True, lw=0.4, alpha=0.5)
handles, labels_ = axes[1].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.035), handlelength=1.6, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "training_curves.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1))
x = range(len(CONFIGS))

# (a) peak GPU memory, annotated with per-line share
bars = axes[0].bar(x, [peak_mb[c] / 1000 for c in CONFIGS],
                   color=[COLORS[c] for c in CONFIGS], width=0.62)
for c, b in zip(CONFIGS, bars):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                 f"{peak_mb[c]/1000:.2f}\n({per_line_mb[c]:.0f} MB/line)",
                 ha="center", va="bottom", fontsize=6.8)
axes[0].set_title("(a) Peak GPU memory (train)")
axes[0].set_ylabel("GB (torch.cuda.max_memory_allocated)")
axes[0].set_ylim(0, 5.6)

# (b) median epoch wall-clock
bars = axes[1].bar(x, [med_sec[c] for c in CONFIGS],
                   color=[COLORS[c] for c in CONFIGS], width=0.62)
for c, b in zip(CONFIGS, bars):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                 f"{med_sec[c]:.0f}s  ({sps[c]:.0f} lines/s)",
                 ha="center", va="bottom", fontsize=6.8)
axes[1].set_title("(b) Epoch wall-clock (median)")
axes[1].set_ylabel("seconds")
axes[1].set_ylim(0, 175)

# (c) test sequence accuracy, bit accuracy annotated
seq = {c: full[c]["test"]["sequence_accuracy"] for c in CONFIGS}
bit = {c: full[c]["test"]["bit_accuracy"] for c in CONFIGS}
bars = axes[2].bar(x, [seq[c] for c in CONFIGS],
                   color=[COLORS[c] for c in CONFIGS], width=0.62)
for c, b in zip(CONFIGS, bars):
    axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                 f"{seq[c]:.3f}\n(bit {bit[c]:.4f})",
                 ha="center", va="bottom", fontsize=6.8)
axes[2].set_title("(c) Test sequence accuracy (greedy)")
axes[2].set_ylabel("exact full-line match, fraction")
axes[2].set_ylim(0, 1.18)

for ax in axes:
    ax.set_xticks(list(x))
    ax.set_xticklabels(CONFIGS)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "blt_tradeoff.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ summary
print(f"{'cfg':4} {'peak GB':>8} {'MB/line':>8} {'med s':>6} {'lines/s':>8}"
      f" {'seq acc':>8} {'bit acc':>8}")
for c in CONFIGS:
    print(f"{c:4} {peak_mb[c]/1000:8.2f} {per_line_mb[c]:8.1f} "
          f"{med_sec[c]:6.0f} {sps[c]:8.1f} {seq[c]:8.4f} {bit[c]:8.4f}")
print("figures written to", OUT)
