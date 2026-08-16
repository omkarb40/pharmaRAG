"""
PharmaRAG report figures.

Run:  python make_figures.py
Out:  ./figures/fig1_..png ... fig5_..png   (300 dpi, light theme, print-safe)

Figures 3 and 4 use locked/hardcoded results and run as-is (see report_analysis/
phase3_discrepancy_table.csv -- fig5's hardcoded latency numbers do NOT match the
extracted aggregate_metrics_*.json values for validation and baseline retrieval;
left unedited per instruction not to touch fig3/4/5 without sign-off).
Figure 2 uses real per-category recall from report_analysis/output/C1_recall_by_category.csv.
Figure 6 (confidence distribution) has been removed: the only audit log in the repo
(logs/audit_20260422.jsonl) has 10 records, which cannot support a distribution, and
the original figure6() read the wrong field names anyway ("confidence"/"decision" vs
the schema's real "confidence_score"/"confidence_decision").
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no display needed; safe on macOS
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

OUTDIR = "figures"

# Print-safe palette. Colour-blind friendly (Okabe-Ito derived).
C_NEUTRAL = "#4C4C4C"
C_PRIMARY = "#0072B2"
C_ACCENT  = "#D55E00"
C_GREEN   = "#009E73"
C_GREY    = "#BBBBBB"

CONFIG_COLORS = {
    "N (no agents)":  "#CC79A7",
    "Baseline":       "#56B4E9",
    "A: expansion":   "#E69F00",
    "B: ctx embed":   "#009E73",
    "C: rerank":      "#0072B2",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------
# FIGURE 2 -- Per-category Recall@5 across configurations
# --------------------------------------------------------------------------
# REAL values, extracted from evaluation/results/full_pipeline_results_{config}.json
# (canonical files, verified against report_analysis/phase3_discrepancy_table.csv),
# averaged over answerable queries per category. Source: report_analysis/output/
# C1_recall_by_category.csv. Config order preserved: N, Baseline, A, B, C.

PER_CATEGORY_RECALL = {
    # category            N       Baseline  A       B       C
    "dosing":            [0.5000, 0.5000, 0.5000, 0.6667, 0.5000],
    "contraindications": [0.2857, 0.4286, 0.4286, 0.4286, 0.7143],
    "adverse_reactions": [0.7500, 0.7500, 0.7500, 0.8750, 0.7500],
    "warnings":          [0.5000, 0.8750, 0.8750, 0.8750, 1.0000],
    "indications":       [0.1667, 0.3333, 0.3333, 0.3333, 0.8333],
    "interactions":      [0.7500, 0.7500, 0.7500, 0.7500, 1.0000],
    "populations":       [0.8333, 0.8333, 0.8333, 0.8333, 1.0000],
    "patient_style":     [0.8333, 0.8333, 1.0000, 0.8333, 0.8333],
    "multi_drug":        [0.7143, 1.0000, 1.0000, 1.0000, 1.0000],
}
PLACEHOLDER_WARNING = False   # real numbers are in


def figure2():
    cfg_names = list(CONFIG_COLORS.keys())
    cats = sorted(PER_CATEGORY_RECALL, key=lambda c: PER_CATEGORY_RECALL[c][-1])
    y = np.arange(len(cats))
    n = len(cfg_names)
    h = 0.78 / n

    fig, ax = plt.subplots(figsize=(10, 0.52 * len(cats) + 2.4))

    # highlight the weakest row behind the bars
    ax.axhspan(-0.5, 0.5, color="#F2F2F2", zorder=0)

    for j, cfg in enumerate(cfg_names):
        vals = [PER_CATEGORY_RECALL[c][j] for c in cats]
        offset = (j - (n - 1) / 2) * h
        ax.barh(y + offset, vals, height=h * 0.90,
                color=CONFIG_COLORS[cfg], label=cfg, edgecolor="none", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in cats])
    ax.set_ylim(-0.6, len(cats) - 0.4)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Recall@5")
    ax.grid(axis="y", visible=False)

    worst = cats[0]
    worst_c = PER_CATEGORY_RECALL[worst][-1]
    # label sits in clear whitespace to the right of the (short) weakest bars
    ax.text(min(worst_c + 0.06, 0.70), 0, "weakest category", va="center",
            ha="left", fontsize=9, style="italic", color=C_NEUTRAL, zorder=4)

    ax.set_title("Recall@5 by query category across pipeline configurations",
                 pad=26)
    ax.legend(ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.4, handlelength=1.4)

    if PLACEHOLDER_WARNING:
        fig.text(0.5, -0.02, "PLACEHOLDER DATA: replace via Appendix B block 3",
                 ha="center", fontsize=8.5, color=C_ACCENT, weight="bold")
    save(fig, "fig2_recall_by_category.png")


# --------------------------------------------------------------------------
# FIGURE 3 -- Groundedness vs unsafe emission  (the money figure)
# --------------------------------------------------------------------------

def figure3():
    configs = ["N\n(no agents)", "Baseline", "C\n(final)"]
    groundedness = [0.945, 0.960, 0.932]
    unsafe = [1.000, 0.185, 0.111]
    x = np.arange(len(configs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)

    # ---- left: groundedness ----
    lo, hi = min(groundedness), max(groundedness)
    ax1.axhspan(lo, hi, color=C_PRIMARY, alpha=0.12, zorder=0)
    ax1.bar(x, groundedness, width=0.55, color=C_PRIMARY,
            edgecolor="none", zorder=3)
    for xi, v in zip(x, groundedness):
        ax1.text(xi, v + 0.03, f"{v:.3f}", ha="center",
                 fontsize=10, weight="bold", zorder=4)
    ax1.set_title("Groundedness (answered queries)", pad=24)
    ax1.set_ylabel("Rate")
    ax1.text(0.5, 1.02, f"range {lo:.3f} to {hi:.3f}: flat across configurations",
             transform=ax1.transAxes, ha="center", fontsize=9,
             style="italic", color=C_NEUTRAL)

    # ---- right: unsafe emission ----
    ax2.bar(x, unsafe, width=0.55, color=C_ACCENT, edgecolor="none", zorder=3)
    for xi, v in zip(x, unsafe):
        ax2.text(xi, v + 0.03, f"{v:.3f}", ha="center",
                 fontsize=10, weight="bold", zorder=4)
    ax2.set_title("Unsafe emission rate (27 should-refuse queries)", pad=24)
    ax2.text(0.5, 1.02,
             "N = 1.000 is architectural: no refusal mechanism exists",
             transform=ax2.transAxes, ha="center", fontsize=9,
             style="italic", color=C_NEUTRAL)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylim(0, 1.14)          # CRITICAL: identical, never autoscale
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="x", visible=False)

    fig.suptitle("Groundedness does not track safety", fontsize=13.5, y=1.06)
    fig.tight_layout()
    save(fig, "fig3_groundedness_vs_safety.png")


# --------------------------------------------------------------------------
# FIGURE 4 -- Semantic weight sweep (RRF degeneracy)
# --------------------------------------------------------------------------

def figure4():
    w = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    recall = np.array([0.655, 0.707, 0.707, 0.724, 0.741, 0.776, 0.776])
    dense_only = 0.776

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    ax.axhline(dense_only, ls="--", lw=1.2, color=C_NEUTRAL, zorder=1)
    ax.text(0.415, dense_only + 0.006, "dense-only ceiling (0.776)",
            fontsize=9, color=C_NEUTRAL)

    ax.axvspan(0.9, 1.0, color=C_GREY, alpha=0.30, zorder=0)
    ax.text(0.95, 0.663, "plateau", ha="center", fontsize=9, color=C_NEUTRAL)

    ax.plot(w, recall, color=C_PRIMARY, lw=2.0, marker="o", ms=6, zorder=3)

    i = int(np.where(np.isclose(w, 0.6))[0][0])
    ax.scatter([w[i]], [recall[i]], s=150, facecolors="none",
               edgecolors=C_ACCENT, lw=2.0, zorder=4)
    ax.annotate("deployed configuration\n(0.6 / 0.4)",
                xy=(w[i], recall[i]), xytext=(0.60, 0.30),
                textcoords="axes fraction", fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.1))

    ax.set_xlabel("Semantic weight in RRF fusion  ($w_s$; lexical weight = $1 - w_s$)")
    ax.set_ylabel("Recall@5")
    ax.set_xlim(0.38, 1.02)
    ax.set_ylim(0.63, 0.80)
    ax.set_xticks(w)
    ax.set_title("Recall rises monotonically with semantic weight:\n"
                 "the lexical retriever contributes no novel candidates at any setting")
    save(fig, "fig4_semantic_weight_sweep.png")


# --------------------------------------------------------------------------
# FIGURE 5 -- Latency breakdown
# --------------------------------------------------------------------------

def figure5():
    stages = ["Query routing", "Retrieval + rerank", "Generation", "Validation"]
    colors = [C_GREEN, C_PRIMARY, C_ACCENT, "#CC79A7"]
    baseline = [2.15, 0.082, 10.3, 0.55]
    config_c = [2.15, 2.27, 10.3, 0.55]
    rows = ["Baseline\n(no rerank)", "Config C\n(rerank)"]
    data = np.array([baseline, config_c])

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    left = np.zeros(len(rows))
    for k, (stage, col) in enumerate(zip(stages, colors)):
        vals = data[:, k]
        ax.barh(rows, vals, left=left, color=col, label=stage,
                height=0.5, edgecolor="white", linewidth=0.8)
        for r in range(len(rows)):
            if vals[r] >= 0.9:
                ax.text(left[r] + vals[r] / 2, r, f"{vals[r]:.2f}s",
                        ha="center", va="center", fontsize=9,
                        color="white", weight="bold")
        left += vals

    for r in range(len(rows)):
        ax.text(left[r] + 0.25, r, f"total {left[r]:.2f}s",
                va="center", fontsize=9.5, weight="bold", color=C_NEUTRAL)

    ax.set_xlabel("Latency (seconds)")
    ax.set_xlim(0, left.max() * 1.16)
    ax.grid(axis="y", visible=False)
    ax.set_title("Generation dominates end-to-end latency; "
                 "reranking and validation are cheap by comparison")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig5_latency_breakdown.png")


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    figure5()
    print("\nDone. Figures in ./figures/")