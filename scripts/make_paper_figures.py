"""
Paper figures, generated directly from the results files (no hand-entered numbers).
  python scripts/make_paper_figures.py      -> figures/paper/fig*.pdf|png + figure_data.json
Final configuration (C + F1 + F2) uses run 1 (evaluation/results/final_run1.json);
run 2 is the repeat-run file and is reported in the text as the replication.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "evaluation/results", ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

# ---- style: print-first, validated palette (dataviz reference slots) --------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#cfcdc7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
FULL_W = 6.9  # inches, two-column width

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)

# ---- data ---------------------------------------------------------------------------------
CFG = {
    "dev": [("N", "configN_noagents"), ("Baseline", "baseline"), ("A", "optA_expand"), ("B", "optB_context"),
            ("C", "optC_rerank"), ("C − guard", "optC_rerank_noGuard"), ("C − self-refusal", "optC_rerank_noSelfRefusal"),
            ("C + check", "optC_rerank_intent"), ("C + F1", "optC_rerank_respectAbstain"),
            ("C + check + F1", "optC_rerank_intent_respectAbstain"), ("C + F1 + F2", "FINAL_RUN1")],
    "heldout": [("C", "optC_rerank__heldout_v1"), ("C + check", "optC_rerank_intent__heldout_v1"),
                ("C + F1", "optC_rerank_respectAbstain__heldout_v1"), ("C + check + F1", "optC_rerank_intent_respectAbstain__heldout_v1")],
}

def load(tag):
    f = RES / ("final_run1.json" if tag == "FINAL_RUN1" else f"full_pipeline_results_{tag}.json")
    return json.load(open(f))

def summarise(R):
    d = {}
    for grp in ("refuse", "answer"):
        X = [x for x in R if x["expected_behavior"] == grp]
        c = dict(n=len(X), answered=0, override_released=0, ref_model=0, ref_guard=0, ref_check=0)
        for x in X:
            if x["confidence_decision"] == "INSUFFICIENT_EVIDENCE":
                src = x.get("refusal_source") or ("llm_self" if x.get("llm_self_refused") else "guard")
                if x.get("abstention_override"): src = "llm_self"
                c[{"llm_self": "ref_model", "guard": "ref_guard", "intent_check": "ref_check"}[src]] += 1
            elif x.get("llm_self_refused"):
                c["override_released"] += 1
            else:
                c["answered"] += 1
        d[grp] = c
    ans = [x for x in R if x["confidence_decision"] != "INSUFFICIENT_EVIDENCE" and x.get("total_sentences", 0) > 0]
    g = [x["groundedness_score"] for x in ans]
    d["groundedness"] = sum(g) / len(g) if g else None
    d["answered_n"] = len(ans)
    return d

DATA = {s: {n: summarise(load(t)) for n, t in L} for s, L in CFG.items()}
unsafe = lambda d: (d["refuse"]["answered"] + d["refuse"]["override_released"]) / d["refuse"]["n"]
helpful = lambda d: (d["answer"]["answered"] + d["answer"]["override_released"]) / d["answer"]["n"]
json.dump(DATA, open(OUT / "figure_data.json", "w"), indent=2)

# ---- Figure 1: architecture ---------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.45)); ax.set_xlim(0, 112); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, w, title, sub=None, edge=MUTED, y=19, h=9, fill="#ffffff", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", lw=0.9, ec=edge, fc=fill))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), title, ha="center", va="center", fontsize=6.4,
                color=INK, fontweight="bold" if bold else "normal", linespacing=1.0)
        if sub: ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center", va="center", fontsize=5.3, color=INK2)
    def arrow(p0, p1, color=MUTED, ls="-", lw=0.9, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5, lw=lw, color=color, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    y, h, ym = 19, 9, 23.5
    stages = [(0.5, 8, "Query", None, MUTED, False), (11, 14, "Query Router", "→ label section", BLUE, True),
              (27.5, 16.5, "Hybrid retrieval", "BM25 + dense · RRF", MUTED, False), (46.5, 11.5, "Rerank", "cross-encoder", MUTED, False),
              (60.5, 16.5, "Generator", "Gemma 3 12B (local)", INK2, False), (79.5, 10.5, "Evidence\nValidator", None, BLUE, True),
              (92.5, 10, "Refusal\nGuard", None, BLUE, True)]
    for x, w, t, sub, e, b in stages: box(x, w, t, sub, e, bold=b)
    for (x0, w0, *_), (x1, *__) in zip(stages[:-1], stages[1:]): arrow((x0 + w0 + 0.4, ym), (x1 - 0.4, ym))
    # decisions
    arrow((102.9, ym), (105.2, ym))
    for k, lab in enumerate(["ANSWER", "CAUTION", "REFUSE"]):
        ax.text(105.6, ym + 2.6 - 2.6 * k, lab, fontsize=5.9, color=INK2, va="center")
    # optional question-level safety check
    box(11, 26, "Safety-intent check", "optional · question only", AQUA, y=31.5, h=7, fill="#f3fbf8")
    arrow((4.5, y + h + 0.4), (10.6, 35))
    arrow((37.4, 35), (97.5, y + h + 0.5), color=AQUA, ls=(0, (3, 2)))
    ax.text(68, 37.2, "DECLINE → refuse before generation is released", fontsize=5.9, color=INK2, ha="center")
    # results-v1 override path and Fix 1
    ax.add_patch(FancyArrowPatch((68.75, y - 0.4), (97.5, y - 0.4), arrowstyle="-|>", mutation_scale=6.5, lw=1.0,
                                 color=ORANGE, connectionstyle="arc3,rad=0.32", shrinkA=0, shrinkB=0))
    ax.text(83.25, 9.6, "results-v1: the guard could release an answer the model had refused", fontsize=5.9, color=ORANGE, ha="center")
    ax.text(83.25, 6.9, "Fix 1: the model's refusal is final", fontsize=6.1, color=INK, ha="center", fontweight="bold")
    # audit log
    ax.add_patch(FancyBboxPatch((0.5, 0.6), 110.5, 3.2, boxstyle="round,pad=0.2,rounding_size=1", lw=0.6, ec=GRID, fc="#f7f6f3"))
    ax.text(55.75, 2.2, "JSONL audit log per request: routed section · retrieved chunks · per-sentence support · decision and reason · stage latency",
            fontsize=5.5, color=INK2, ha="center", va="center")
    save(fig, "fig1_architecture")

# ---- Figure 2: grounded but unsafe ----------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    pts = {"dev": ["N", "Baseline", "B", "C", "C − self-refusal", "C + check", "C + F1 + F2"], "heldout": ["C", "C + F1"]}
    off = {("dev", "N"): (0.004, 0, "left"), ("dev", "C − self-refusal"): (-0.004, 0, "right"), ("dev", "Baseline"): (0.004, 2.5, "left"),
           ("dev", "B"): (0.004, 0, "left"), ("dev", "C"): (0.004, -2.5, "left"), ("dev", "C + check"): (-0.004, 0, "right"),
           ("dev", "C + F1 + F2"): (-0.004, 0, "right"), ("heldout", "C"): (0.004, 0, "left"), ("heldout", "C + F1"): (0.004, 0, "left")}
    for s, mk, col in (("dev", "o", BLUE), ("heldout", "s", ORANGE)):
        for n in pts[s]:
            d = DATA[s][n]; x, y = d["groundedness"], 100 * unsafe(d)
            ax.scatter(x, y, s=28, marker=mk, color=col, ec="white", lw=0.8, zorder=3)
            dx, dy, ha = off[(s, n)]
            ax.text(x + dx, y + dy, ("held-out " if s == "heldout" else "") + n, fontsize=6.3, color=INK2, ha=ha, va="center")
    ax.scatter([], [], s=24, marker="o", color=BLUE, label="Development set (82 q.)")
    ax.scatter([], [], s=24, marker="s", color=ORANGE, label="Held-out twin pairs (52 q.)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=1.0)
    ax.set_xlim(0.84, 1.0); ax.set_ylim(-6, 106)
    ax.set_xlabel("Groundedness of released answers"); ax.set_ylabel("Should-decline questions answered (%)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig2_grounded_but_unsafe")

# ---- Figure 4: safety-helpfulness trade-off, one point per set -------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for n in ["C", "C + F1", "C + check"]:
        d0, d1 = DATA["dev"][n], DATA["heldout"][n]
        ax.annotate("", xy=(100 * helpful(d1), 100 * unsafe(d1)), xytext=(100 * helpful(d0), 100 * unsafe(d0)),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7, shrinkA=5, shrinkB=5), zorder=1)
    lab = {("dev", "N"): (-2.0, 0, "right"), ("dev", "Baseline"): (-2.0, 0, "right"), ("dev", "C"): (2.2, -1.8, "left"),
           ("dev", "C + F1"): (-2.0, 0, "right"), ("dev", "C + check"): (0, -5.5, "center"), ("dev", "C + F1 + F2"): (2.0, 0, "left"),
           ("heldout", "C"): (2.2, 0, "left"), ("heldout", "C + F1"): (2.2, 0, "left"), ("heldout", "C + check"): (0, 5.0, "center")}
    for (s, n), (dx, dy, ha) in lab.items():
        d = DATA[s][n]; x, y = 100 * helpful(d), 100 * unsafe(d)
        ax.scatter(x, y, s=30, marker="o" if s == "dev" else "s", color=BLUE if s == "dev" else ORANGE, ec="white", lw=0.8, zorder=3)
        ax.text(x + dx, y + dy, n, fontsize=6.4, color=INK2, ha=ha, va="center", zorder=4)
    ax.scatter([], [], marker="o", color=BLUE, s=24, label="Development set")
    ax.scatter([], [], marker="s", color=ORANGE, s=24, label="Held-out twin pairs")
    ax.plot([], [], color=MUTED, lw=0.8, label="same configuration, dev → held-out")
    ax.legend(loc="upper left", frameon=False, fontsize=6.4, handletextpad=0.3)
    ax.set_xlim(25, 108); ax.set_ylim(-9, 106)
    ax.set_xlabel("Answerable questions answered (%)"); ax.set_ylabel("Should-decline questions answered (%)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(106, -6, "better ↘", fontsize=6.4, color=INK2, ha="right", va="center")
    save(fig, "fig4_tradeoff")

# ---- Figure 3: who refuses ----------------------------------------------------------------
SEG = [("ref_model", "Refused by the model", BLUE), ("ref_check", "Refused by safety check", AQUA),
       ("ref_guard", "Refused by guard", VIOLET), ("override_released", "Model refused, guard released", ORANGE),
       ("answered", "Answered", NEUTRAL)]
def fig4():
    panels = [("dev", "refuse", "a  Development: should-decline (n = 27)"), ("dev", "answer", "b  Development: answerable (n = 55)"),
              ("heldout", "refuse", "c  Held-out: should-decline (n = 26)"), ("heldout", "answer", "d  Held-out: answerable twins (n = 26)")]
    order = {"dev": ["N", "Baseline", "C", "C − guard", "C − self-refusal", "C + check", "C + F1", "C + F1 + F2"],
             "heldout": ["C", "C + F1", "C + check", "C + check + F1"]}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.1), gridspec_kw=dict(height_ratios=[8, 4.6], hspace=0.55, wspace=0.08))
    for ax, (s, grp, title) in zip(axes.flat, panels):
        names = order[s][::-1]
        for i, n in enumerate(names):
            c = DATA[s][n][grp]; left = 0
            for key, _, col in SEG:
                v = c[key]
                if not v: continue
                ax.barh(i, v, left=left, height=0.66, color=col, ec="white", lw=1.0)
                if v >= 2:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=6.2,
                            color="white" if col in (BLUE, VIOLET, ORANGE) else INK)
                left += v
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if ax in axes[:, 0] else [""] * len(names))
        ax.set_xlim(0, DATA[s][names[0]][grp]["n"]); ax.set_title(title, loc="left", color=INK, pad=4)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_xlabel("questions", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
    fig.legend(handles, [l for _, l, _ in SEG], loc="lower center", ncol=5, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=1.2)
    save(fig, "fig3_refusal_attribution")

# ---- Appendix figure A1: safety check by category on the held-out set -----------------------
def figA1():
    I = json.load(open(RES / "intent_check_intent-policy-v2_heldout_v1.json"))
    names = {1: "1 Individual\ndose", 2: "2 Specific\npopulation", 3: "3 Stop or\nswitch", 4: "4 Exceed dose\nor misuse", 5: "5 Off-label\nuse"}
    cats = sorted({r["heldout_category"] for r in I})
    dec = [[r for r in I if r["heldout_category"] == c and r["expected_behavior"] == "refuse"] for c in cats]
    twn = [[r for r in I if r["heldout_category"] == c and r["expected_behavior"] == "answer"] for c in cats]
    a = [sum(r["decision"] == "DECLINE" for r in g) for g in dec]
    b = [sum(r["decision"] == "DECLINE" for r in g) for g in twn]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.7, 2.4)); w = 0.36
    xs = range(len(cats))
    for off, vals, grp, col, lab in ((-w / 2 - 0.01, a, dec, BLUE, "Should-decline question declined"),
                                     (w / 2 + 0.01, b, twn, ORANGE, "Answerable twin wrongly declined")):
        ax.bar([x + off for x in xs], [100 * v / len(g) for v, g in zip(vals, grp)], width=w, color=col, label=lab, zorder=3)
        for x, v, g in zip(xs, vals, grp):
            ax.text(x + off, 100 * v / len(g) + 2.5, f"{v}/{len(g)}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([names[c] for c in cats], fontsize=6.8)
    ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% of questions")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    save(fig, "figA1_check_by_category")

for f in (fig1, fig2, fig3, fig4, figA1): f()
print("figures written to", OUT.relative_to(ROOT), sorted(p.name for p in OUT.glob("*.p*")))
