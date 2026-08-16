"""
PharmaRAG architecture diagram (Figure 1).

Run:  python make_architecture.py
Out:  figures/fig1_architecture.png   (300 dpi, light theme, print-safe)

Pure matplotlib, no graphviz dependency.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# palette (matches make_figures.py)
C_INGEST = "#D9D9D9"     # offline ingestion, greyed
C_RETR   = "#9ECAE8"     # retrieval stages
C_GEN    = "#F0C27A"     # generation
C_AGENT  = "#0072B2"     # the three agents, filled dark
C_OUT_OK = "#009E73"
C_OUT_CAU= "#E69F00"
C_OUT_NO = "#D55E00"
C_AUDIT  = "#EDEDED"
C_EDGE   = "#3A3A3A"
C_TEXT   = "#1A1A1A"

plt.rcParams.update({
    "font.size": 9,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, label, sub=None, fc=C_RETR, tc=C_TEXT,
        fontsize=9, weight="normal", radius=1.4):
    """Rounded box with centred label and optional smaller sub-label."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=1.1, edgecolor=C_EDGE, facecolor=fc, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fontsize, color=tc, weight=weight, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fontsize - 1.4, color=tc, alpha=0.85, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=tc, weight=weight, zorder=4)
    return (x, y, w, h)


def arrow(p1, p2, style="-|>", ls="-", color=C_EDGE, lw=1.3,
          rad=0.0, zorder=2):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, linestyle=ls,
                        color=color, linewidth=lw, mutation_scale=13,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=1, shrinkB=1, zorder=zorder)
    ax.add_patch(a)


def right(b):   return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):    return (b[0], b[1] + b[3] / 2)
def top(b):     return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b):  return (b[0] + b[2] / 2, b[1])


def band(y, h, label, color="#FAFAFA", label_x=3.0):
    ax.add_patch(Rectangle((1.5, y), 97, h, facecolor=color,
                           edgecolor="#DDDDDD", linewidth=0.9, zorder=0))
    ax.text(label_x, y + h - 2.4, label, fontsize=8.6, style="italic",
            color="#666666", ha="left", va="center", zorder=1)


# ==========================================================================
# BAND A -- offline ingestion
# ==========================================================================
band(78, 19, "OFFLINE INGESTION (run once per corpus build)")

BH, BY = 8.5, 82
a1 = box(4,    BY, 15.5, BH, "DailyMed SPL", "XML, SetID-pinned", fc=C_INGEST)
a2 = box(23.5, BY, 15.5, BH, "LOINC parse", "section metadata", fc=C_INGEST)
a3 = box(43,   BY, 15.5, BH, "Chunking", "500 tok / 50 overlap", fc=C_INGEST)
a4 = box(62.5, BY, 15.5, BH, "PubMedBERT", "768-d embeddings", fc=C_INGEST)
a5 = box(82,   BY, 14,   BH, "ChromaDB", "723 chunks", fc=C_INGEST)

for s, t in [(a1, a2), (a2, a3), (a3, a4), (a4, a5)]:
    arrow(right(s), left(t))

# ==========================================================================
# BAND B -- online retrieval
# ==========================================================================
band(45, 29, "ONLINE: RETRIEVAL")

q  = box(4,  56, 15.5, 8.5, "User query", fc="white")
r1 = box(23.5, 56, 15.5, 8.5, "Query Router", "query \u2192 SPL section",
         fc=C_AGENT, tc="white", weight="bold")

# parallel retrievers
d1 = box(43, 61.5, 15.5, 7.0, "Dense search", "cosine, ChromaDB", fc=C_RETR)
d2 = box(43, 47.5, 15.5, 7.0, "BM25", "lexical", fc=C_RETR)

fu = box(62.5, 54.5, 15.5, 8.5, "Weighted RRF", "k=60, 0.6 / 0.4", fc=C_RETR)
rr = box(82, 54.5, 14, 8.5, "Cross-encoder", "top-20 \u2192 top-5", fc=C_RETR)

arrow(right(q), left(r1))
arrow(right(r1), left(d1), rad=-0.18)
arrow(right(r1), left(d2), rad=0.18)
arrow(right(d1), left(fu), rad=0.18)
arrow(right(d2), left(fu), rad=-0.18)
arrow(right(fu), left(rr))

# index feeds dense search
arrow(bottom(a5), (89, 76.0), ls=(0, (4, 3)), style="-", color="#8A8A8A")
arrow((89, 76.0), (50.75, 76.0), ls=(0, (4, 3)), style="-", color="#8A8A8A")
arrow((50.75, 76.0), top(d1), ls=(0, (4, 3)), style="-|>", color="#8A8A8A")
ax.text(70, 77.3, "vector index", fontsize=8, color="#8A8A8A", ha="center")

# ==========================================================================
# BAND C -- generation and governance
# ==========================================================================
band(6, 33, "ONLINE: GENERATION AND GOVERNANCE", label_x=20.0)

gen = box(4, 27, 19, 9.0, "Gemma 3 12B", "Ollama, local, T=0", fc=C_GEN)
ev  = box(28, 27, 19, 9.0, "Evidence Validator",
          "per-sentence, \u03c4=0.35", fc=C_AGENT, tc="white", weight="bold")
gd  = box(52, 27, 19, 9.0, "Refusal Guard",
          "C = .25r + .55g + .20n", fc=C_AGENT, tc="white", weight="bold")

arrow(right(gen), left(ev))
arrow(right(ev), left(gd))

# rerank output wraps down into generation
arrow((89, 54.5), (89, 42.0), style="-")
arrow((89, 42.0), (13.5, 42.0), style="-")
arrow((13.5, 42.0), top(gen), style="-|>")
ax.text(51, 43.3, "top-5 chunks as generation context",
        fontsize=8.4, color="#555555", ha="center")

# three outcomes
o1 = box(76, 33.5, 20, 6.4, "ANSWER", "C \u2265 0.65", fc=C_OUT_OK, tc="white")
o2 = box(76, 26.0, 20, 6.4, "ANSWER WITH CAUTION", "0.45 \u2264 C < 0.65",
         fc=C_OUT_CAU, fontsize=8.4)
o3 = box(76, 18.5, 20, 6.4, "INSUFFICIENT EVIDENCE", "C < 0.45",
         fc=C_OUT_NO, tc="white", fontsize=8.4)

arrow(right(gd), left(o1), rad=-0.16)
arrow(right(gd), left(o2))
arrow(right(gd), left(o3), rad=0.16)

# ==========================================================================
# AUDIT LOG
# ==========================================================================
au = box(4, 9.5, 92, 6.6, "Audit log (JSONL)",
         "request id  \u00b7  routed section  \u00b7  chunk ids + scores  \u00b7  "
         "per-stage latency  \u00b7  groundedness  \u00b7  confidence  \u00b7  decision",
         fc=C_AUDIT, fontsize=9)

for src in [bottom(r1), bottom(gen), bottom(ev), bottom(gd)]:
    arrow(src, (src[0], 16.1), ls=(0, (3, 3)),
          color="#9A9A9A", lw=0.95, style="-|>")

# ==========================================================================
# LEGEND
# ==========================================================================
handles = [
    ("Agentic governance layer", C_AGENT),
    ("Retrieval stage", C_RETR),
    ("Generation", C_GEN),
    ("Offline ingestion", C_INGEST),
]
for i, (lab, col) in enumerate(handles):
    x0 = 4 + i * 24
    ax.add_patch(FancyBboxPatch((x0, 2.4), 3.2, 2.6,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=col, edgecolor=C_EDGE,
                                linewidth=1.0, zorder=3))
    ax.text(x0 + 4.2, 3.7, lab, fontsize=8.8, va="center", color=C_TEXT)

ax.text(50, 97.5, "PharmaRAG system architecture",
        ha="center", va="center", fontsize=14, weight="bold", color=C_TEXT)

out = os.path.join(OUTDIR, "fig1_architecture.png")
fig.savefig(out)
plt.close(fig)
print("wrote", out)