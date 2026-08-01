"""
Multi-annotator IAA (5 raters).
Reports:
  - Fleiss' kappa on the binary relevance judgment (each candidate: relevant Y/N)
  - Exact gold-chunk selection: per-rater agreement with the reference key
  - Non-expert (1-4) vs domain-expert (5) agreement
  - Per-query unanimity
"""
import json
from pathlib import Path
from collections import defaultdict

KEY = json.load(open("evaluation/iaa/answer_key.json"))
EXPERT = "annotator5"           # domain expert
RATERS = ["annotator1", "annotator2", "annotator3", "annotator4", "annotator5"]

def parse(path):
    """Return {qid: {'yn': {letter: bool}, 'best': letter}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3: continue
        qid, yn_raw, best = parts[0], parts[1], parts[2].upper()
        if qid not in KEY: continue
        letters = list(KEY[qid]["letters"].keys())
        vals = [v.strip().upper() for v in yn_raw.split(",")]
        if len(vals) != len(letters): 
            print(f"  ! {path} {qid}: {len(vals)} vals, expected {len(letters)}"); continue
        out[qid] = {"yn": {L: (v == "Y") for L, v in zip(letters, vals)}, "best": best}
    return out

raters = {r: parse(f"evaluation/iaa/{r}.txt") for r in RATERS}

# queries all raters completed
common = set(KEY)
for r in RATERS:
    common &= set(raters[r])
common = sorted(common)
print(f"Queries scored by all {len(RATERS)} raters: {len(common)}")

# ---- Fleiss' kappa on binary (candidate-level) judgments ----
# Each "item" = one (query, candidate) pair. Each rater says relevant / not.
items = []
for qid in common:
    for L in KEY[qid]["letters"]:
        n_yes = sum(1 for r in RATERS if raters[r][qid]["yn"].get(L, False))
        n_no = len(RATERS) - n_yes
        items.append((n_yes, n_no))

N, n = len(items), len(RATERS)
if N:
    p_yes = sum(y for y, _ in items) / (N * n)
    p_no = 1 - p_yes
    Pe = p_yes**2 + p_no**2
    Pi = [(y*(y-1) + no*(no-1)) / (n*(n-1)) for y, no in items]
    Pbar = sum(Pi) / N
    fleiss = (Pbar - Pe) / (1 - Pe) if (1 - Pe) else 0
    band = ("almost perfect" if fleiss>0.8 else "substantial" if fleiss>0.6 else
            "moderate" if fleiss>0.4 else "fair" if fleiss>0.2 else "slight")
    print(f"\nFLEISS' KAPPA (binary relevance, {n} raters, {N} candidate-items): {fleiss:.4f}  [{band}]")

# ---- exact gold-chunk selection agreement with the reference key ----
print("\nEXACT GOLD-CHUNK SELECTION (best-choice vs reference key):")
for r in RATERS:
    hit = sum(1 for qid in common if raters[r][qid]["best"] == KEY[qid]["gold_letter"])
    tag = " (expert)" if r == EXPERT else ""
    print(f"  {r}{tag}: {hit}/{len(common)} = {hit/len(common):.1%}")

# ---- non-expert vs expert agreement ----
print("\nNON-EXPERT vs DOMAIN-EXPERT (best-choice match with expert):")
nonexperts = [r for r in RATERS if r != EXPERT and r != "annotator1"]
for r in nonexperts:
    agree = sum(1 for qid in common if raters[r][qid]["best"] == raters[EXPERT][qid]["best"])
    print(f"  {r} vs {EXPERT}: {agree}/{len(common)} = {agree/len(common):.1%}")
you_expert = sum(1 for qid in common if raters['annotator1'][qid]["best"] == raters[EXPERT][qid]["best"])
print(f"  annotator1 (you) vs {EXPERT}: {you_expert}/{len(common)} = {you_expert/len(common):.1%}")

# ---- per-query unanimity on best choice ----
unanimous = sum(1 for qid in common
                if len({raters[r][qid]["best"] for r in RATERS}) == 1)
print(f"\nUNANIMOUS best-choice queries: {unanimous}/{len(common)} = {unanimous/len(common):.1%}")

# ---- disagreements worth inspecting ----
print("\nQUERIES WITH DISAGREEMENT (best-choice spread):")
for qid in common:
    choices = {r: raters[r][qid]["best"] for r in RATERS}
    if len(set(choices.values())) > 1:
        gold = KEY[qid]["gold_letter"]
        print(f"  {qid} (gold={gold}): " + ", ".join(f"{r[-1]}={c}" for r, c in choices.items()))
