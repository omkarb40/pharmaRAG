"""Apply AUTO + HUMAN gold choices -> evaluation/test_queries_v3.json (v2 is left untouched)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
Q = json.load(open(ROOT / "evaluation/test_queries_v2.json")); wrapped = isinstance(Q, dict)
qs = Q["queries"] if wrapped else Q
auto = json.load(open(ROOT / "evaluation/reannotation_auto_20261016.json"))
ans = json.load(open(ROOT / "evaluation/reannotation_answers_20261016.json"))
blank = [k for k, v in ans.items() if not v["choice"]]
if blank: sys.exit(f"Unanswered: {blank}")
flag = []
for q in qs:
    if q["id"] in auto:
        q["reference_chunk_id"] = auto[q["id"]]; q["gold_provenance"] = "auto_remap_verbatim_20261016"
    elif q["id"] in ans:
        a = ans[q["id"]]
        if a["choice"].upper() == "X":
            q["needs_rewrite"] = True; flag.append(q["id"]); continue
        q["reference_chunk_id"] = a["candidates"][a["choice"].upper()]
        q["gold_provenance"] = "human_reannotation_20261016"
        if a.get("note"): q["reannotation_note"] = a["note"]
json.dump(Q if wrapped else qs, open(ROOT / "evaluation/test_queries_v3.json", "w"), indent=2)
print(f"wrote evaluation/test_queries_v3.json  (auto {len(auto)}, human {len(ans) - len(flag)}, needs_rewrite {flag})")
