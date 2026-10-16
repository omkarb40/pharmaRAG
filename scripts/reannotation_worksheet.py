"""
Gold re-annotation after the 2026-10-16 corpus snapshot.
  python scripts/reannotation_worksheet.py   -> writes worksheet + answers template
  (you fill evaluation/reannotation_answers_20261016.json by hand)
  python scripts/apply_reannotation.py       -> writes evaluation/test_queries_v3.json
AUTO items: the original reference snippet occurs verbatim in exactly one new chunk of the
same drug; only the chunk id is re-pointed (no judgment). Every other item is a HUMAN choice.
"""
import difflib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
N = [json.loads(l) for l in open(ROOT / "data/processed/chunks.jsonl")]
O = {c["chunk_id"]: c for c in map(json.loads, open(ROOT / "data/processed/chunks_pre_corpusfix_20261015.jsonl"))}
Q = json.load(open(ROOT / "evaluation/test_queries_v2.json")); Q = Q["queries"] if isinstance(Q, dict) else Q
R = json.load(open(ROOT / "evaluation/gold_reannotation_needed.json"))
norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()
ids = lambda b: [e if isinstance(e, str) else e["id"] for e in R[b]]
todo = ids("text_changed_snippet_still_present") + ids("moved") + ids("lost")
todo += [i for i in R["review_boxed"] if i not in todo]
qby = {q["id"]: q for q in Q}

auto, human = {}, []
for qid in todo:
    q = qby[qid]; old = O.get(q["reference_chunk_id"], {})
    drugs = {old.get("drug_name") or q.get("expected_drug")}
    snip = norm(q.get("reference_snippet"))
    exact = [c for c in N if c["drug_name"] in drugs and snip and snip in norm(c["text"])]
    if len(exact) == 1 and qid not in R["review_boxed"]:
        auto[qid] = exact[0]["chunk_id"]; continue
    # candidates: same drug(s), ranked by similarity to the OLD gold text (or snippet)
    alias = {a.lower() for a in q.get("expected_drug_aliases", [])}
    pool = [c for c in N if c["drug_name"] in drugs or c["drug_name"].lower() in alias]
    ref = norm(old.get("text") or q.get("reference_snippet"))
    score = lambda c: difflib.SequenceMatcher(None, ref[:1500], norm(c["text"])[:1500]).ratio()
    cands = sorted(pool, key=score, reverse=True)[:3]
    if q.get("expected_section") == "boxed_warning" or qid in R["review_boxed"]:
        for c in pool:
            if c["section_name"] == "boxed_warning" and c not in cands: cands.append(c)
    human.append((q, old, cands))

out = ROOT / "evaluation/reannotation_worksheet_20261016.md"
L = ["# Gold re-annotation worksheet (corpus snapshot 2026-10-16)\n",
     f"{len(auto)} items re-pointed automatically (verbatim snippet, single chunk). {len(human)} need your choice.\n",
     "For each item, pick the ONE candidate that best answers the question as the old gold did, "
     "write its letter in `evaluation/reannotation_answers_20261016.json`. "
     "Use `\"X\"` if no candidate answers it (the query will be flagged for a rewrite). Judge from the label text only.\n"]
template = {}
for q, old, cands in human:
    L += [f"\n---\n## {q['id']}  ({q.get('expected_drug')} · expected section: {q.get('expected_section')})\n",
          f"**Question:** {q['query']}\n",
          f"**Old gold text** ({old.get('chunk_id','missing')}, {old.get('section_name','?')}):\n> {(old.get('text') or '(chunk no longer exists)')[:500]}\n"]
    for i, c in enumerate(cands):
        L.append(f"**{chr(65+i)}** `{c['chunk_id']}` · {c['drug_name']} · {c['section_name']} · chunk {c['chunk_index']+1}/{c['total_chunks']}\n> {c['text'][:500]}\n")
    template[q["id"]] = {"choice": "", "candidates": {chr(65+i): c["chunk_id"] for i, c in enumerate(cands)}, "note": ""}
out.write_text("\n".join(L))
json.dump(template, open(ROOT / "evaluation/reannotation_answers_20261016.json", "w"), indent=2)
json.dump(auto, open(ROOT / "evaluation/reannotation_auto_20261016.json", "w"), indent=2)
print(f"AUTO: {len(auto)}  HUMAN: {len(human)}\nworksheet: {out.relative_to(ROOT)}")
