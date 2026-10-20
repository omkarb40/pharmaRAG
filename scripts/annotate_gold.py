"""
Interactive gold re-annotation (one item at a time, saves after every answer, resumable).
  python scripts/annotate_gold.py
Keys: A/B/C/... = pick that chunk   X = no candidate answers it   S = skip for now
      N = add a note first           Q = quit (progress is saved)
"""
import json, textwrap, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ANS = ROOT / "evaluation/reannotation_answers_20261016.json"
chunks = {c["chunk_id"]: c for c in map(json.loads, open(ROOT / "data/processed/chunks.jsonl"))}
old = {c["chunk_id"]: c for c in map(json.loads, open(ROOT / "data/processed/chunks_pre_corpusfix_20261015.jsonl"))}
Q = json.load(open(ROOT / "evaluation/test_queries_v2.json")); Q = {q["id"]: q for q in (Q["queries"] if isinstance(Q, dict) else Q)}
ans = json.load(open(ANS))
wrap = lambda t, n=900: "\n".join(textwrap.wrap((t or "")[:n], 100, initial_indent="    ", subsequent_indent="    "))
todo = [k for k, v in ans.items() if not v["choice"]]
print(f"\n{len(ans) - len(todo)}/{len(ans)} done. {len(todo)} to go.\n")
for n, qid in enumerate(todo, 1):
    item, q = ans[qid], Q[qid]
    o = old.get(q.get("reference_chunk_id"), {})
    print("=" * 104)
    print(f"[{n}/{len(todo)}] {qid}   expected drug: {q.get('expected_drug')}   expected section: {q.get('expected_section')}")
    print(f"\nQUESTION: {q['query']}\n")
    print(f"OLD GOLD ({o.get('drug_name','?')} / {o.get('section_name','?')}) -- for reference only, it may have been wrong:")
    print(wrap(o.get("text") or "(no longer exists)", 400))
    for letter, cid in item["candidates"].items():
        c = chunks[cid]
        print(f"\n  [{letter}] {c['drug_name']} / {c['section_name']} / chunk {c['chunk_index']+1} of {c['total_chunks']}")
        print(wrap(c["text"]))
    while True:
        k = input(f"\n{qid} -> choose {'/'.join(item['candidates'])}, X (none answers it), S (skip), N (note), Q (quit): ").strip().upper()
        if k == "Q":
            json.dump(ans, open(ANS, "w"), indent=2); print("Saved. Run again to resume."); sys.exit(0)
        if k == "S": break
        if k == "N":
            item["note"] = input("  note: ").strip(); continue
        if k == "X" or k in item["candidates"]:
            item["choice"] = k; json.dump(ans, open(ANS, "w"), indent=2); break
        print("  not a valid key")
left = [k for k, v in ans.items() if not v["choice"]]
print(f"\nDone for now. Remaining: {len(left)} {left}")
if not left: print("All answered. Next: python scripts/apply_reannotation.py")
