"""
Dev-set evaluation of the Safety-Intent Check alone (~3 min, needs Ollama).
  python scripts/eval_intent_check.py                      # 85-query dev set (v2 questions)
  python scripts/eval_intent_check.py --queries <file>     # e.g. the held-out set, AFTER the freeze tag
What matters:
  - false declines on answerable questions  (must be ~0: each one adds to over-refusal, issue #2)
  - declines on adversarial questions        (the class the guard could not catch)
  near_scope / out_of_scope refuse items are evidence-absent; the policy ALLOWS them by design
  (the generator's abstention handles those), so they are reported but not scored as misses.
"""
import json, sys, collections, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generation.safety_intent import SafetyIntentChecker, POLICY_VERSION

qfile = Path(sys.argv[sys.argv.index("--queries") + 1]) if "--queries" in sys.argv else Path("evaluation/test_queries_v2.json")
Q = json.load(open(qfile)); Q = Q["queries"] if isinstance(Q, dict) else Q
chk = SafetyIntentChecker()
rows = []
for i, q in enumerate(Q):
    r = chk.check(q["query"])
    rows.append({"id": q["id"], "scope": q.get("scope"), "expected_behavior": q.get("expected_behavior"),
                 "query": q["query"], **r})
    print(f"[{i+1}/{len(Q)}] {q['id']:<8} {q.get('scope',''):<13} exp={q.get('expected_behavior'):<7} -> {r['decision']:<7} {r['reason'][:70]}")

by = collections.defaultdict(collections.Counter)
for r in rows: by[(r["scope"], r["expected_behavior"])][r["decision"]] += 1
ans = [r for r in rows if r["expected_behavior"] == "answer"]
adv = [r for r in rows if r["scope"] == "adversarial"]
fd = [r["id"] for r in ans if r["decision"] == "DECLINE"]
print(f"\n=== {POLICY_VERSION} on {qfile.name} ===")
for k, v in sorted(by.items()): print(f"  {k[0]:<13} expected {k[1]:<7} {dict(v)}")
print(f"  FALSE DECLINES on answerable: {len(fd)}/{len(ans)}  {fd}")
print(f"  adversarial declined:        {sum(r['decision']=='DECLINE' for r in adv)}/{len(adv)}")
print(f"  parse failures:              {sum(not r['parse_ok'] for r in rows)}")
print(f"  mean latency:                {sum(r['latency_ms'] for r in rows)/len(rows):.0f} ms")
out = Path("evaluation/results") / f"intent_check_{POLICY_VERSION}_{qfile.stem}.json"
json.dump(rows, open(out, "w"), indent=2); print(f"  saved {out}")
