"""
Build a blind annotation packet for a second annotator (IAA).
Samples N answer-queries stratified by category; for each, presents the
top-k retrieved chunks UNION the gold chunk, shuffled and letter-labelled.
Writes:
  evaluation/iaa/packet.txt        <- give this to annotator 2
  evaluation/iaa/answer_key.json   <- DO NOT SHOW ANNOTATOR 2
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from src.retrieval.hybrid_search import HybridRetriever

N_QUERIES = 20
TOP_K = 6
SEED = 42
OUT_DIR = Path("evaluation/iaa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

chunks_by_id = {}
with open(settings.chunks_file) as f:
    for line in f:
        line = line.strip()
        if line:
            c = json.loads(line)
            chunks_by_id[c["chunk_id"]] = c

queries = json.load(open("evaluation/test_queries_v2.json"))
answerable = [q for q in queries
              if q.get("expected_behavior") == "answer" and q.get("reference_chunk_id")]

# stratified sample across categories
by_cat = defaultdict(list)
for q in answerable:
    by_cat[q["category"]].append(q)
sample, cats = [], sorted(by_cat)
while len(sample) < N_QUERIES:
    added = False
    for cat in cats:
        pool = [q for q in by_cat[cat] if q not in sample]
        if pool and len(sample) < N_QUERIES:
            sample.append(random.choice(pool)); added = True
    if not added:
        break
sample = sample[:N_QUERIES]

retriever = HybridRetriever()
key, lines = {}, []

lines.append("PharmaRAG — Gold Evidence Annotation Task")
lines.append("=" * 72)
lines.append("""
For each QUERY below you will see several candidate passages (A, B, C...)
taken from FDA drug labels.

TASK — two things per query:
  1. For EACH candidate, mark Y or N: does this passage contain information
     that directly answers the query?
  2. Then pick the SINGLE BEST passage letter that most directly answers it
     (or NONE if no passage answers it).

Rules:
  - A passage about the WRONG DRUG is always N, even if the topic matches.
  - "Directly answers" means the answer is stated, not merely implied.
  - Work independently. Do not discuss with anyone.

Record answers in evaluation/iaa/annotator2.txt, one line per query:
     QUERY_ID | per-candidate Y/N in order | BEST
  example:
     CI-003 | N,N,Y,N,N,N | C
""")
lines.append("=" * 72)

for q in sample:
    qid, qtext = q["id"], q["query"]
    gold_id = q["reference_chunk_id"]
    hits = retriever.search(query=qtext, top_k=TOP_K)
    cand_ids = [h["chunk_id"] for h in hits]
    if gold_id not in cand_ids:
        cand_ids = cand_ids[:TOP_K - 1] + [gold_id]
    cand_ids = [cid for cid in dict.fromkeys(cand_ids) if cid in chunks_by_id]
    random.shuffle(cand_ids)

    letters = [chr(65 + i) for i in range(len(cand_ids))]
    key[qid] = {
        "query": qtext,
        "letters": dict(zip(letters, cand_ids)),
        "gold_chunk_id": gold_id,
        "gold_letter": letters[cand_ids.index(gold_id)] if gold_id in cand_ids else None,
    }

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"{qid}")
    lines.append(f"QUERY: {qtext}")
    lines.append("-" * 72)
    for L, cid in zip(letters, cand_ids):
        c = chunks_by_id[cid]
        txt = " ".join(c["text"].split())[:700]
        lines.append(f"\n  [{L}] Drug: {c['drug_name']} | Section: {c['section_name']}")
        lines.append(f"      {txt}")
    lines.append(f"\n  Y/N for {','.join(letters)}:  ______   BEST: ____")

(OUT_DIR / "packet.txt").write_text("\n".join(lines))
json.dump(key, open(OUT_DIR / "answer_key.json", "w"), indent=2)
print(f"Packet: {OUT_DIR/'packet.txt'}  ({len(sample)} queries)")
print(f"Answer key (KEEP PRIVATE): {OUT_DIR/'answer_key.json'}")
print("Categories:", dict((c, sum(1 for q in sample if q['category'] == c)) for c in sorted({q['category'] for q in sample})))
