import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.indexing.embedder import PubMedEmbedder

emb = PubMedEmbedder()
THRESH = 0.35
rs = json.load(open("evaluation/results/optC_full_FINAL_t0.json"))

misaligned = 0   # cited chunk fails, but SOME chunk supports -> attribution error
true_unsup = 0   # NO chunk supports -> genuine unsupported claim
for r in rs:
    ans = r.get("answer", "")
    cits = {c["citation_id"]: c for c in r.get("citations", [])}
    all_snips = [c.get("text_snippet", "") for c in r.get("citations", [])]
    if not ans or not cits:
        continue
    snip_vecs = emb.embed_texts(all_snips) if all_snips else None
    for sent in re.split(r'(?<=[.!?])\s+', ans):
        marks = [int(m) for m in re.findall(r'\[(\d+)\]', sent)]
        clean = re.sub(r'\[\d+\]', '', sent).strip()
        if not marks or len(clean) < 15:
            continue
        sv = emb.embed_texts([clean])[0]
        for m in marks:
            if m not in cits:
                continue
            cited_score = float(np.dot(sv, emb.embed_texts([cits[m].get("text_snippet","")])[0]))
            if cited_score >= THRESH:
                continue
            # cited chunk failed — does ANY chunk in this answer support it?
            best_any = float(np.max(np.dot(snip_vecs, sv))) if snip_vecs is not None else 0
            if best_any >= THRESH:
                misaligned += 1
            else:
                true_unsup += 1

total_fail = misaligned + true_unsup
print(f"Failed citations: {total_fail}")
print(f"  misalignment (some chunk supports, wrong one cited): {misaligned}")
print(f"  genuinely unsupported (no chunk supports):           {true_unsup}")
