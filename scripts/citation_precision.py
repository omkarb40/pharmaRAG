"""
Post-hoc citation precision: for each cited sentence, does the cited
chunk actually support it (cosine >= threshold)?
Lower bound — uses the 200-char text_snippet stored with each citation.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.indexing.embedder import PubMedEmbedder

THRESH = 0.35
emb = PubMedEmbedder()

FILES = [
    ("baseline", "baseline_full_FINAL_t0.json"),
    ("optC", "optC_full_FINAL_t0.json"),
]

for tag, fname in FILES:
    rs = json.load(open(f"evaluation/results/{fname}"))
    total_cites, good_cites = 0, 0
    for r in rs:
        ans = r.get("answer", "")
        cits = {c["citation_id"]: c for c in r.get("citations", [])}
        if not ans or not cits:
            continue
        for sent in re.split(r'(?<=[.!?])\s+', ans):
            marks = [int(m) for m in re.findall(r'\[(\d+)\]', sent)]
            clean = re.sub(r'\[\d+\]', '', sent).strip()
            if not marks or len(clean) < 15:
                continue
            sv = emb.embed_texts([clean])[0]
            for m in marks:
                if m not in cits:
                    continue
                total_cites += 1
                snippet = cits[m].get("text_snippet", "")
                cv = emb.embed_texts([snippet])[0]
                if float(np.dot(sv, cv)) >= THRESH:
                    good_cites += 1
    prec = good_cites / total_cites if total_cites else 0
    print(f"{tag}: citation precision {prec:.4f}  ({good_cites}/{total_cites} citations support their claim)")
