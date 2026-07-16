import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.indexing.embedder import PubMedEmbedder

emb = PubMedEmbedder()
THRESH = 0.35

rs = json.load(open("evaluation/results/optC_full_FINAL_t0.json"))
fails = 0
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
            snippet = cits[m].get("text_snippet", "")
            score = float(np.dot(sv, emb.embed_texts([snippet])[0]))
            if score < THRESH:
                fails += 1
                if fails <= 12:
                    print(f"[{r['query_id']}] score={score:.2f} snippet_len={len(snippet)}")
                    print(f"  claim: {clean[:110]}")
                    print(f"  cited: {snippet[:110]}")
                    print()
