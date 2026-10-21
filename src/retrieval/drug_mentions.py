"""
Detect which corpus drugs a question names (Fix 2, post-hoc, 2026-10-21).
Used for multi-drug questions: when >=2 distinct drugs are named, retrieval runs once
per drug so that evidence for every named drug can reach the top-k.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache

from configs.settings import settings


@lru_cache(maxsize=1)
def _name_table() -> list[tuple[str, frozenset]]:
    table: dict[str, set] = {}
    with open(settings.data_dir / "drug_lists" / "ms_drugs.csv") as f:
        for r in csv.DictReader(f):
            for n in (r["drug_name"], r["generic_name"]):
                table.setdefault(n.lower(), set()).add(r["drug_name"])
    # longest names first so "natalizumab-sztn" wins over "natalizumab"
    return sorted(((n, frozenset(b)) for n, b in table.items()), key=lambda x: -len(x[0]))


def drug_groups(query: str) -> list[list[str]]:
    """Return one list of brand names per distinct drug named in the query."""
    text, found = query.lower(), []
    for name, brands in _name_table():
        if re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text):
            if not any(brands <= g for g in found):
                found.append(set(brands))
    return [sorted(g) for g in found]
