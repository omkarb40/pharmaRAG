"""
Option A: Query Expansion.

Appends domain synonyms to a query before retrieval so lay phrasing
("liver problems") also carries the clinical terms drug labels use
("hepatic impairment", "hepatotoxicity"). Expansion is applied to the
query string only — the index is unchanged. This isolates the effect
of vocabulary bridging from any indexing change.

Design notes for the paper:
  - Transparent, rule-based (not model-based) so every expansion is
    inspectable and reproducible.
  - Bidirectional where sensible (liver->hepatic AND hepatic->liver)
    so both patient- and clinician-phrased queries benefit.
  - Conservative: only high-precision medical synonyms, to avoid
    injecting noise that would hurt precision.
"""

# Each key, if found as a whole word in the query (case-insensitive),
# appends its expansion terms. Keep terms high-precision.
SYNONYM_MAP = {
    # hepatic
    "liver": ["hepatic", "hepatotoxicity", "hepatic impairment"],
    "hepatic": ["liver"],
    # renal
    "kidney": ["renal", "renal impairment"],
    "renal": ["kidney"],
    # cardiac
    "heart": ["cardiac", "cardiovascular", "bradycardia"],
    "cardiac": ["heart"],
    "heart rate": ["bradycardia", "heart rate reduction"],
    # neuro / PML
    "brain infection": ["progressive multifocal leukoencephalopathy", "PML"],
    "pml": ["progressive multifocal leukoencephalopathy"],
    # pregnancy / lactation
    "pregnancy": ["pregnant", "use in specific populations", "fetal"],
    "pregnant": ["pregnancy", "fetal harm"],
    "breastfeeding": ["lactation", "breastfed"],
    # general safety vocabulary
    "side effects": ["adverse reactions", "adverse events"],
    "side effect": ["adverse reactions"],
    "tired": ["fatigue", "somnolence"],
    "interactions": ["drug interactions"],
    "not be used": ["contraindicated", "contraindications"],
    "should not": ["contraindicated"],
    "black box": ["boxed warning"],
    "vaccinated": ["vaccination", "live vaccines", "immunization"],
    "vaccine": ["vaccination", "live vaccines"],
}


def expand_query(query: str) -> str:
    """
    Return the query with high-precision domain synonyms appended.
    Original query text is preserved verbatim at the front so exact
    matches still score; expansions are appended, not substituted.
    """
    q_lower = query.lower()
    additions = []
    seen = set()

    for trigger, expansions in SYNONYM_MAP.items():
        if trigger in q_lower:
            for term in expansions:
                key = term.lower()
                if key not in q_lower and key not in seen:
                    additions.append(term)
                    seen.add(key)

    if not additions:
        return query
    return query + " " + " ".join(additions)