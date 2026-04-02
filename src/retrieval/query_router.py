"""
Agent 1: Query Router.
Classifies user questions into relevant SPL section types.

Why this matters:
  Without routing, a query about "contraindications" might retrieve
  chunks from "dosage_and_administration" just because they mention
  the same drug. Routing narrows the search space, which:
    1. Improves retrieval precision (fewer irrelevant chunks)
    2. Reduces hallucination (LLM sees less noise)
    3. Speeds up retrieval (smaller search space)

How it works:
  Uses the LLM to classify the question → returns 1-3 section names.
  These are passed as metadata filters to the hybrid retriever.

  Example:
    Query: "What are the side effects of fingolimod?"
    Router output: ["adverse_reactions"]

    Query: "Can I take this drug during pregnancy?"
    Router output: ["use_in_specific_populations", "contraindications"]
"""

import ollama

from configs.settings import settings


SECTION_DESCRIPTIONS = {
    "indications_and_usage": "What the drug is approved to treat, therapeutic uses",
    "contraindications": "When the drug should NOT be used, absolute restrictions",
    "warnings_and_precautions": "Serious risks, safety warnings, monitoring requirements",
    "boxed_warning": "Most serious FDA warnings (black box warnings)",
    "adverse_reactions": "Side effects, adverse events from clinical trials",
    "dosage_and_administration": "How to take the drug, dosing, administration method",
    "drug_interactions": "Interactions with other medications, food, substances",
    "use_in_specific_populations": "Use in pregnancy, pediatrics, elderly, renal/hepatic impairment",
}

ROUTER_PROMPT = """You are a pharmaceutical query classifier. Given a user question about a drug, identify which FDA drug label section(s) are most likely to contain the answer.

Available sections:
{sections}

Rules:
- Return 1-3 section names, most relevant first.
- Return ONLY the section names as a comma-separated list, nothing else.
- If the question is about side effects, return: adverse_reactions
- If the question is about when NOT to use a drug, return: contraindications
- If the question is about pregnancy or children, return: use_in_specific_populations
- If the question is about dangerous risks or black box, return: warnings_and_precautions,boxed_warning
- If the question is about how much to take, return: dosage_and_administration
- If the question is about mixing with other drugs, return: drug_interactions
- If unsure, return: warnings_and_precautions,adverse_reactions

User question: {query}

Sections:"""


class QueryRouter:
    """Routes user queries to the most relevant SPL sections."""

    def __init__(self):
        self.model = settings.llm_model
        print(f"[QueryRouter] Initialized with model: {self.model}")

    def route(self, query: str) -> list[str]:
        """
        Classify query into 1-3 relevant section names.
        Returns list of section_name strings.
        """
        sections_text = "\n".join(
            f"- {name}: {desc}" for name, desc in SECTION_DESCRIPTIONS.items()
        )
        prompt = ROUTER_PROMPT.format(sections=sections_text, query=query)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 60},
            )

            raw = response["message"]["content"].strip()

            # Parse the response — expect comma-separated section names
            sections = []
            for part in raw.split(","):
                cleaned = part.strip().lower().replace(" ", "_")
                # Remove any extra text the LLM might add
                for known in SECTION_DESCRIPTIONS:
                    if known in cleaned:
                        sections.append(known)
                        break

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for s in sections:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)

            if unique:
                return unique[:3]

        except Exception as e:
            print(f"[QueryRouter] Error: {e}")

        # Fallback: broadest safety-relevant sections
        return ["warnings_and_precautions", "adverse_reactions"]