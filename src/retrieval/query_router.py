"""
Agent 1: Query Router.
Classifies incoming queries to likely SPL section types.
Uses the LLM to map user questions to label section taxonomy.
"""

import ollama

from configs.settings import settings


SECTION_DESCRIPTIONS = {
    "indications_and_usage": "What the drug is approved to treat",
    "contraindications": "When the drug should NOT be used",
    "warnings_and_precautions": "Serious risks and safety warnings",
    "boxed_warning": "Most serious FDA warnings (black box)",
    "adverse_reactions": "Side effects reported in clinical trials",
    "dosage_and_administration": "How to take/administer the drug, dosing",
    "drug_interactions": "Interactions with other medications",
    "use_in_specific_populations": "Use in pregnancy, pediatrics, elderly, renal/hepatic impairment",
}

ROUTER_PROMPT = """You are a pharmaceutical query classifier. Given a user question about a drug, identify which FDA drug label section(s) are most likely to contain the answer.

Available sections:
{sections}

Rules:
- Return 1-3 section names, most relevant first.
- Return ONLY the section names as a comma-separated list.
- If unsure, return: warnings_and_precautions,adverse_reactions

User question: {query}

Sections:"""


class QueryRouter:
    """Routes user queries to the most relevant SPL sections."""

    def __init__(self):
        self.model = settings.llm_model

    def route(self, query: str) -> list[str]:
        """
        Classify query into 1-3 relevant SPL section names.
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
                options={"temperature": 0.0, "num_predict": 100},
            )

            raw = response["message"]["content"].strip()
            # Parse comma-separated section names
            sections = [s.strip().lower().replace(" ", "_") for s in raw.split(",")]
            # Validate against known sections
            valid = [s for s in sections if s in SECTION_DESCRIPTIONS]

            if not valid:
                return ["warnings_and_precautions", "adverse_reactions"]

            return valid

        except Exception as e:
            print(f"[QueryRouter] Error: {e}")
            return ["warnings_and_precautions", "adverse_reactions"]