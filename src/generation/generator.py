"""
LLM answer generator.
Produces citation-backed answers from retrieved evidence chunks.
"""

import ollama
import time

from configs.settings import settings


GENERATION_PROMPT = """You are a pharmaceutical safety assistant. Answer the user's question using ONLY the provided evidence chunks. Follow these rules strictly:

1. Use ONLY information from the evidence chunks below.
2. Cite every claim using [1], [2], etc. matching the chunk numbers.
3. If the evidence is insufficient, say "Insufficient evidence to answer this question."
4. Be concise and precise. Use clinical language appropriate for healthcare professionals.
5. Never invent or assume information not in the evidence.

Evidence Chunks:
{evidence}

User Question: {query}

Answer:"""


class AnswerGenerator:
    """Generates grounded, citation-backed answers using Ollama."""

    def __init__(self):
        self.model = settings.llm_model

    def _format_evidence(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as numbered evidence."""
        evidence_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            drug = meta.get("drug_name", "Unknown")
            section = meta.get("section_name", "Unknown")
            text = chunk.get("text", "")
            evidence_parts.append(
                f"[{i}] Drug: {drug} | Section: {section}\n{text}"
            )
        return "\n\n".join(evidence_parts)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate an answer with citations.
        Returns dict with answer, citations, and timing.
        """
        evidence = self._format_evidence(chunks)
        prompt = GENERATION_PROMPT.format(evidence=evidence, query=query)

        start = time.time()

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": settings.llm_temperature,
                    "num_predict": settings.llm_max_tokens,
                },
            )

            answer = response["message"]["content"].strip()
            gen_time = time.time() - start

            # Build citation map
            citations = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                citations.append({
                    "citation_id": i,
                    "chunk_id": chunk.get("chunk_id"),
                    "drug_name": meta.get("drug_name"),
                    "section_name": meta.get("section_name"),
                    "text_snippet": chunk.get("text", "")[:200],
                    "retrieval_score": chunk.get("fused_score", 0),
                })

            return {
                "answer": answer,
                "citations": citations,
                "generation_time_ms": round(gen_time * 1000),
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations": [],
                "generation_time_ms": 0,
            }