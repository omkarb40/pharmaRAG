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
    """Generates grounded, citation-backed answers via Ollama."""

    def __init__(self):
        self.model = settings.llm_model
        # Verify Ollama connection on startup
        try:
            ollama.list()
            print(f"[Generator] Ollama connected. Model: {self.model}")
        except Exception as e:
            print(f"[Generator] WARNING: Cannot connect to Ollama: {e}")
            print(f"[Generator] Make sure 'ollama serve' is running.")

    def _format_evidence(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks as numbered evidence for the LLM.
        
        Each chunk becomes:
          [1] Drug: Tysabri | Section: contraindications
          <chunk text>
        
        This numbering is what the LLM uses for citations.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            drug = meta.get("drug_name", "Unknown")
            generic = meta.get("generic_name", "")
            section = meta.get("section_name", "Unknown").replace("_", " ").title()
            text = chunk.get("text", "")

            header = f"[{i}] Drug: {drug}"
            if generic:
                header += f" ({generic})"
            header += f" | Section: {section}"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate a cited answer from retrieved evidence.
        
        Returns:
          {
            "answer": "The answer text with [1][2] citations...",
            "citations": [...],         # Map of citation numbers to chunk metadata
            "generation_time_ms": 2340,  # How long the LLM took
            "model": "gemma3:12b",
            "num_evidence_chunks": 5,
          }
        """
        # Format evidence
        evidence = self._format_evidence(chunks)
        prompt = GENERATION_PROMPT.format(evidence=evidence, query=query)

        # Call LLM
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

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}. Is Ollama running?",
                "citations": [],
                "generation_time_ms": 0,
                "model": self.model,
                "num_evidence_chunks": len(chunks),
            }

        # Build citation map
        # Each citation [i] maps back to the chunk that provided it
        citations = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            citations.append({
                "citation_id": i,
                "chunk_id": chunk.get("chunk_id", ""),
                "drug_name": meta.get("drug_name", ""),
                "generic_name": meta.get("generic_name", ""),
                "section_name": meta.get("section_name", ""),
                "retrieval_score": chunk.get("fused_score", 0),
                "text_snippet": chunk.get("text", "")[:200],
            })

        return {
            "answer": answer,
            "citations": citations,
            "generation_time_ms": round(gen_time * 1000),
            "model": self.model,
            "num_evidence_chunks": len(chunks),
        }