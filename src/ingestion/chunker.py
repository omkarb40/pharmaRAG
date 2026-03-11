"""
Section-aware chunker.
Chunks parsed SPL sections into retrieval-sized pieces while
preserving section metadata (drug, section type, provenance).
"""

import hashlib
import json
from pathlib import Path

from configs.settings import settings


class SectionAwareChunker:
    """
    Chunks text by token-approximate character windows.
    Each chunk retains full provenance: drug, section, position.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        # Approximate chars (1 token ≈ 4 chars for English)
        self.chunk_size_chars = chunk_size * 4
        self.chunk_overlap_chars = chunk_overlap * 4

    def _generate_chunk_id(self, drug_name: str, section: str, idx: int) -> str:
        """Deterministic chunk ID for deduplication."""
        raw = f"{drug_name}::{section}::{idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def chunk_section(self, section: dict) -> list[dict]:
        """
        Split a single section into overlapping chunks.
        Tries to break on sentence boundaries.
        """
        text = section["text"]
        drug = section["drug_name"]
        sec_name = section["section_name"]

        if len(text) <= self.chunk_size_chars:
            chunk_id = self._generate_chunk_id(drug, sec_name, 0)
            return [{
                "chunk_id": chunk_id,
                "drug_name": drug,
                "set_id": section["set_id"],
                "section_name": sec_name,
                "loinc_code": section["loinc_code"],
                "chunk_index": 0,
                "total_chunks": 1,
                "text": text,
            }]

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size_chars

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation near the boundary
                search_zone = text[end - 100: end + 100] if end + 100 < len(text) else text[end - 100:]
                for punct in [". ", ".\n", "? ", "! "]:
                    last_punct = search_zone.rfind(punct)
                    if last_punct != -1:
                        end = (end - 100) + last_punct + len(punct)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = self._generate_chunk_id(drug, sec_name, idx)
                chunks.append({
                    "chunk_id": chunk_id,
                    "drug_name": drug,
                    "set_id": section["set_id"],
                    "section_name": sec_name,
                    "loinc_code": section["loinc_code"],
                    "chunk_index": idx,
                    "text": chunk_text,
                })
                idx += 1

            start = end - self.chunk_overlap_chars

        # Backfill total_chunks
        for c in chunks:
            c["total_chunks"] = len(chunks)

        return chunks

    def chunk_all(self, sections: list[dict]) -> list[dict]:
        """Chunk all sections, return flat list of chunks."""
        all_chunks = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section))

        print(f"[Chunker] Total chunks: {len(all_chunks)}")
        return all_chunks

    def save_chunks(self, chunks: list[dict], output_path: Path):
        """Save chunks to JSONL for reproducibility."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        print(f"[Chunker] Saved {len(chunks)} chunks to {output_path}")

    @staticmethod
    def load_chunks(path: Path) -> list[dict]:
        """Load chunks from JSONL."""
        chunks = []
        with open(path, "r") as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        return chunks