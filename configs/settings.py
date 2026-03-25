'''
Central configuration file for the application. This file contains all the settings and parameters that can be adjusted to customize the behavior of the application. It includes database configurations, API keys, logging settings, and other important parameters that are essential for the application's functionality.
Make sure to keep this file secure, especially if it contains sensitive information such as API keys or database credentials. It is recommended to use environment variables or a secure vault to store sensitive information and reference them in this configuration file.
'''
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings:
    """All project settings in one place."""

    def __init__(self):
        # === Paths ===
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.data_dir / "index"
        self.logs_dir = self.project_root / "logs"
        self.chunks_file = self.processed_dir / "chunks.jsonl"

        # === LLM ===
        self.ollama_base_url = "http://localhost:11434"
        self.llm_model = "gemma3:12b"
        self.llm_temperature = 0.1
        self.llm_max_tokens = 1024

        # === Embeddings ===
        self.embedding_model = "neuml/pubmedbert-base-embeddings"
        self.embedding_dimension = 768

        # === ChromaDB ===
        self.chroma_dir = self.index_dir / "chromadb"
        self.chroma_collection_name = "pharma_rag_chunks"

        # === BM25 ===
        self.bm25_dir = self.index_dir / "bm25"

        # === Retrieval ===
        self.semantic_weight = 0.6
        self.bm25_weight = 0.4
        self.rrf_k = 60
        self.top_k_retrieval = 20
        self.top_k_final = 5

        # === Monitoring ===
        self.enable_audit_logging = True


# Singleton instance — import this everywhere
settings = Settings()