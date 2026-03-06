'''
Central configuration file for the application. This file contains all the settings and parameters that can be adjusted to customize the behavior of the application. It includes database configurations, API keys, logging settings, and other important parameters that are essential for the application's functionality.
Make sure to keep this file secure, especially if it contains sensitive information such as API keys or database credentials. It is recommended to use environment variables or a secure vault to store sensitive information and reference them in this configuration file.
'''
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # === Project Paths ===
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    index_dir: Path = data_dir / "index"
    logs_dir: Path = project_root / "logs"

    # === LLM Configuration ===
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma3:12b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # === Embedding Configuration ===
    embedding_model: str = "neuml/pubmedbert-base-embeddings"
    embedding_dimension: int = 768

    # === ChromaDB ===
    chroma_collection_name: str = "pharma_rag_chunks"

    # === Retrieval Configuration ===
    top_k_retrieval: int = 10
    top_k_final: int = 5
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6

    # === Chunking Configuration ===
    chunk_size: int = 512
    chunk_overlap: int = 64

    # === Agent Thresholds ===
    refusal_confidence_threshold: float = 0.4
    evidence_support_threshold: float = 0.5

    # === DailyMed API ===
    dailymed_base_url: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    # === Monitoring ===
    enable_audit_logging: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()