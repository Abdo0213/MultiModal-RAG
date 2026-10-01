from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class Settings:
    # Embedding model settings
    embed_model: str = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")
    embed_device: str = os.getenv("EMBED_DEVICE", "auto")  # auto | cpu | cuda | mps
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # Vector DB settings
    chroma_dir: str = os.getenv("CHROMA_DIR", "./storage/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "mm_rag")

    # Retrieval settings
    top_k_images: int = int(os.getenv("TOP_K_IMAGES", "4"))
    top_k_texts: int = int(os.getenv("TOP_K_TEXTS", "4"))
    fusion_alpha: float = float(os.getenv("FUSION_ALPHA", "0.7"))
    text_chunk_tokens: int = int(os.getenv("TEXT_CHUNK_TOKENS", "60"))
    text_chunk_overlap: int = int(os.getenv("TEXT_CHUNK_OVERLAP", "10"))

    # Generation (Ollama) settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "https://ollama.com")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2-vision:latest")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    # Data settings
    data_dir: str = os.getenv("DATA_DIR", "./data")
    images_per_breed: int = int(os.getenv("IMAGES_PER_BREED", "40"))
    hf_dataset: str = os.getenv("HF_DATASET", "timm/oxford-iiit-pet")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


# Global default configuration instance
settings = Settings()
