"""Tests for Multimodal Retriever."""
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.embedder import normalize_l2
from mmrag.retriever import MultimodalRetriever
from mmrag.store import ChromaStore


class DummyEmbedder:
    """Lightweight dummy embedder for retriever unit tests."""
    def __init__(self):
        self.device = "cpu"
        self.model_name = "dummy-clip"

    def embed_images(self, images, batch_size=None):
        np.random.seed(len(images) + 10)
        return normalize_l2(np.random.randn(len(images), 512).astype(np.float32))

    def embed_texts(self, texts, batch_size=None):
        np.random.seed(len(texts) + 20)
        return normalize_l2(np.random.randn(len(texts), 512).astype(np.float32))


@pytest.fixture
def retriever_setup(tmp_path):
    chroma_dir = tmp_path / "chroma_retriever_test"
    store = ChromaStore(persist_directory=chroma_dir, collection_name="test_retriever")
    embedder = DummyEmbedder()

    # Pre-populate store
    np.random.seed(100)
    img_embs = normalize_l2(np.random.randn(5, 512).astype(np.float32))
    txt_embs = normalize_l2(np.random.randn(5, 512).astype(np.float32))

    store.add_items(
        ids=[f"img_{i}" for i in range(5)],
        embeddings=img_embs,
        documents=[f"Image {i}" for i in range(5)],
        metadatas=[{"modality": "image", "label": f"breed_{i}"} for i in range(5)],
    )
    store.add_items(
        ids=[f"txt_{i}" for i in range(5)],
        embeddings=txt_embs,
        documents=[f"Text chunk {i}" for i in range(5)],
        metadatas=[{"modality": "text", "label": f"breed_{i}"} for i in range(5)],
    )

    retriever = MultimodalRetriever(
        store=store,
        embedder=embedder,
        top_k_images=2,
        top_k_texts=2,
        default_fusion_alpha=0.5,
    )
    yield retriever
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_retriever_mean_fusion(retriever_setup):
    """Test multimodal query with mean fusion."""
    from PIL import Image
    dummy_img = Image.new("RGB", (32, 32))
    res = retriever_setup.retrieve(
        image=dummy_img,
        text="query about pets",
        fusion="mean",
        alpha=0.6,
        top_k_images=2,
        top_k_texts=2,
    )

    assert res.query_type == "multimodal"
    assert res.fusion_mode == "mean"
    assert len(res.images) == 2
    assert len(res.texts) == 2
    for img in res.images:
        assert img["metadata"]["modality"] == "image"
    for txt in res.texts:
        assert txt["metadata"]["modality"] == "text"


def test_retriever_rrf_fusion(retriever_setup):
    """Test multimodal query with Reciprocal Rank Fusion (RRF)."""
    from PIL import Image
    dummy_img = Image.new("RGB", (32, 32))
    res = retriever_setup.retrieve(
        image=dummy_img,
        text="query about pets",
        fusion="rrf",
        top_k_images=2,
        top_k_texts=2,
    )

    assert res.query_type == "multimodal"
    assert res.fusion_mode == "rrf"
    assert len(res.images) <= 2
    assert len(res.texts) <= 2
    for img in res.images:
        assert "rrf_score" in img
