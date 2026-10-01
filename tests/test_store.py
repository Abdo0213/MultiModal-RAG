"""Tests for ChromaStore wrapper."""
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.embedder import normalize_l2
from mmrag.store import ChromaStore


@pytest.fixture
def temp_chroma_dir(tmp_path):
    chroma_path = tmp_path / "chroma_test"
    yield chroma_path
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)


def test_chroma_add_and_query(temp_chroma_dir):
    """Test inserting items and querying by modality."""
    store = ChromaStore(persist_directory=temp_chroma_dir, collection_name="test_col")

    # Generate synthetic embeddings (512-dim)
    np.random.seed(42)
    embs = normalize_l2(np.random.randn(4, 512).astype(np.float32))

    ids = ["img_1", "img_2", "txt_1", "txt_2"]
    docs = ["Image 1", "Image 2", "Text passage 1", "Text passage 2"]
    metas = [
        {"modality": "image", "label": "persian"},
        {"modality": "image", "label": "beagle"},
        {"modality": "text", "label": "persian"},
        {"modality": "text", "label": "beagle"},
    ]

    store.add_items(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    assert store.count() == 4
    assert store.count_by_modality() == {"image": 2, "text": 2}

    # Query with the exact embedding of img_1
    results_img = store.query(query_embedding=embs[0], modality="image", k=1)
    assert len(results_img) == 1
    assert results_img[0]["id"] == "img_1"
    assert results_img[0]["metadata"]["label"] == "persian"
    assert results_img[0]["score"] > 0.99  # Exact match cosine similarity ~ 1.0

    # Query for texts only
    results_txt = store.query(query_embedding=embs[0], modality="text", k=2)
    assert len(results_txt) == 2
    for r in results_txt:
        assert r["metadata"]["modality"] == "text"
