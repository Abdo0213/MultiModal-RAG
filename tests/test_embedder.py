"""Tests for CLIP Embedder and normalization."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.embedder import CLIPEmbedder, normalize_l2


@pytest.fixture(scope="module")
def embedder():
    return CLIPEmbedder(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=4)


def test_vector_l2_normalization():
    """Verify that normalize_l2 yields unit vectors (norm = 1.0)."""
    random_vecs = np.random.randn(5, 512).astype(np.float32)
    normalized = normalize_l2(random_vecs)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_text_embedding_properties(embedder):
    """Verify that text embeddings are 2D float32 arrays with unit L2 norm."""
    texts = ["A fluffy Persian cat on a sofa.", "A friendly Golden Retriever playing outside."]
    embs = embedder.embed_texts(texts)

    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 512
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_image_embedding_properties(embedder):
    """Verify that image embeddings are 2D float32 arrays with unit L2 norm."""
    img1 = Image.new("RGB", (64, 64), color="red")
    img2 = Image.new("RGB", (64, 64), color="blue")
    embs = embedder.embed_images([img1, img2])

    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 512
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
