"""Tests for CLIP chunking module."""
import sys
from pathlib import Path
import pytest

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.chunking import CLIPChunker


@pytest.fixture(scope="module")
def chunker():
    return CLIPChunker(
        model_name="openai/clip-vit-base-patch32",
        max_chunk_tokens=60,
        overlap_tokens=10,
    )


def test_chunking_token_limit(chunker):
    """Verify that every chunk produced has <= 77 CLIP tokens (and <= max_chunk_tokens)."""
    long_article = """
    The Persian cat is a long-haired breed of cat characterized by a round face and short muzzle.
    The name refers to Iran, the country formerly known as Persia, where cats similar to the modern
    Persian breed were first discovered. Since the late 19th century, cat fanciers in Great Britain
    and North America have bred them for extreme facial flat features and extraordinarily thick,
    luxurious coats. The cat is quiet and affectionate, making it an ideal indoor companion for families.
    Grooming must be conducted daily to prevent tangling and hairballs. Selective breeding has favored
    round heads, short faces, snub noses, chubby cheeks, and small, rounded ears.
    """ * 3

    chunks = chunker.chunk_text(long_article)
    assert len(chunks) > 1

    for chk in chunks:
        # Strict hard limit for CLIP is 77 tokens
        assert chk.token_count <= 77, f"Chunk exceeded 77 tokens: {chk.token_count}"
        # Chunker target limit
        assert chk.token_count <= 75


def test_chunking_empty_and_short(chunker):
    """Test chunking with empty string or single short sentence."""
    assert chunker.chunk_text("") == []

    single_sentence = "The Bengal cat has a wild leopard-like appearance."
    chunks = chunker.chunk_text(single_sentence)
    assert len(chunks) == 1
    assert chunks[0].text == single_sentence
    assert chunks[0].token_count < 60
