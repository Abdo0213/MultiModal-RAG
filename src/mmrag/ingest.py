"""Ingestion pipeline: reads manifest -> chunks text -> embeds -> upserts to ChromaDB."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from mmrag.chunking import CLIPChunker
from mmrag.config import settings
from mmrag.embedder import CLIPEmbedder
from mmrag.store import ChromaStore


def load_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    """Load records from a jsonl manifest file."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ingest_corpus(
    manifest_path: Optional[str | Path] = None,
    store: Optional[ChromaStore] = None,
    embedder: Optional[CLIPEmbedder] = None,
    chunker: Optional[CLIPChunker] = None,
    batch_size: Optional[int] = None,
    force_reset: bool = False,
) -> Dict[str, int]:
    """
    Ingest images and text articles from manifest into ChromaDB.
    Idempotent and resumable.
    """
    manifest_file = Path(manifest_path or (settings.data_path / "manifest.jsonl"))
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_file}. Please run data prep first."
        )

    records = load_manifest(manifest_file)
    print(f"Loaded {len(records)} entries from manifest '{manifest_file}'.")

    # Initialize store, embedder, chunker
    if store is None:
        store = ChromaStore(
            persist_directory=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )

    if force_reset:
        print("[Ingest] Resetting existing ChromaDB collection...")
        store.reset()

    existing_ids = store.get_existing_ids()
    print(f"Found {len(existing_ids)} existing documents in collection '{store.collection_name}'.")

    if embedder is None:
        embedder = CLIPEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=batch_size or settings.embed_batch_size,
        )

    if chunker is None:
        chunker = CLIPChunker(
            model_name=settings.embed_model,
            max_chunk_tokens=settings.text_chunk_tokens,
            overlap_tokens=settings.text_chunk_overlap,
        )

    # Separate modalities
    image_records = [r for r in records if r.get("modality") == "image"]
    text_records = [r for r in records if r.get("modality") == "text"]

    # 1. Ingest Images
    new_image_records = [r for r in image_records if r["id"] not in existing_ids]
    print(f"Ingesting images: {len(new_image_records)} new out of {len(image_records)} total...")

    bs = batch_size or settings.embed_batch_size
    ingested_images = 0

    if new_image_records:
        for i in range(0, len(new_image_records), bs):
            batch = new_image_records[i : i + bs]
            ids = [r["id"] for r in batch]
            paths = [r["path"] for r in batch]
            metas = [
                {
                    "modality": "image",
                    "path": r["path"],
                    "label": r.get("label", ""),
                    "source": r.get("source", ""),
                }
                for r in batch
            ]
            docs = [f"Image file of {r.get('label', 'subject')}" for r in batch]

            embeddings = embedder.embed_images(paths, batch_size=len(batch))
            store.add_items(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            ingested_images += len(batch)
            print(f"  [Images] Ingested {ingested_images}/{len(new_image_records)}...", end="\r")
        print(f"\n  [Images] Completed {ingested_images} image embeddings.")

    # 2. Ingest Text Documents (Chunked)
    print(f"Processing text articles: {len(text_records)} documents...")
    all_chunks_to_embed = []
    chunk_ids = []
    chunk_metas = []
    chunk_docs = []

    for r in tqdm(text_records, desc="Chunking text documents"):
        file_path = Path(r["path"])
        if not file_path.exists():
            print(f"[warn] Text file missing: {file_path}")
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        chunks = chunker.chunk_text(raw_text)

        for chk in chunks:
            chk_id = f"{r['id']}_chk_{chk.chunk_index}"
            if chk_id in existing_ids:
                continue

            chunk_ids.append(chk_id)
            all_chunks_to_embed.append(chk.text)
            chunk_docs.append(chk.text)
            chunk_metas.append({
                "modality": "text",
                "parent_id": r["id"],
                "chunk_index": chk.chunk_index,
                "token_count": chk.token_count,
                "label": r.get("label", ""),
                "source": r.get("source", ""),
                "path": r["path"],
            })

    ingested_text_chunks = 0
    if all_chunks_to_embed:
        print(f"Embedding and upserting {len(all_chunks_to_embed)} text chunks...")
        for i in range(0, len(all_chunks_to_embed), bs):
            b_ids = chunk_ids[i : i + bs]
            b_texts = all_chunks_to_embed[i : i + bs]
            b_metas = chunk_metas[i : i + bs]
            b_docs = chunk_docs[i : i + bs]

            embeddings = embedder.embed_texts(b_texts, batch_size=len(b_texts))
            store.add_items(ids=b_ids, embeddings=embeddings, documents=b_docs, metadatas=b_metas)
            ingested_text_chunks += len(b_ids)
            print(f"  [Texts] Ingested {ingested_text_chunks}/{len(all_chunks_to_embed)} chunks...", end="\r")
        print(f"\n  [Texts] Completed {ingested_text_chunks} text chunk embeddings.")

    summary = store.count_by_modality()
    print(f"\n[Ingest Summary] Store now contains {summary.get('image', 0)} images and {summary.get('text', 0)} text chunks.")
    return {"new_images": ingested_images, "new_text_chunks": ingested_text_chunks, "total": store.count()}


if __name__ == "__main__":
    ingest_corpus()
