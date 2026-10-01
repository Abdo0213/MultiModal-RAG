"""ChromaDB storage layer with metadata filtering and cosine similarity."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
import numpy as np


class ChromaStore:
    """
    Wrapper around ChromaDB PersistentClient for multimodal vector storage.
    Configured with cosine distance space and metadata filtering per modality.
    """

    def __init__(
        self,
        persist_directory: str | Path = "./storage/chroma",
        collection_name: str = "mm_rag",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_existing_ids(self) -> Set[str]:
        """Return the set of all item IDs currently stored in the collection."""
        res = self.collection.get(include=[])
        return set(res.get("ids", []))

    def add_items(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add or update items into ChromaDB in batches.
        """
        if not ids:
            return

        # Ensure embeddings is a list of lists of floats
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings

        # Chroma batch size limit safety (5000 max per batch)
        batch_size = 2000
        for i in range(0, len(ids), batch_size):
            b_ids = ids[i : i + batch_size]
            b_emb = embeddings_list[i : i + batch_size]
            b_docs = documents[i : i + batch_size]
            b_meta = metadatas[i : i + batch_size]

            self.collection.upsert(
                ids=b_ids,
                embeddings=b_emb,
                documents=b_docs,
                metadatas=b_meta,
            )

    def query(
        self,
        query_embedding: np.ndarray,
        modality: Optional[str] = None,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Query vector DB with a single 1D/2D embedding.
        If modality is specified ('image' or 'text'), filters results by where={"modality": modality}.
        Returns list of dicts: {id, document, metadata, score (cosine similarity 0..1), distance}.
        """
        if query_embedding.ndim == 1:
            q_emb = [query_embedding.tolist()]
        elif query_embedding.ndim == 2:
            q_emb = query_embedding.tolist()
        else:
            raise ValueError(f"Invalid query embedding dimensions: {query_embedding.shape}")

        where_filter = {"modality": modality} if modality else None

        # Chroma will error if n_results is greater than available items
        total_in_collection = self.collection.count()
        if total_in_collection == 0:
            return []

        n_results = min(k, total_in_collection)

        try:
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # Fallback if no items match filter
            print(f"[ChromaStore query warning] {e}")
            return []

        out: List[Dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return out

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for item_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance = 1 - cosine_similarity. Similarity = 1 - distance.
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            out.append({
                "id": item_id,
                "document": doc,
                "metadata": meta,
                "distance": float(dist),
                "score": similarity,
            })

        return out

    def count(self) -> int:
        """Total number of items in collection."""
        return self.collection.count()

    def count_by_modality(self) -> Dict[str, int]:
        """Count items per modality."""
        try:
            img_res = self.collection.get(where={"modality": "image"}, include=[])
            img_count = len(img_res.get("ids", []))
            txt_res = self.collection.get(where={"modality": "text"}, include=[])
            txt_count = len(txt_res.get("ids", []))
            return {"image": img_count, "text": txt_count}
        except Exception:
            return {"image": 0, "text": 0}

    def reset(self) -> None:
        """Delete and re-create collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
