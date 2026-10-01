"""Multimodal retriever supporting image-only, text-only, and fused queries with per-modality retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from PIL import Image

from mmrag.config import settings
from mmrag.embedder import CLIPEmbedder, normalize_l2
from mmrag.store import ChromaStore


@dataclass
class RetrievalResult:
    images: List[Dict[str, Any]] = field(default_factory=list)
    texts: List[Dict[str, Any]] = field(default_factory=list)
    query_type: str = "image_only"
    fusion_mode: Optional[str] = None
    alpha: Optional[float] = None


class MultimodalRetriever:
    """
    Retriever for multimodal RAG supporting separate per-modality retrieval,
    embedding fusion (weighted mean), and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        embedder: Optional[CLIPEmbedder] = None,
        top_k_images: Optional[int] = None,
        top_k_texts: Optional[int] = None,
        default_fusion_alpha: Optional[float] = None,
    ):
        self.store = store or ChromaStore(
            persist_directory=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )
        self.embedder = embedder or CLIPEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
        )
        self.top_k_images = top_k_images or settings.top_k_images
        self.top_k_texts = top_k_texts or settings.top_k_texts
        self.default_fusion_alpha = (
            default_fusion_alpha if default_fusion_alpha is not None else settings.fusion_alpha
        )

    def _embed_query_images(
        self, images: Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]]
    ) -> np.ndarray:
        """Embed one or multiple query images and return normalized average vector."""
        if not isinstance(images, (list, tuple)):
            images = [images]

        embs = self.embedder.embed_images(images)
        if len(embs) == 1:
            return embs[0]
        # Average multiple images and L2-normalize
        avg_emb = np.mean(embs, axis=0, keepdims=True)
        return normalize_l2(avg_emb)[0]

    def _embed_query_text(self, text: str) -> np.ndarray:
        """Embed a query text string and return normalized vector."""
        embs = self.embedder.embed_texts([text])
        return embs[0]

    def _fuse_embeddings_mean(
        self, img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Fused embedding: normalize(alpha * img_emb + (1 - alpha) * txt_emb).
        """
        fused = alpha * img_emb + (1.0 - alpha) * txt_emb
        fused = fused.reshape(1, -1)
        return normalize_l2(fused)[0]

    def _reciprocal_rank_fusion(
        self,
        list_a: List[Dict[str, Any]],
        list_b: List[Dict[str, Any]],
        top_k: int,
        k_const: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion (RRF).
        Score(d) = sum(1 / (k_const + rank))
        """
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(list_a, start=1):
            doc_id = item["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_const + rank))
            item_map[doc_id] = item

        for rank, item in enumerate(list_b, start=1):
            doc_id = item["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_const + rank))
            if doc_id not in item_map:
                item_map[doc_id] = item

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        merged_results = []
        for doc_id in sorted_ids:
            res = dict(item_map[doc_id])
            res["rrf_score"] = rrf_scores[doc_id]
            merged_results.append(res)

        return merged_results

    def retrieve(
        self,
        image: Optional[Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]]] = None,
        text: Optional[str] = None,
        fusion: Literal["mean", "rrf"] = "mean",
        alpha: Optional[float] = None,
        top_k_images: Optional[int] = None,
        top_k_texts: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Execute retrieval based on provided modalities (image only, image + text, or text only).
        Retrieves top_k_images and top_k_texts separately to avoid modality gap imbalance.
        """
        k_img = top_k_images or self.top_k_images
        k_txt = top_k_texts or self.top_k_texts
        alpha_val = self.default_fusion_alpha if alpha is None else alpha

        has_image = image is not None and (not isinstance(image, (list, tuple)) or len(image) > 0)
        has_text = text is not None and len(text.strip()) > 0

        if not has_image and not has_text:
            raise ValueError("At least one query modality (image or text) must be provided.")

        # Case 1: Image only
        if has_image and not has_text:
            q_emb = self._embed_query_images(image)
            ret_images = self.store.query(q_emb, modality="image", k=k_img)
            ret_texts = self.store.query(q_emb, modality="text", k=k_txt)
            return RetrievalResult(
                images=ret_images,
                texts=ret_texts,
                query_type="image_only",
            )

        # Case 2: Text only
        if has_text and not has_image:
            q_emb = self._embed_query_text(text)
            ret_images = self.store.query(q_emb, modality="image", k=k_img)
            ret_texts = self.store.query(q_emb, modality="text", k=k_txt)
            return RetrievalResult(
                images=ret_images,
                texts=ret_texts,
                query_type="text_only",
            )

        # Case 3: Image + Text Multimodal Fusion
        img_emb = self._embed_query_images(image)
        txt_emb = self._embed_query_text(text)

        if fusion == "mean":
            fused_emb = self._fuse_embeddings_mean(img_emb, txt_emb, alpha=alpha_val)
            ret_images = self.store.query(fused_emb, modality="image", k=k_img)
            ret_texts = self.store.query(fused_emb, modality="text", k=k_txt)
            return RetrievalResult(
                images=ret_images,
                texts=ret_texts,
                query_type="multimodal",
                fusion_mode="mean",
                alpha=alpha_val,
            )

        elif fusion == "rrf":
            # Image query results
            img_q_images = self.store.query(img_emb, modality="image", k=k_img * 2)
            img_q_texts = self.store.query(img_emb, modality="text", k=k_txt * 2)

            # Text query results
            txt_q_images = self.store.query(txt_emb, modality="image", k=k_img * 2)
            txt_q_texts = self.store.query(txt_emb, modality="text", k=k_txt * 2)

            # Fuse ranked lists per modality
            ret_images = self._reciprocal_rank_fusion(img_q_images, txt_q_images, top_k=k_img)
            ret_texts = self._reciprocal_rank_fusion(img_q_texts, txt_q_texts, top_k=k_txt)

            return RetrievalResult(
                images=ret_images,
                texts=ret_texts,
                query_type="multimodal",
                fusion_mode="rrf",
                alpha=alpha_val,
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}. Choose 'mean' or 'rrf'.")
