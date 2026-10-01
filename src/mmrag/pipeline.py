"""End-to-end Multimodal RAG pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from PIL import Image

from mmrag.config import settings
from mmrag.embedder import CLIPEmbedder
from mmrag.generator import OllamaGenerator
from mmrag.retriever import MultimodalRetriever, RetrievalResult
from mmrag.store import ChromaStore


class MultimodalRAGPipeline:
    """
    Unified multimodal RAG pipeline for retrieval and augmented vision-LLM generation.
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        embedder: Optional[CLIPEmbedder] = None,
        retriever: Optional[MultimodalRetriever] = None,
        generator: Optional[OllamaGenerator] = None,
    ):
        self.store = store or ChromaStore(
            persist_directory=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )
        self.embedder = embedder or CLIPEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
        self.retriever = retriever or MultimodalRetriever(
            store=self.store,
            embedder=self.embedder,
            top_k_images=settings.top_k_images,
            top_k_texts=settings.top_k_texts,
            default_fusion_alpha=settings.fusion_alpha,
        )
        self.generator = generator or OllamaGenerator(
            host=settings.ollama_host,
            api_key=settings.ollama_api_key,
            model_name=settings.ollama_model,
            temperature=settings.ollama_temperature,
        )

    def answer(
        self,
        image: Optional[Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]]] = None,
        text: Optional[str] = None,
        fusion: Literal["mean", "rrf"] = "mean",
        alpha: Optional[float] = None,
        top_k_images: Optional[int] = None,
        top_k_texts: Optional[int] = None,
        generate_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end multimodal query:
        1. Retrieve top-k images and top-k text chunks separately.
        2. Format prompt with citations and call vision LLM.
        3. Return structured response with answer, citations, and scores.
        """
        t0 = time.time()
        retrieval_res: RetrievalResult = self.retriever.retrieve(
            image=image,
            text=text,
            fusion=fusion,
            alpha=alpha,
            top_k_images=top_k_images,
            top_k_texts=top_k_texts,
        )
        t_retrieval = time.time() - t0

        t1 = time.time()
        if generate_answer:
            gen_res = self.generator.generate(
                query_images=image,
                query_text=text,
                retrieved_images=retrieval_res.images,
                retrieved_texts=retrieval_res.texts,
            )
            answer_text = gen_res["answer"]
            gen_status = gen_res.get("status", "success")
        else:
            answer_text = "Generation skipped (retrieval only)."
            gen_status = "skipped"
        t_generation = time.time() - t1

        return {
            "answer": answer_text,
            "status": gen_status,
            "retrieved_images": retrieval_res.images,
            "retrieved_texts": retrieval_res.texts,
            "query_info": {
                "query_type": retrieval_res.query_type,
                "fusion_mode": retrieval_res.fusion_mode,
                "alpha": retrieval_res.alpha,
                "has_image": image is not None,
                "has_text": bool(text and text.strip()),
            },
            "latency": {
                "retrieval_sec": round(t_retrieval, 4),
                "generation_sec": round(t_generation, 4),
                "total_sec": round(t_retrieval + t_generation, 4),
            },
        }
