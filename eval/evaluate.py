"""Evaluation script for Multimodal RAG retrieval benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src/ is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.config import settings
from mmrag.embedder import CLIPEmbedder
from mmrag.retriever import MultimodalRetriever
from mmrag.store import ChromaStore


def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    """Load test query entries from test_queries.jsonl."""
    if not path.exists():
        raise FileNotFoundError(f"Test queries not found at: {path}")
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    return queries


def calculate_metrics(retrieved_items: List[Dict[str, Any]], ground_truth_label: str, k: int) -> Dict[str, float]:
    """
    Compute Recall@k (Hit@k: 1 if at least one correct label in top k)
    and Precision@k (fraction of top k that match ground truth).
    """
    top_k_items = retrieved_items[:k]
    if not top_k_items:
        return {"recall@k": 0.0, "precision@k": 0.0}

    matches = [
        1.0 if item.get("metadata", {}).get("label") == ground_truth_label else 0.0
        for item in top_k_items
    ]
    recall_at_k = 1.0 if sum(matches) > 0 else 0.0
    precision_at_k = sum(matches) / len(top_k_items)

    return {"recall@k": recall_at_k, "precision@k": precision_at_k}


def run_evaluation(
    model_name: str = settings.embed_model,
    limit_queries: int = 100,
    k_values: List[int] = [1, 3, 5],
    output_path: Path = ROOT_DIR / "eval" / "results.json",
) -> Dict[str, Any]:
    """Run full benchmark matrix across query modalities and fusion strategies."""
    test_queries_file = settings.data_path / "test_queries.jsonl"
    if not test_queries_file.exists():
        # Fallback to manifest if test split not created yet
        test_queries_file = settings.data_path / "manifest.jsonl"
        print(f"[warn] test_queries.jsonl not found, sampling from {test_queries_file}...")

    queries = load_test_queries(test_queries_file)
    # Filter only image queries with existing paths
    img_queries = [q for q in queries if q.get("modality") == "image" and Path(q.get("path", "")).exists()]
    if limit_queries and len(img_queries) > limit_queries:
        img_queries = img_queries[:limit_queries]

    print(f"\n=======================================================")
    print(f"Starting Multimodal RAG Evaluation")
    print(f"Model: {model_name}")
    print(f"Test queries: {len(img_queries)}")
    print(f"=======================================================\n")

    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    embedder = CLIPEmbedder(model_name=model_name, device=settings.embed_device)
    retriever = MultimodalRetriever(store=store, embedder=embedder)

    max_k = max(k_values)
    results: Dict[str, Any] = {
        "config": {
            "model_name": model_name,
            "num_queries": len(img_queries),
            "collection": settings.chroma_collection,
        },
        "experiments": {},
    }

    # Helper evaluator
    def evaluate_strategy(name: str, get_results_fn):
        print(f"Evaluating strategy: {name} ...")
        metric_sums: Dict[str, float] = {}

        for q in img_queries:
            query_label = q.get("label", "")
            img_path = q.get("path", "")

            ret_res = get_results_fn(img_path, query_label)

            for k in k_values:
                # Image->Image metrics
                img_metrics = calculate_metrics(ret_res.images, query_label, k)
                metric_sums[f"img2img_recall@{k}"] = metric_sums.get(f"img2img_recall@{k}", 0.0) + img_metrics["recall@k"]
                metric_sums[f"img2img_prec@{k}"] = metric_sums.get(f"img2img_prec@{k}", 0.0) + img_metrics["precision@k"]

                # Image->Text metrics
                txt_metrics = calculate_metrics(ret_res.texts, query_label, k)
                metric_sums[f"img2txt_recall@{k}"] = metric_sums.get(f"img2txt_recall@{k}", 0.0) + txt_metrics["recall@k"]
                metric_sums[f"img2txt_prec@{k}"] = metric_sums.get(f"img2txt_prec@{k}", 0.0) + txt_metrics["precision@k"]

        n = max(1, len(img_queries))
        averaged = {k: round(v / n, 4) for k, v in metric_sums.items()}
        results["experiments"][name] = averaged
        return averaged

    # 1. Image-only retrieval
    evaluate_strategy(
        "Image_Only",
        lambda img_path, label: retriever.retrieve(image=img_path, top_k_images=max_k, top_k_texts=max_k),
    )

    # 2. Image + Text (Mean fusion with different alphas)
    for alpha in [0.25, 0.50, 0.75]:
        evaluate_strategy(
            f"Image+Text_Mean_alpha_{alpha:.2f}",
            lambda img_path, label, a=alpha: retriever.retrieve(
                image=img_path,
                text="Information and breed characteristics for this pet",
                fusion="mean",
                alpha=a,
                top_k_images=max_k,
                top_k_texts=max_k,
            ),
        )

    # 3. Image + Text (RRF fusion)
    evaluate_strategy(
        "Image+Text_RRF",
        lambda img_path, label: retriever.retrieve(
            image=img_path,
            text="Information and breed characteristics for this pet",
            fusion="rrf",
            top_k_images=max_k,
            top_k_texts=max_k,
        ),
    )

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Experiment':<30} | {'I->I R@1':<8} | {'I->I R@5':<8} | {'I->T R@1':<8} | {'I->T R@5':<8}")
    print("=" * 80)
    for exp_name, metrics in results["experiments"].items():
        i2i_r1 = metrics.get("img2img_recall@1", 0.0)
        i2i_r5 = metrics.get("img2img_recall@5", 0.0)
        i2t_r1 = metrics.get("img2txt_recall@1", 0.0)
        i2t_r5 = metrics.get("img2txt_recall@5", 0.0)
        print(f"{exp_name:<30} | {i2i_r1:<8.3f} | {i2i_r5:<8.3f} | {i2t_r1:<8.3f} | {i2t_r5:<8.3f}")
    print("=" * 80)
    print(f"Results successfully saved to '{output_path}'.\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal retrieval")
    parser.add_argument("--model", type=str, default=settings.embed_model, help="Hugging Face CLIP model name")
    parser.add_argument("--limit", type=int, default=100, help="Max test queries to run")
    parser.add_argument("--out", type=str, default=str(ROOT_DIR / "eval" / "results.json"), help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(model_name=args.model, limit_queries=args.limit, output_path=Path(args.out))
