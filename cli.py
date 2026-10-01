"""Command Line Interface for Multimodal RAG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is in python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.config import settings
from mmrag.ingest import ingest_corpus
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.store import ChromaStore


def cmd_ingest(args):
    """Run corpus ingestion into ChromaDB."""
    print("=== Multimodal RAG: Ingestion ===")
    manifest_path = args.manifest or (settings.data_path / "manifest.jsonl")
    ingest_corpus(
        manifest_path=manifest_path,
        force_reset=args.reset,
        batch_size=args.batch_size,
    )


def cmd_status(args):
    """Display vector DB statistics."""
    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    counts = store.count_by_modality()
    print("=== Multimodal RAG: ChromaDB Status ===")
    print(f"Collection: '{settings.chroma_collection}' at '{settings.chroma_dir}'")
    print(f"Total documents: {store.count()}")
    print(f"  - Images: {counts.get('image', 0)}")
    print(f"  - Text chunks: {counts.get('text', 0)}")


def cmd_ask(args):
    """Query the multimodal RAG system."""
    if not args.image and not args.text:
        print("[Error] Please provide at least --image or --text.")
        sys.exit(1)

    print("=== Multimodal RAG: Query ===")
    if args.image:
        print(f"Image Query: {args.image}")
    if args.text:
        print(f"Text Query:  {args.text}")
    print(f"Fusion Mode: {args.fusion} (alpha={args.alpha})")
    print("Executing query...\n")

    pipeline = MultimodalRAGPipeline()
    result = pipeline.answer(
        image=args.image,
        text=args.text,
        fusion=args.fusion,
        alpha=args.alpha,
        top_k_images=args.top_k_images,
        top_k_texts=args.top_k_texts,
        generate_answer=not args.no_gen,
    )

    print("=" * 60)
    print("RETRIEVED IMAGES:")
    print("=" * 60)
    for idx, img in enumerate(result["retrieved_images"], 1):
        score_info = f"score: {img.get('score', 0):.4f}"
        if "rrf_score" in img:
            score_info += f" | rrf: {img['rrf_score']:.5f}"
        print(f"[{idx}] {img.get('id')} | Label: {img.get('metadata', {}).get('label')} | Path: {img.get('metadata', {}).get('path')} ({score_info})")

    print("\n" + "=" * 60)
    print("RETRIEVED TEXT PASSAGES:")
    print("=" * 60)
    for idx, txt in enumerate(result["retrieved_texts"], 1):
        score_info = f"score: {txt.get('score', 0):.4f}"
        if "rrf_score" in txt:
            score_info += f" | rrf: {txt['rrf_score']:.5f}"
        print(f"[{idx}] {txt.get('id')} | Label: {txt.get('metadata', {}).get('label')} | Source: {txt.get('metadata', {}).get('source')} ({score_info})")
        print(f"    Excerpt: {txt.get('document', '')[:160]}...\n")

    print("=" * 60)
    print("GENERATED ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"Latencies: Retrieval={result['latency']['retrieval_sec']}s | Generation={result['latency']['generation_sec']}s | Total={result['latency']['total_sec']}s")


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG CLI (CLIP + ChromaDB + Ollama)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest manifest images and text into ChromaDB")
    ingest_parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.jsonl")
    ingest_parser.add_argument("--reset", action="store_true", help="Reset existing collection before ingestion")
    ingest_parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding")
    ingest_parser.set_defaults(func=cmd_ingest)

    # Status subcommand
    status_parser = subparsers.add_parser("status", help="Show ChromaDB collection statistics")
    status_parser.set_defaults(func=cmd_status)

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Query the RAG pipeline with image, text, or both")
    ask_parser.add_argument("--image", type=str, default=None, help="Path to input query image")
    ask_parser.add_argument("--text", type=str, default=None, help="Input query text / prompt")
    ask_parser.add_argument("--fusion", choices=["mean", "rrf"], default="mean", help="Multimodal fusion strategy")
    ask_parser.add_argument("--alpha", type=float, default=settings.fusion_alpha, help="Weight for image embedding (0..1)")
    ask_parser.add_argument("--top-k-images", type=int, default=settings.top_k_images, help="Number of images to retrieve")
    ask_parser.add_argument("--top-k-texts", type=int, default=settings.top_k_texts, help="Number of text chunks to retrieve")
    ask_parser.add_argument("--no-gen", action="store_true", help="Skip Ollama answer generation (retrieval only)")
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
