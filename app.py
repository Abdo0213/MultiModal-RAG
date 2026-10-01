"""Web Application server for Multimodal RAG with HTML/JS and Gradio support."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmrag.config import settings
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.store import ChromaStore

# Initialize FastAPI App
app = FastAPI(
    title="Multimodal RAG API",
    description="Multimodal RAG with CLIP embeddings, ChromaDB, and Ollama VLM",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
STATIC_DIR = ROOT_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Pipeline singleton instance
_pipeline: Optional[MultimodalRAGPipeline] = None


def get_pipeline() -> MultimodalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalRAGPipeline()
    return _pipeline


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the modern HTML frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Multimodal RAG API is running</h1>")


@app.get("/api/status")
async def get_status():
    """Return database and backend status."""
    store = ChromaStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    counts = store.count_by_modality()
    return {
        "status": "ready",
        "collection": settings.chroma_collection,
        "total": store.count(),
        "images": counts.get("image", 0),
        "texts": counts.get("text", 0),
        "model": settings.embed_model,
        "ollama_model": settings.ollama_model,
    }


@app.get("/api/sample-images")
async def get_sample_images():
    """Return a curated list of sample test images from the dataset."""
    images_dir = settings.data_path / "images"
    samples = []
    if images_dir.exists():
        for breed_dir in sorted(images_dir.iterdir()):
            if breed_dir.is_dir():
                jpgs = list(breed_dir.glob("*.jpg"))
                if jpgs:
                    rel_path = f"data/images/{breed_dir.name}/{jpgs[0].name}"
                    samples.append({
                        "breed": breed_dir.name.replace("_", " ").title(),
                        "slug": breed_dir.name,
                        "path": rel_path,
                        "url": f"/media/{rel_path}",
                    })
            if len(samples) >= 8:
                break
    return JSONResponse(content={"samples": samples})


@app.get("/media/{file_path:path}")
async def serve_media(file_path: str):
    """Serve dataset images or local files securely."""
    # Resolve relative to root or data dir
    target = Path(file_path)
    if not target.is_absolute():
        target = (ROOT_DIR / file_path).resolve()

    if target.exists() and target.is_file():
        return FileResponse(str(target))
    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.post("/api/query")
async def handle_query(
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    fusion: str = Form("mean"),
    alpha: float = Form(0.5),
    top_k_images: int = Form(4),
    top_k_texts: int = Form(4),
):
    """Execute multimodal RAG query and return generated answer and retrieved items."""
    pipeline = get_pipeline()

    pil_image = None
    if image is not None and image.filename:
        try:
            pil_image = Image.open(image.file).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image upload: {str(e)}"},
            )

    result = pipeline.answer(
        image=pil_image,
        text=text,
        fusion=fusion,
        alpha=alpha,
        top_k_images=top_k_images,
        top_k_texts=top_k_texts,
    )
    return JSONResponse(content=result)


def create_gradio_app():
    """Create Gradio UI as an alternative frontend."""
    import gradio as gr

    pipeline = get_pipeline()

    def gradio_predict(img, txt, fusion_type, alpha_val, k_img, k_txt):
        if img is None and (not txt or not txt.strip()):
            return "Please provide an image or text prompt.", [], []

        res = pipeline.answer(
            image=img,
            text=txt,
            fusion=fusion_type,
            alpha=float(alpha_val),
            top_k_images=int(k_img),
            top_k_texts=int(k_txt),
        )

        # Prepare gallery
        gallery = []
        for item in res.get("retrieved_images", []):
            p = item.get("metadata", {}).get("path")
            lbl = item.get("metadata", {}).get("label", "")
            score = item.get("score", 0.0)
            if p and Path(p).exists():
                gallery.append((p, f"{lbl} (score: {score:.3f})"))

        # Prepare text passages
        passages = []
        for idx, item in enumerate(res.get("retrieved_texts", []), 1):
            src = item.get("metadata", {}).get("source", "")
            lbl = item.get("metadata", {}).get("label", "")
            score = item.get("score", 0.0)
            doc = item.get("document", "")
            passages.append(f"**[text:{idx}]** ({lbl} | {src} | score: {score:.3f}):\n{doc}\n")

        formatted_passages = "\n---\n".join(passages) if passages else "No text retrieved."
        return res["answer"], gallery, formatted_passages

    with gr.Blocks(title="Multimodal RAG (CLIP + ChromaDB + Ollama)") as demo:
        gr.Markdown("# 🐾 Multimodal RAG Explorer\n*Unified CLIP Embeddings &bull; ChromaDB &bull; Ollama Vision LLM*")
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Query Image (Optional)")
                in_txt = gr.Textbox(lines=3, label="Text Query / Question", placeholder="e.g., Identify this breed and describe its care...")
                with gr.Accordion("Retrieval Settings", open=False):
                    in_fusion = gr.Radio(["mean", "rrf"], value="mean", label="Fusion Strategy")
                    in_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Fusion Alpha (Image Weight)")
                    in_k_img = gr.Slider(1, 10, value=4, step=1, label="Top-K Images")
                    in_k_txt = gr.Slider(1, 10, value=4, step=1, label="Top-K Text Chunks")
                btn_run = gr.Button("Submit Multimodal Query", variant="primary")

            with gr.Column():
                out_answer = gr.Markdown(label="Generated Answer with Citations")
                out_gallery = gr.Gallery(label="Retrieved Images", columns=2, height="auto")
                out_text = gr.Markdown(label="Retrieved Text Passages")

        btn_run.click(
            gradio_predict,
            inputs=[in_img, in_txt, in_fusion, in_alpha, in_k_img, in_k_txt],
            outputs=[out_answer, out_gallery, out_text],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Start Multimodal RAG Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface instead of FastAPI")
    args = parser.parse_args()

    if args.gradio:
        print(f"Launching Gradio UI at http://{args.host}:{args.port}...")
        demo = create_gradio_app()
        demo.launch(server_name=args.host, server_port=args.port)
    else:
        print(f"Starting Multimodal RAG Web App at http://{args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
