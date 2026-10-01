# Multimodal RAG (CLIP + ChromaDB + Ollama)

A multimodal Retrieval-Augmented Generation (RAG) system that unifies **images** and **independent text documents** (Wikipedia articles) into a single, shared embedding space using **CLIP**, indexed with **ChromaDB**, and synthesized with vision-capable LLMs via **Ollama Cloud / Local**.

---

## 🌟 Overview & Capabilities

Traditional RAG systems only index text. This system indexes both **visual documents (photos)** and **independent prose documents (Wikipedia articles)** within a shared vector space, enabling true cross-modal search:

1. **Image-Only Queries:** Upload a pet photo $\rightarrow$ retrieves visually similar images **and** relevant Wikipedia text passages explaining the breed, care, and origin.
2. **Image + Text Fusion Queries:** Upload an image alongside a specific question (e.g. *"What is the grooming routine and origin of this pet?"*) $\rightarrow$ retrieves cross-modal knowledge using weighted vector fusion or Reciprocal Rank Fusion (RRF).
3. **Text-Only Queries:** Query with natural language text $\rightarrow$ retrieves both matching images and relevant text articles.
4. **Grounded Generation with Citations:** Passes query inputs, retrieved images, and text chunks to Ollama to synthesize accurate, hallucination-resistant answers with clickable source citations (e.g., `[text:1]`, `[image:2]`).

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   User Query (Img/Txt) │
                                  └───────────┬────────────┘
                                              │
                                     [ CLIP Embedder ]
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Multimodal Vector Retrieval   │
                             │  (ChromaDB Cosine Space, Top-K) │
                             └────────┬───────────────┬────────┘
                                      │               │
                            (Top-K Images)     (Top-K Text Chunks)
                                      │               │
                                      ▼               ▼
                             ┌─────────────────────────────────┐
                             │  Prompt Builder & Downscaler   │
                             │  (Context + Image Encodings)    │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                                     [ Ollama Cloud / VLM ]
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │   Cited Answer & Visual Gallery │
                             └─────────────────────────────────┘
```

### Key Technical Decisions
* **Shared Embedding Space:** Powered by `openai/clip-vit-base-patch32` (PyTorch + Hugging Face `transformers`).
* **CLIP Token-Aware Chunking:** Enforces strict $\le 77$ token chunking limits with sentence boundary packing and overlap.
* **Modality-Gap Mitigation:** Performs separate per-modality top-K searches (`where={"modality": ...}`) to prevent high image-image similarities from drowning out image-text pairs.
* **Fusion Algorithms:** Supports both **Normalized Weighted Mean** ($\alpha$) and **Reciprocal Rank Fusion (RRF)**.
* **Modern Glassmorphic Web App:** Built with FastAPI + Vanilla CSS/JS featuring quick-select test pet thumbnails, cosine similarity progress bars, and citation linking.

---

## 📁 Repository Structure

```
multimodal-rag/
├── README.md                   # Project overview & running instructions
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .env                        # Active environment configuration
├── scripts/
│   └── prepare_data.py         # Dataset preparation & 10% evaluation split
├── src/mmrag/
│   ├── config.py               # Typed configuration loader
│   ├── chunking.py             # Sentence boundary & CLIP token chunker
│   ├── embedder.py             # CLIPEmbedder with auto-device & L2-normalization
│   ├── store.py                # ChromaDB persistent collection wrapper
│   ├── ingest.py               # Resumable corpus ingestion pipeline
│   ├── retriever.py            # Multimodal retriever (Image, Text, Mean & RRF Fusion)
│   ├── generator.py            # Ollama client with image downscaling & citations
│   └── pipeline.py             # End-to-end RAG coordinator
├── static/
│   ├── index.html              # Glassmorphic web UI
│   ├── style.css               # Design system & animations
│   └── app.js                  # Frontend interactive logic & citation navigation
├── app.py                      # FastAPI server & Gradio alternative launcher
├── cli.py                      # Command-line interface tool
├── eval/
│   ├── evaluate.py             # Evaluation benchmark suite
│   └── results.json            # Metric output reports
└── tests/                      # Pytest suite for core components
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

```bash
# Clone or navigate to the repository
cd "Task 1 - Mutli Modal RAG"

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env` and configure your settings:
```env
# Embedding Model (Runs locally via PyTorch)
EMBED_MODEL=openai/clip-vit-base-patch32
EMBED_DEVICE=auto

# Vector DB
CHROMA_DIR=./storage/chroma
CHROMA_COLLECTION=mm_rag

# Retrieval Parameters
TOP_K_IMAGES=4
TOP_K_TEXTS=4
FUSION_ALPHA=0.7

# Generation (Ollama Cloud or Local)
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:120b
```

---

## 📖 Running the System

### Step 1: Prepare the Dataset
Downloads images from the Oxford-IIIT Pet dataset and fetches corresponding Wikipedia articles:
```bash
python scripts/prepare_data.py
```
*Outputs: `data/images/`, `data/texts/`, `data/manifest.jsonl`, and `data/test_queries.jsonl` (10% held-out test split).*

### Step 2: Ingest Data into ChromaDB
Chunks text documents, generates L2-normalized CLIP embeddings, and indexes them into ChromaDB:
```bash
python cli.py ingest
```

### Step 3: Launch the Web Application
Start the modern interactive Web UI:
```bash
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser:
* **Quick Test Samples:** Click any sample thumbnail (Persian, Beagle, Bengal, Shiba Inu, etc.) to instantly load and test.
* **Upload Image:** Drag & drop any pet photo.
* **Ask Questions:** Query for breed traits, history, or grooming requirements.
* **Adjust Fusion:** Tune $\alpha$ (Image vs. Text weight) or toggle to Reciprocal Rank Fusion (RRF).

*(Optional: Run with `--gradio` to launch the Gradio interface instead: `python app.py --gradio`)*

---

## 💻 CLI Usage

You can also query or inspect the system directly from the terminal:

* **Check Vector DB Status:**
  ```bash
  python cli.py status
  ```

* **Image-Only Query:**
  ```bash
  python cli.py ask --image data/images/persian/0.jpg
  ```

* **Multimodal Query (Image + Question):**
  ```bash
  python cli.py ask --image data/images/beagle/0.jpg --text "What is the origin and temperament of this breed?" --fusion mean --alpha 0.7
  ```

* **Text-Only Query:**
  ```bash
  python cli.py ask --text "Which pet breed has a wild leopard-like spotted coat?"
  ```

---

## 📊 Evaluation & Tests

### Run Retrieval Benchmark
Evaluates Image $\rightarrow$ Image and Image $\rightarrow$ Text retrieval metrics (Recall@K and Precision@K) across fusion strategies:
```bash
python eval/evaluate.py
```
*Outputs a formatted summary table in the terminal and saves detailed metrics to `eval/results.json`.*

### Run Unit Tests
Verifies token chunking ($\le 77$ tokens), unit normalization ($L_2 = 1.0$), Chroma roundtrips, and retrieval fusion:
```bash
pytest tests/
```
