"""
Build the demo corpus:
  - IMAGES: Oxford-IIIT Pets subset (pet photos, label = breed)
  - TEXTS : Wikipedia article text about each breed (NOT captions of the images)
  - TEST SPLIT: 10% held-out images saved to test_queries.jsonl for evaluation

Output:
  data/images/<breed>/<id>.jpg
  data/texts/<breed>.txt
  data/manifest.jsonl     # Indexed documents: id, modality, path, label, source
  data/test_queries.jsonl  # Held-out images for eval: id, modality, path, label, source

Run:
  python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
N_PER_BREED = int(os.getenv("IMAGES_PER_BREED", "40"))
HF_DATASET = os.getenv("HF_DATASET", "timm/oxford-iiit-pet")
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "mm-rag-demo/0.1 (educational project; contact@example.com)"}

# Manual overrides for breed names to find Wikipedia articles reliably
BREED_SEARCH_OVERRIDES: Dict[str, str] = {
    "abyssinian": "Abyssinian cat",
    "american_bulldog": "American Bulldog",
    "american_pit_bull_terrier": "American Pit Bull Terrier",
    "basset_hound": "Basset Hound",
    "beagle": "Beagle",
    "bengal": "Bengal cat",
    "birman": "Birman",
    "bombay": "Bombay cat",
    "boxer": "Boxer dog",
    "british_shorthair": "British Shorthair",
    "bullmastiff": "Bullmastiff",
    "cairn_terrier": "Cairn Terrier",
    "chihuahua": "Chihuahua dog",
    "egyptian_mau": "Egyptian Mau",
    "english_cocker_spaniel": "English Cocker Spaniel",
    "english_setter": "English Setter",
    "german_shorthaired": "German Shorthaired Pointer",
    "great_pyrenees": "Great Pyrenees",
    "havanese": "Havanese dog",
    "japanese_chin": "Japanese Chin",
    "keeshond": "Keeshond",
    "leonberger": "Leonberger",
    "maine_coon": "Maine Coon",
    "miniature_pinscher": "Miniature Pinscher",
    "newfoundland": "Newfoundland dog",
    "persian": "Persian cat",
    "pomeranian": "Pomeranian dog",
    "pug": "Pug",
    "ragdoll": "Ragdoll",
    "russian_blue": "Russian Blue",
    "saint_bernard": "Saint Bernard dog",
    "samoyed": "Samoyed dog",
    "scottish_terrier": "Scottish Terrier",
    "shiba_inu": "Shiba Inu",
    "siamese": "Siamese cat",
    "sphynx": "Sphynx cat",
    "staffordshire_bull_terrier": "Staffordshire Bull Terrier",
    "wheaten_terrier": "Soft-coated Wheaten Terrier",
    "yorkshire_terrier": "Yorkshire Terrier",
}


def slug(s: str) -> str:
    """Normalize a string to a clean snake_case slug."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def wiki_text(breed_slug: str, original_name: str) -> Optional[Tuple[str, str]]:
    """Search Wikipedia for the breed, returning (title, plain_text_extract)."""
    search_term = BREED_SEARCH_OVERRIDES.get(
        breed_slug, f"{original_name.replace('_', ' ')} breed"
    )

    q = {
        "action": "query",
        "list": "search",
        "srsearch": search_term,
        "srlimit": 3,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=q, headers=HEADERS, timeout=30).json()
        hits = r.get("query", {}).get("search", [])
        if not hits:
            return None

        title = hits[0]["title"]
        q_page = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        r_page = requests.get(WIKI_API, params=q_page, headers=HEADERS, timeout=30).json()
        pages = r_page.get("query", {}).get("pages", {})
        if not pages:
            return None

        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        return title, extract
    except Exception as e:
        print(f"[error] Failed to fetch Wikipedia text for {breed_slug}: {e}")
        return None


def main():
    random.seed(42)
    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "texts").mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    test_queries: List[dict] = []

    print(f"Loading dataset {HF_DATASET}...")
    ds = None
    candidate_datasets = [HF_DATASET, "timm/oxford-iiit-pet", "pcuenq/oxford-pets", "nelorth/oxford-pets"]
    # remove duplicates preserving order
    seen = set()
    candidate_datasets = [c for c in candidate_datasets if not (c in seen or seen.add(c))]

    for d_name in candidate_datasets:
        try:
            print(f"Attempting to load '{d_name}'...")
            ds = load_dataset(d_name, split="train")
            print(f"Successfully loaded dataset from '{d_name}'.")
            break
        except Exception as e:
            print(f"Could not load '{d_name}': {e}")

    if ds is None:
        print("[Fatal] Could not load Oxford Pet dataset from any candidate source.")
        return

    # Extract label names
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        label_names = ds.features["label"].names
    else:
        # Fallback if label is string or custom
        label_names = sorted(list(set(row.get("label", "unknown") for row in ds)))

    counts: Dict[str, int] = {}
    breed_images: Dict[str, List[dict]] = {}

    print(f"Collecting up to {N_PER_BREED} images per breed...")
    for i, row in enumerate(tqdm(ds, desc="Processing images")):
        label_val = row["label"]
        raw_breed = label_names[label_val] if isinstance(label_val, int) and label_val < len(label_names) else str(label_val)
        breed = slug(raw_breed)

        if counts.get(breed, 0) >= N_PER_BREED:
            continue
        counts[breed] = counts.get(breed, 0) + 1

        out = DATA_DIR / "images" / breed / f"{i}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(out, quality=90)

        item = {
            "id": f"img_{i}",
            "modality": "image",
            "path": str(out).replace("\\", "/"),
            "label": breed,
            "source": HF_DATASET,
        }
        breed_images.setdefault(breed, []).append(item)

    # Split 10% of images for held-out evaluation queries
    for breed, items in breed_images.items():
        n_test = max(1, int(len(items) * 0.10))
        # Deterministic shuffle for split
        shuffled = list(items)
        random.shuffle(shuffled)
        test_items = shuffled[:n_test]
        train_items = shuffled[n_test:]

        test_queries.extend(test_items)
        manifest.extend(train_items)

    print(f"\nFetching Wikipedia articles for {len(counts)} breeds...")
    missing_wiki = []
    for breed in tqdm(sorted(counts.keys()), desc="Wikipedia articles"):
        res = wiki_text(breed, breed)
        if not res or len(res[1]) < 300:
            missing_wiki.append(breed)
            print(f"[warn] No usable Wikipedia text for {breed}")
            continue

        title, text = res
        out = DATA_DIR / "texts" / f"{breed}.txt"
        out.write_text(text, encoding="utf-8")
        manifest.append({
            "id": f"txt_{breed}",
            "modality": "text",
            "path": str(out).replace("\\", "/"),
            "label": breed,
            "source": f"wikipedia:{title}",
        })

    if missing_wiki:
        print(f"\n[summary] Missing Wikipedia articles for {len(missing_wiki)} breeds: {missing_wiki}")

    # Write manifest.jsonl (indexed documents)
    manifest_path = DATA_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    # Write test_queries.jsonl (held-out test queries)
    test_queries_path = DATA_DIR / "test_queries.jsonl"
    with open(test_queries_path, "w", encoding="utf-8") as f:
        for t in test_queries:
            f.write(json.dumps(t) + "\n")

    n_indexed_images = sum(1 for m in manifest if m["modality"] == "image")
    n_indexed_texts = sum(1 for m in manifest if m["modality"] == "text")
    print(f"\nDone! Dataset prepared in '{DATA_DIR}':")
    print(f"  - Indexed images: {n_indexed_images}")
    print(f"  - Indexed texts:  {n_indexed_texts}")
    print(f"  - Held-out test queries: {len(test_queries)}")
    print(f"  - Manifest written to: {manifest_path}")
    print(f"  - Test queries written to: {test_queries_path}")


if __name__ == "__main__":
    main()
