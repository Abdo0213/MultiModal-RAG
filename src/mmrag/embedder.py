"""CLIP Embedding model wrapper for images and text."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def normalize_l2(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a 2D numpy array of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _extract_tensor(features: Any) -> torch.Tensor:
    """Extract a 2D tensor whether model returns a raw Tensor or ModelOutput."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0, :]
    return features[0]


class CLIPEmbedder:
    """
    Local CLIP embedding model wrapper supporting text and image embedding
    with automatic hardware device selection and strict L2 normalization.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[CLIPEmbedder] Loading model '{model_name}' on device '{self.device}'...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_image(self, img_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Convert a path or existing PIL image into an RGB PIL Image."""
        if isinstance(img_input, (str, Path)):
            img = Image.open(str(img_input))
        elif isinstance(img_input, Image.Image):
            img = img_input
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")
        return img.convert("RGB")

    def embed_images(
        self, images: Sequence[Union[str, Path, Image.Image]], batch_size: int | None = None
    ) -> np.ndarray:
        """
        Embed a list of images into L2-normalized float32 numpy arrays.
        """
        if not images:
            return np.empty((0, self.model.config.projection_dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(images), bs):
            batch_inputs = images[i : i + bs]
            pil_images = [self._load_image(img) for img in batch_inputs]

            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                raw_features = self.model.get_image_features(**inputs)
                image_features = _extract_tensor(raw_features).float()

            embeddings = image_features.cpu().numpy()
            all_embeddings.append(embeddings)

        stacked = np.vstack(all_embeddings).astype(np.float32)
        return normalize_l2(stacked)

    def embed_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> np.ndarray:
        """
        Embed a list of text strings into L2-normalized float32 numpy arrays.
        Enforces max_length=77 truncation as a safety net.
        """
        if not texts:
            return np.empty((0, self.model.config.projection_dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch_texts = list(texts[i : i + bs])
            inputs = self.processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                raw_features = self.model.get_text_features(**inputs)
                text_features = _extract_tensor(raw_features).float()

            embeddings = text_features.cpu().numpy()
            all_embeddings.append(embeddings)

        stacked = np.vstack(all_embeddings).astype(np.float32)
        return normalize_l2(stacked)
