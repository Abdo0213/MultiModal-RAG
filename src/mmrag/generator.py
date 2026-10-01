"""Vision-language generation using Ollama with citation-aware prompting."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import ollama
from PIL import Image

from mmrag.config import settings


def downscale_and_encode_image(
    image_input: Union[str, Path, Image.Image], max_size: int = 768
) -> str:
    """
    Downscale an image to max_size on the longest edge and encode as a base64 JPEG string.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img = img.convert("RGB")
    width, height = img.size

    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OllamaGenerator:
    """
    Ollama vision-capable LLM client with multimodal context prompting and source citations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        raw_host = host or settings.ollama_host
        # Normalize host (strip trailing slashes and optional /v1 for ollama native client)
        if raw_host:
            raw_host = raw_host.strip().rstrip("/")
            if raw_host.endswith("/v1"):
                raw_host = raw_host[:-3].rstrip("/")
        self.host = raw_host

        self.api_key = api_key if api_key is not None else settings.ollama_api_key
        self.model_name = model_name or settings.ollama_model
        self.temperature = (
            temperature if temperature is not None else settings.ollama_temperature
        )

        headers = {}
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"

        self.client = ollama.Client(host=self.host, headers=headers if headers else None)

    def build_prompt(
        self,
        query_text: Optional[str],
        retrieved_texts: List[Dict[str, Any]],
        retrieved_images: List[Dict[str, Any]],
    ) -> str:
        """
        Construct structured system and user context prompting with numbered citations.
        """
        parts = [
            "You are an expert multimodal AI assistant. Answer the user's question using ONLY the provided context and images.",
            "Rules for your response:",
            "1. Answer factually based on the retrieved text passages and retrieved images.",
            "2. Whenever citing information, include exact source tags such as [text:1], [text:2], or [image:1], [image:2].",
            "3. If the context does not contain enough information to answer with confidence, explicitly state what is known and what is missing.",
            "4. Be concise, well-structured, and helpful.\n",
            "--- RETRIEVED TEXT CONTEXT ---",
        ]

        if retrieved_texts:
            for idx, item in enumerate(retrieved_texts, start=1):
                doc = item.get("document", "").strip()
                source = item.get("metadata", {}).get("source", "unknown")
                label = item.get("metadata", {}).get("label", "unknown")
                score = item.get("score", 0.0)
                parts.append(f"[text:{idx}] (Source: {source} | Breed: {label} | Score: {score:.3f}):\n{doc}\n")
        else:
            parts.append("No text passages retrieved.\n")

        parts.append("--- RETRIEVED IMAGE REFERENCES ---")
        if retrieved_images:
            for idx, item in enumerate(retrieved_images, start=1):
                label = item.get("metadata", {}).get("label", "unknown")
                score = item.get("score", 0.0)
                parts.append(f"[image:{idx}] (Breed: {label} | Similarity: {score:.3f})")
        else:
            parts.append("No matching images retrieved.")

        parts.append("\n--- USER QUERY ---")
        if query_text:
            parts.append(f"User Question/Prompt: {query_text}")
        else:
            parts.append("User Query: Please identify the breed, describe its key characteristics, history, and care details based on the provided image and retrieved context.")

        return "\n".join(parts)

    def generate(
        self,
        query_images: Optional[Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]]] = None,
        query_text: Optional[str] = None,
        retrieved_images: Optional[List[Dict[str, Any]]] = None,
        retrieved_texts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send multimodal context (query images + retrieved images + text chunks) to Ollama.
        """
        retrieved_images = retrieved_images or []
        retrieved_texts = retrieved_texts or []

        prompt = self.build_prompt(query_text, retrieved_texts, retrieved_images)

        # Prepare images for VLM input: query images first, then retrieved images
        all_image_b64: List[str] = []

        if query_images:
            if not isinstance(query_images, (list, tuple)):
                query_images = [query_images]
            for img in query_images:
                try:
                    all_image_b64.append(downscale_and_encode_image(img, max_size=768))
                except Exception as e:
                    print(f"[warn] Failed to encode query image: {e}")

        for item in retrieved_images:
            img_path = item.get("metadata", {}).get("path")
            if img_path and Path(img_path).exists():
                try:
                    all_image_b64.append(downscale_and_encode_image(img_path, max_size=768))
                except Exception as e:
                    print(f"[warn] Failed to encode retrieved image {img_path}: {e}")

        # Gather candidate models to try
        models_to_try = [self.model_name]

        # Query available models from Ollama account if possible
        try:
            available_resp = self.client.list()
            available_models = [m.model for m in available_resp.models] if hasattr(available_resp, "models") else []
            if available_models:
                print(f"[Ollama] Available models on account: {available_models}")
                for av_m in available_models:
                    if av_m not in models_to_try:
                        models_to_try.append(av_m)
        except Exception as list_err:
            pass

        for fallback in [
            "llama3.2-vision:latest",
            "llama3.2-vision",
            "llava:latest",
            "llava",
            "qwen2.5-vl:latest",
            "minicpm-v:latest",
            "gpt-oss:120b-cloud",
            "gpt-oss:20b-cloud",
            "llama3.3:70b-cloud",
            "deepseek-v3.1:671b-cloud",
        ]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for m_name in models_to_try:
            try:
                # If model is text-only cloud model and images are provided, handle gracefully
                img_payload = all_image_b64 if all_image_b64 else None
                try:
                    response = self.client.generate(
                        model=m_name,
                        prompt=prompt,
                        images=img_payload,
                        options={"temperature": self.temperature},
                    )
                except Exception as img_err:
                    # If model doesn't support images, fallback to text-prompt only
                    if img_payload and ("image" in str(img_err).lower() or "multimodal" in str(img_err).lower()):
                        response = self.client.generate(
                            model=m_name,
                            prompt=prompt,
                            options={"temperature": self.temperature},
                        )
                    else:
                        raise img_err

                answer = response.get("response", "").strip()
                return {
                    "answer": answer,
                    "prompt": prompt,
                    "model": m_name,
                    "status": "success",
                }
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "retired" in err_str or "not found" in err_str or "404" in err_str or "410" in err_str or "unavailable" in err_str:
                    print(f"[Ollama] Model '{m_name}' unavailable ({e}), trying next candidate...")
                    continue
                else:
                    print(f"[Ollama] Error with '{m_name}': {e}, trying next candidate...")
                    continue

        error_msg = (
            f"[Ollama Error] Could not generate answer using model '{self.model_name}' "
            f"at '{self.host}'. Details: {str(last_error)}.\n\n"
            f"Troubleshooting:\n"
            f"1. For local Ollama: ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull {self.model_name}`).\n"
            f"2. For cloud Ollama: verify your `OLLAMA_API_KEY` and host URL in `.env`."
        )
        print(error_msg)
        return {
            "answer": error_msg,
            "prompt": prompt,
            "model": self.model_name,
            "status": "error",
            "error": str(last_error),
        }
