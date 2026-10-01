"""CLIP-token-aware sentence chunker for multimodal RAG."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer, CLIPTokenizerFast


@dataclass
class TextChunk:
    text: str
    token_count: int
    chunk_index: int


class CLIPChunker:
    """
    Chunks text documents into segments that strictly fit within CLIP's 77 token limit.
    Splits on sentence boundaries first, then packs sentences up to `max_chunk_tokens`
    with token overlap.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_chunk_tokens: int = 60,
        overlap_tokens: int = 10,
        tokenizer: Optional[CLIPTokenizerFast] = None,
    ):
        self.max_chunk_tokens = min(max_chunk_tokens, 75)  # Leave room for BOS/EOS tokens (limit is 77)
        self.overlap_tokens = overlap_tokens

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        """Count CLIP tokens including special tokens (BOS/EOS)."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return len(tokens)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly."""
        text = text.strip()
        if not text:
            return []
        
        # Split on sentence boundaries (., !, ?, \n)
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a single sentence that exceeds max_chunk_tokens into smaller word pieces.
        """
        words = sentence.split()
        pieces = []
        current_piece = []

        for word in words:
            candidate = " ".join(current_piece + [word])
            if self.count_tokens(candidate) <= self.max_chunk_tokens:
                current_piece.append(word)
            else:
                if current_piece:
                    pieces.append(" ".join(current_piece))
                current_piece = [word]

        if current_piece:
            pieces.append(" ".join(current_piece))

        return pieces

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk a given text string into token-bounded chunks with overlap.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Ensure no single sentence exceeds max_chunk_tokens
        normalized_sentences: List[str] = []
        for s in sentences:
            if self.count_tokens(s) > self.max_chunk_tokens:
                normalized_sentences.extend(self._split_long_sentence(s))
            else:
                normalized_sentences.append(s)

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        chunk_idx = 0

        i = 0
        while i < len(normalized_sentences):
            sentence = normalized_sentences[i]
            candidate_sentences = current_sentences + [sentence]
            candidate_text = " ".join(candidate_sentences)
            token_count = self.count_tokens(candidate_text)

            if token_count <= self.max_chunk_tokens:
                current_sentences.append(sentence)
                i += 1
            else:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    c_tokens = self.count_tokens(chunk_text)
                    chunks.append(TextChunk(text=chunk_text, token_count=c_tokens, chunk_index=chunk_idx))
                    chunk_idx += 1

                    # Compute overlap by keeping suffix sentences
                    overlap_sentences: List[str] = []
                    for s in reversed(current_sentences):
                        test_text = " ".join([s] + overlap_sentences)
                        # Without BOS/EOS overhead for pure overlap content calculation
                        if len(self.tokenizer.encode(test_text, add_special_tokens=False)) <= self.overlap_tokens:
                            overlap_sentences.insert(0, s)
                        else:
                            break

                    # Avoid infinite loops if overlap equals full sentence list
                    if len(overlap_sentences) >= len(current_sentences):
                        overlap_sentences = current_sentences[-1:] if len(current_sentences) > 1 else []

                    current_sentences = overlap_sentences
                else:
                    # Single sentence larger than budget (edge case), add directly
                    c_tokens = self.count_tokens(sentence)
                    chunks.append(TextChunk(text=sentence, token_count=c_tokens, chunk_index=chunk_idx))
                    chunk_idx += 1
                    i += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            c_tokens = self.count_tokens(chunk_text)
            chunks.append(TextChunk(text=chunk_text, token_count=c_tokens, chunk_index=chunk_idx))

        return chunks
