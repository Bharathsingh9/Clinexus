"""Biomedical Embedding Service.

Specification Section 2, Section 4.2, Appendix A6 (Req ID: RET-2):
- Wraps SentenceTransformer for biomedical embeddings.
- Default resolved model: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb.
- Empirical properties verified in DEC-020:
  * max_seq_length: 100 tokens
  * output dimension: 768
  * pooling: mean pooling over BertModel token embeddings
"""

import logging
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service providing dense biomedical embeddings and token operations."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name: str = model_name or settings.EMBEDDING_MODEL
        self.device: str = device or ("cuda" if False else "cpu")
        self._model: Optional[SentenceTransformer] = None
        self._dimension: int = 768
        self._max_seq_length: int = 100

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(self._model, "get_sentence_embedding_dimension", None)
            if callable(dim_fn):
                self._dimension = dim_fn()
            else:
                self._dimension = 768
            self._max_seq_length = int(getattr(self._model, "max_seq_length", 100))
        return self._model

    @property
    def dimension(self) -> int:
        _ = self.model  # Ensure initialized
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        _ = self.model  # Ensure initialized
        return self._max_seq_length

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Fallback approximation: 1 token ~= 4 chars if tokenizer missing
        return max(1, len(text.split()))

    def get_effective_chunk_params(self) -> Tuple[int, int]:
        """Compute effective chunk size and overlap adhering to:
        - effective_chunk_size = min(settings.CHUNK_TOKENS, model_max_tokens - special_tokens)
        - overlap < effective_chunk_size / 2 (default proposed: 20 tokens)
        """
        tokenizer = getattr(self.model, "tokenizer", None)
        special_tokens = 2
        if tokenizer is not None and hasattr(tokenizer, "num_special_tokens_to_add"):
            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        content_limit = max(10, self.max_seq_length - special_tokens)
        effective_chunk = min(settings.CHUNK_TOKENS, content_limit)
        # Overlap must be strictly smaller than half the chunk
        max_allowed_overlap = max(1, (effective_chunk // 2) - 1)
        # Default proposed overlap is 20 tokens (or max_allowed_overlap if smaller)
        overlap = min(20, max_allowed_overlap)
        return effective_chunk, overlap

    def chunk_text_if_needed(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> List[str]:
        """Apply chunker ONLY when text exceeds effective token limit per SPEC 9.3."""
        effective_chunk, default_overlap = self.get_effective_chunk_params()
        limit = max_tokens if max_tokens is not None else effective_chunk
        overlap = overlap_tokens if overlap_tokens is not None else default_overlap
        # Guarantee overlap < limit / 2
        overlap = min(overlap, max(0, (limit // 2) - 1))

        token_count = self.count_tokens(text)
        if token_count <= limit:
            return [text]

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            # Fallback word-level chunking
            words = text.split()
            chunks: List[str] = []
            step = max(1, limit - overlap)
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + limit])
                if chunk:
                    chunks.append(chunk)
                if i + limit >= len(words):
                    break
            return chunks if chunks else [text]

        # Token-based chunking with exact offsets
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        step = max(1, limit - overlap)
        for i in range(0, len(token_ids), step):
            sub_ids = token_ids[i : i + limit]
            decoded = tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
            if decoded:
                chunks.append(decoded)
            if i + limit >= len(token_ids):
                break
        return chunks if chunks else [text]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        emb = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        b_size = batch_size or settings.EMBED_BATCH_SIZE
        embeddings = self.model.encode(
            texts,
            batch_size=b_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
