"""
kg/embeddings/backend.py

Local embedding backend. See config.py's DEFAULT_MODEL note for why this is
decoupled from the vLLM extraction server rather than served alongside it.
"""

import logging
import threading
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """
    Thin wrapper over sentence-transformers.

    Model is a PROCESS-WIDE singleton (class-level, not instance-level) —
    graph_store.py builds one PathFinder/EntityResolver per sample, and
    without this every one of 200 samples would reload a ~130MB checkpoint
    from disk. Reload only happens if a different model_name is requested
    than whatever is currently loaded.
    """

    _lock = threading.Lock()
    _model = None
    _model_name: Optional[str] = None

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        with EmbeddingBackend._lock:
            if (
                EmbeddingBackend._model is not None
                and EmbeddingBackend._model_name == self.model_name
            ):
                return
            # deferred import: sentence-transformers pulls in torch, no need
            # to pay that cost for anyone who imports kg.embeddings but never
            # actually uses the semantic tier (use_semantic=False).
            from sentence_transformers import SentenceTransformer # type: ignore

            logger.info(f"Loading embedding model {self.model_name} (once per process)...")
            EmbeddingBackend._model = SentenceTransformer(self.model_name)
            EmbeddingBackend._model_name = self.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        L2-normalized embeddings, shape (len(texts), dim).
        Empty input -> empty (0, 0) array, so callers can skip the search
        without a separate length check.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_loaded()
        vectors = EmbeddingBackend._model.encode(
            texts,
            normalize_embeddings=True,   # so cosine similarity == plain dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)