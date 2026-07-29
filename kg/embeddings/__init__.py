"""
kg/embeddings

    backend.py  EmbeddingBackend — local sentence-transformers wrapper
    index.py    SemanticIndex — brute-force cosine similarity search
    config.py   model choice, similarity threshold, match cap
"""

from . import config
from .backend import EmbeddingBackend
from .index import SemanticIndex

__all__ = ["EmbeddingBackend", "SemanticIndex", "config"]