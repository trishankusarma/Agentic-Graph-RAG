"""
kg/embeddings/index.py

Brute-force cosine-similarity search over a small set of short strings.

No faiss, on purpose: each PathFinder is scoped to ONE sample's hypergraph —
a few hundred entities at most, not corpus-scale. A dot product over a few
hundred rows is faster than the overhead of building and querying an ANN
index, and it keeps the dependency footprint down. If this ever gets pointed
at a corpus-scale shared index instead of a per-sample one, revisit this.
"""

import numpy as np


class SemanticIndex:
    """
    Args:
        ids:     Identifier per row (e.g. normalized entity ids).
        vectors: L2-normalized embeddings, shape (len(ids), dim).
    """

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self.vectors = vectors

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, float]]:
        """Top-k ids with cosine similarity >= min_similarity, best first."""
        if len(self.ids) == 0 or query_vector.size == 0:
            return []
        sims = self.vectors @ query_vector          # both normalized -> dot == cosine
        top_idx = np.argsort(-sims)[:top_k]
        return [
            (self.ids[i], float(sims[i]))
            for i in top_idx
            if sims[i] >= min_similarity
        ]

    @classmethod
    def build(cls, backend, id_text_pairs: list[tuple[str, str]]) -> "SemanticIndex":
        """id_text_pairs: [(id, text_to_embed), ...]. Empty input -> empty index."""
        if not id_text_pairs:
            return cls(ids=[], vectors=np.zeros((0, 0), dtype=np.float32))
        ids = [pair[0] for pair in id_text_pairs]
        texts = [pair[1] for pair in id_text_pairs]
        vectors = backend.encode(texts)
        return cls(ids=ids, vectors=vectors)