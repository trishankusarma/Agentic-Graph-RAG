"""
kg/graph/path_finder.py

Facade composing projection + resolver + traversal + stats.

Public API is UNCHANGED from before the semantic tier was added, so every
existing call site (GraphStore, ChainBuilder, TerminalResolver, Pairing)
keeps working with no edits:

    path_between, edges_on_path, neighbors,
    resolve_entity, entities_for_title, index_samples, stats

New, additive only: use_semantic / embedding_backend constructor args
(default on), and a semantic_rescues diagnostic property.

    pf.resolver.explain("Marco Da Silva")   # why didn't this resolve?
    pf.traversal.best_edge_for_hop(a, b)    # which hyperedge justifies this?
    pf.semantic_rescues                     # how often the embedding tier fired
    pf.G                                    # the raw NX graph
"""

import logging
from typing import Optional

from kg.embeddings import EmbeddingBackend

from .projection import build_nx_graph
from .resolver import EntityResolver
from .stats import graph_stats
from .traversal import Traversal

logger = logging.getLogger(__name__)


class PathFinder:
    """
    Projects a KnowledgeHypergraph into NetworkX and exposes traversal.

    Does NOT know about HotpotQA samples or broken-hop logic — pure graph ops.
    (index_samples is the one exception, and it only reads chunk_id/title
    metadata, never anything about questions or answers.)

    Args:
        hypergraph:        KnowledgeHypergraph from hypergraph_builder.py
        directed:           Use DiGraph if True, Graph if False (default).
                            Undirected is almost always what you want — a
                            bridge chain traverses relations in whichever
                            direction the extractor happened to write them.
        use_semantic:       Enable the embedding-similarity fallback tier in
                            entity/title resolution. Default on; set False to
                            A/B against lexical-only resolution, or to avoid
                            the sentence-transformers dependency entirely.
        embedding_backend:  Inject a specific EmbeddingBackend (e.g. a
                            different model). Defaults to a lazily-created one.
    """

    def __init__(
        self,
        hypergraph,
        directed:          bool                       = False,
        use_semantic:      bool                        = False,
        embedding_backend: Optional[EmbeddingBackend]  = None,
    ):
        self.hypergraph = hypergraph
        self.directed   = directed

        self.G         = build_nx_graph(hypergraph, directed=directed)
        self.resolver  = EntityResolver(
            hypergraph,
            use_semantic=use_semantic,
            embedding_backend=embedding_backend,
        )
        self.traversal = Traversal(self.G, hypergraph)

        logger.info(
            f"PathFinder ready — "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} projected edges "
            f"(from {hypergraph.num_edges()} hyperedges)"
        )

    # ---- traversal ---------------------------------------------------- #

    def path_between(self, src: str, dst: str) -> list[str]:
        return self.traversal.path_between(src, dst)

    def edges_on_path(self, src: str, dst: str):
        return self.traversal.edges_on_path(src, dst)

    def neighbors(self, entity: str) -> set[str]:
        return self.traversal.neighbors(entity)

    # ---- resolution --------------------------------------------------- #

    def resolve_entity(self, label: str) -> set[str]:
        return self.resolver.resolve_entity(label)

    def entities_for_title(self, title: str) -> set[str]:
        return self.resolver.entities_for_title(title)

    def index_samples(self, samples) -> None:
        self.resolver.index_samples(samples)

    # ---- diagnostics -------------------------------------------------- #

    def stats(self) -> dict:
        return graph_stats(self.G, self.hypergraph, self.directed)

    @property
    def semantic_rescues(self) -> int:
        """How many resolve calls on this graph needed the embedding tier —
        every lexical tier had already missed. Near zero means the tier isn't
        doing anything on this sample; check across a full corpus run before
        judging whether it's worth its cost."""
        return self.resolver.semantic_rescues