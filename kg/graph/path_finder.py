"""
kg/graph/path_finder.py

Facade composing projection + resolver + traversal + stats.

Every existing call site (GraphStore, ChainBuilder, TerminalResolver, Pairing)
keeps working with no edits:

    path_between, edges_on_path, neighbors,
    resolve_entity, entities_for_title, index_samples, stats

What changed is that each of those now delegates to a component you can reach
directly when debugging:

    pf.resolver.explain("Marco Da Silva")   # why didn't this resolve?
    pf.traversal.best_edge_for_hop(a, b)    # which hyperedge justifies this?
    pf.G                                    # the raw NX graph
"""

import logging

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
        hypergraph: KnowledgeHypergraph from hypergraph_builder.py
        directed:   Use DiGraph if True, Graph if False (default). Undirected
                    is almost always what you want — a bridge chain traverses
                    relations in whichever direction the extractor happened to
                    write them.
    """

    def __init__(self, hypergraph, directed: bool = False):
        self.hypergraph = hypergraph
        self.directed   = directed

        self.G         = build_nx_graph(hypergraph, directed=directed)
        self.resolver  = EntityResolver(hypergraph)
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