"""
kg/path_finder.py

Graph traversal over a projected NetworkX graph.
Owns: NX graph construction, path finding, entity-title resolution, stats.

Does NOT know about HotpotQA samples or broken hop logic — pure graph ops.
"""

import logging
from typing import Optional

import networkx as nx

from .hypergraph_builder import HyperEdge, KnowledgeHypergraph
from .reasoning_models import HopResult, PathResult

logger = logging.getLogger(__name__)


class PathFinder:
    """
    Projects a KnowledgeHypergraph into NetworkX and exposes
    graph traversal primitives.

    Args:
        hypergraph: KnowledgeHypergraph from hypergraph_builder.py
        directed:   Use DiGraph if True, Graph if False (default).
    """

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        directed:   bool = False,
    ):
        self.hypergraph = hypergraph
        self.directed   = directed
        self._chunk_title_index: dict[str, str] = {}
        self.G = self._build_nx_graph()
        logger.info(
            f"PathFinder ready — "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} projected edges "
            f"(from {hypergraph.num_edges()} hyperedges)"
        )

    def _build_nx_graph(self) -> nx.Graph:
        """
        Project hyperedges into binary NX graph via clique expansion.
        [e1, e2, e3] → (e1↔e2), (e1↔e3), (e2↔e3).
        Multiple hyperedges on the same pair accumulate in edge_ids list.
        """
        G = nx.DiGraph() if self.directed else nx.Graph()

        for edge in self.hypergraph.edges.values():
            entities = edge.entities

            for eid in entities:
                if eid not in G:
                    node = self.hypergraph.nodes.get(eid)
                    G.add_node(eid, label=node.label if node else eid)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    src, dst = entities[i], entities[j]
                    if G.has_edge(src, dst):
                        G[src][dst]["edge_ids"].append(edge.edge_id)
                    else:
                        G.add_edge(
                            src, dst,
                            edge_ids=[edge.edge_id],
                            relation=edge.relation,
                            is_gold=edge.is_gold,
                        )
        return G

    def path_between(self, src: str, dst: str) -> list[str]:
        """Shortest entity path src → dst. Returns [] if unreachable."""
        if src not in self.G or dst not in self.G:
            return []
        try:
            return nx.shortest_path(self.G, src, dst)
        except nx.NetworkXNoPath:
            return []

    def edges_on_path(self, src: str, dst: str) -> PathResult:
        """Shortest path with full HopResult chain per hop."""
        src  = self._normalize(src)
        dst  = self._normalize(dst)
        path = self.path_between(src, dst)

        if not path:
            return PathResult(src=src, dst=dst, hops=[], found=False)

        hops = []
        for i in range(len(path) - 1):
            hop_src = path[i]
            hop_dst = path[i + 1]
            edge    = self._best_edge_for_hop(hop_src, hop_dst)
            hops.append(HopResult(
                src=hop_src, dst=hop_dst,
                edge=edge, is_broken=(edge is None),
            ))
        return PathResult(src=src, dst=dst, hops=hops, found=True)

    def neighbors(self, entity: str) -> set[str]:
        """All entities one hop away from entity."""
        entity = self._normalize(entity)
        if entity not in self.G:
            return set()
        return set(self.G.neighbors(entity))

    def index_samples(self, samples) -> None:
        """
        Build chunk_id → article_title index from loaded samples.
        Enables tier-1 (exact chunk-title) matching in _entities_for_title.
        Call after PathFinder init for best resolution accuracy.
        """
        self._chunk_title_index = {
            chunk.chunk_id: chunk.title
            for sample in samples
            for chunk in sample.chunks
        }
        logger.info(f"Indexed {len(self._chunk_title_index)} chunk-title mappings")

    def entities_for_title(self, title: str) -> set[str]:
        """
        Return all graph entity ids associated with a Wikipedia article title.

        Tier 1 — chunk title index (exact article match via chunk metadata)
        Tier 2 — entity label equals normalized title exactly
                 e.g. entity "shirley temple" ↔ title "Shirley Temple"
        Tier 3 — substring match
                 e.g. entity "jonathan stark" ↔ title "Jonathan Stark (tennis)"
        """
        title_norm = self._normalize(title)
        result: set[str] = set()

        for node in self.hypergraph.nodes.values():
            # tier 2 — exact label match
            if node.entity_id == title_norm:
                result.add(node.entity_id)
                continue

            # tier 3 — substring match
            if title_norm in node.entity_id or node.entity_id in title_norm:
                result.add(node.entity_id)
                continue

            # tier 1 — chunk title metadata
            for chunk_id in node.chunks:
                chunk_title = self._chunk_title_index.get(chunk_id, "")
                if self._normalize(chunk_title) == title_norm:
                    result.add(node.entity_id)
                    break

        return result

    def stats(self) -> dict:
        degrees = [d for _, d in self.G.degree()]
        return {
            "num_nodes":      self.G.number_of_nodes(),
            "num_edges":      self.G.number_of_edges(),
            "num_hyperedges": self.hypergraph.num_edges(),
            "avg_degree":     round(sum(degrees) / max(len(degrees), 1), 2),
            "num_components": nx.number_connected_components(self.G)
                              if not self.directed else "N/A (directed)",
            "density":        round(nx.density(self.G), 4),
        }

    def _best_edge_for_hop(self, src: str, dst: str) -> Optional[HyperEdge]:
        """Best hyperedge covering a binary hop — prefer gold, then highest arity."""
        if not self.G.has_edge(src, dst):
            return None
        edge_ids = self.G[src][dst]["edge_ids"]
        edges = [
            self.hypergraph.edges[eid]
            for eid in edge_ids if eid in self.hypergraph.edges
        ]
        if not edges:
            return None
        edges.sort(key=lambda e: (e.is_gold, len(e.entities)), reverse=True)
        return edges[0]

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()