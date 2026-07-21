"""
kg/graph/traversal.py

Path finding over the projected NX graph. Pure graph operations — takes node
ids that are already resolved, returns PathResult.

Deliberately knows nothing about resolution: callers pass exact node ids, and
resolution is EntityResolver's job. Keeping these apart is what lets you answer
"is this broken because the entity is missing, or because the path is?" — two
questions that were tangled when both lived on one class.
"""

import logging
from typing import Optional

import networkx as nx # type: ignore

from kg.reasoning_models import HopResult, PathResult
from kg.text import normalize

logger = logging.getLogger(__name__)


class Traversal:
    """
    Args:
        G:          Projected NX graph from projection.build_nx_graph.
        hypergraph: Needed to map a projected binary edge back to the
                    hyperedge that justifies it.
    """

    def __init__(self, G, hypergraph):
        self.G          = G
        self.hypergraph = hypergraph

    def path_between(self, src: str, dst: str) -> list[str]:
        """Shortest entity path src -> dst. [] if either end is absent or
        no path exists — the caller cannot distinguish those two cases from
        the return value alone, which is intentional: both mean 'no route'."""
        if src not in self.G or dst not in self.G:
            return []
        try:
            return nx.shortest_path(self.G, src, dst)
        except nx.NetworkXNoPath:
            return []

    def edges_on_path(self, src: str, dst: str) -> PathResult:
        """
        Shortest path with a full HopResult chain, one per hop.

        A hop is marked broken when no hyperedge can be recovered for that
        pair — which should not normally happen, since the projected edge only
        exists because some hyperedge created it. It fires if the hypergraph
        and the projection have drifted out of sync.
        """
        src  = normalize(src)
        dst  = normalize(dst)
        path = self.path_between(src, dst)

        if not path:
            return PathResult(src=src, dst=dst, hops=[], found=False)

        hops = []
        for i in range(len(path) - 1):
            hop_src = path[i]
            hop_dst = path[i + 1]
            edge    = self.best_edge_for_hop(hop_src, hop_dst)
            hops.append(HopResult(
                src=hop_src, dst=hop_dst,
                edge=edge, is_broken=(edge is None),
            ))
        return PathResult(src=src, dst=dst, hops=hops, found=True)

    def neighbors(self, entity: str) -> set[str]:
        """All entities one hop away. Empty set if the entity isn't a node."""
        entity = normalize(entity)
        if entity not in self.G:
            return set()
        return set(self.G.neighbors(entity))

    def best_edge_for_hop(self, src: str, dst: str) -> Optional[object]:
        """
        Best hyperedge covering a binary hop — prefer gold, then highest arity.

        Public (was `_best_edge_for_hop`): useful directly when inspecting why
        a particular hop carries the relation label it does.
        """
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