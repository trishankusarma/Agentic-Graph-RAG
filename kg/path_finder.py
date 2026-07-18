"""
kg/path_finder.py

Graph traversal over a projected NetworkX graph.
Owns: NX graph construction, path finding, entity resolution, stats.

Does NOT know about HotpotQA samples or broken hop logic — pure graph ops.
"""

import logging
from typing import Optional

import networkx as nx

from .hypergraph_builder import HyperEdge, KnowledgeHypergraph
from .reasoning_models import HopResult, PathResult

logger = logging.getLogger(__name__)

# Entities shorter than this are too generic to trust as a fuzzy match
# ("ed", "us", "the"). Raise it if junk pairings persist.
MIN_FUZZY_ENTITY_LEN = 4


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

        self._chunk_title_index:  dict[str, str]      = {}
        self._title_entity_index: dict[str, set[str]] = {}
        self._title_cache:        dict[str, set[str]] = {}
        self._entity_cache:       dict[str, set[str]] = {}

        self.G = self._build_nx_graph()
        logger.info(
            f"PathFinder ready — "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} projected edges "
            f"(from {hypergraph.num_edges()} hyperedges)"
        )

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_nx_graph(self) -> nx.Graph:
        """
        Project hyperedges into a binary NX graph via clique expansion.
        [e1, e2, e3] → (e1↔e2), (e1↔e3), (e2↔e3).
        Multiple hyperedges on the same pair accumulate in edge_ids.

        Entities are de-duplicated per hyperedge first. HypergraphBuilder
        rejects facts with fewer than 2 DISTINCT normalized entities but stores
        the un-deduped list, so an edge like [a, b, a] survives and would add a
        self-loop on `a`. Those inflate edge counts (one sample reported
        density 1.07, impossible for a simple graph) and let shortest_path
        traverse an entity to itself.

        is_gold on a projected edge is the OR across all hyperedges covering
        that pair — keeping only the first writer's value let a gold hyperedge
        be masked by an earlier non-gold one.
        """
        G = nx.DiGraph() if self.directed else nx.Graph()

        for edge in self.hypergraph.edges.values():
            # dict.fromkeys preserves first-seen order while de-duplicating
            entities = list(dict.fromkeys(edge.entities))

            for eid in entities:
                if eid not in G:
                    node = self.hypergraph.nodes.get(eid)
                    G.add_node(eid, label=node.label if node else eid)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    src, dst = entities[i], entities[j]
                    if src == dst:
                        continue          # belt and braces
                    if G.has_edge(src, dst):
                        data = G[src][dst]
                        data["edge_ids"].append(edge.edge_id)
                        data["is_gold"] = data["is_gold"] or edge.is_gold
                    else:
                        G.add_edge(
                            src, dst,
                            edge_ids=[edge.edge_id],
                            is_gold=edge.is_gold,
                        )
        return G

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Entity resolution
    # ------------------------------------------------------------------ #

    def resolve_entity(self, label: str) -> set[str]:
        """
        Resolve a free-text label (typically a dataset answer) to graph nodes.

        Used to decide whether an answer is even IN the graph before asking
        whether it is reachable. An answer that never resolves is not a broken
        hop — no edge insertion can repair it. See TerminalStatus.NOT_AN_ENTITY.

        Tier 1 — exact normalized match
        Tier 2 — token-subset either direction, both sides long enough
                 e.g. "Robert Erskine Childers DSC" ↔ "robert erskine childers"

        Deliberately stricter than plain substring: `"no" in "nolan"` would
        otherwise resolve a predicate answer to a real node and turn a
        PREDICATE into a spurious ENTITY_UNREACHED.
        """
        norm = self._normalize(label)
        if not norm:
            return set()

        cached = self._entity_cache.get(norm)
        if cached is not None:
            return cached

        result = self._resolve_entity_uncached(norm)
        self._entity_cache[norm] = result
        return result

    def _resolve_entity_uncached(self, norm: str) -> set[str]:
        # tier 1 — exact
        if norm in self.hypergraph.nodes:
            return {norm}

        # tier 2 — token subset, guarded by length
        if len(norm) < MIN_FUZZY_ENTITY_LEN:
            return set()

        tokens = set(norm.split())
        if not tokens:
            return set()

        matches = set()
        for node in self.hypergraph.nodes.values():
            if len(node.entity_id) < MIN_FUZZY_ENTITY_LEN:
                continue
            node_tokens = set(node.entity_id.split())
            if node_tokens <= tokens or tokens <= node_tokens:
                matches.add(node.entity_id)
        return matches

    # ------------------------------------------------------------------ #
    # Entity ↔ title resolution
    # ------------------------------------------------------------------ #

    def index_samples(self, samples) -> None:
        """
        Build chunk_id → article_title and title → entity_ids indices.

        Enables tier-1 matching in entities_for_title(). Call right after init;
        tier 1 is the only tier using real article provenance rather than
        string similarity.
        """
        self._chunk_title_index = {
            chunk.chunk_id: chunk.title
            for sample in samples
            for chunk in sample.chunks
        }

        title_index: dict[str, set[str]] = {}
        for node in self.hypergraph.nodes.values():
            for chunk_id in node.chunks:
                title = self._chunk_title_index.get(chunk_id)
                if title:
                    title_index.setdefault(
                        self._normalize(title), set()
                    ).add(node.entity_id)
        self._title_entity_index = title_index

        # Tier 1 results change with the index — memoized values are stale.
        self._title_cache.clear()

        logger.info(
            f"Indexed {len(self._chunk_title_index)} chunk-title mappings, "
            f"{len(self._title_entity_index)} titles with entities"
        )

    def entities_for_title(self, title: str) -> set[str]:
        """
        Return graph entity ids associated with a Wikipedia article title.

        Tiers run in order and the FIRST non-empty one wins — they are not
        unioned, since each is looser than the last and the loosest would
        dominate.

        Tier 1 — chunk title metadata: entities actually extracted from that
                 article's text. Real provenance. Requires index_samples().
        Tier 2 — entity id equals the normalized title exactly.
        Tier 3 — entity's tokens are a subset of the title's tokens, entity at
                 least MIN_FUZZY_ENTITY_LEN chars.

        Tier 3 is narrower than plain containment on purpose: naive substring
        matched "ed" against "Ed Wood" and returned dozens of entities, and
        _best_path() then runs |from| x |to| shortest-path searches over that.
        """
        title_norm = self._normalize(title)

        cached = self._title_cache.get(title_norm)
        if cached is not None:
            return cached

        result = self._resolve_title(title_norm)
        self._title_cache[title_norm] = result
        return result

    def _resolve_title(self, title_norm: str) -> set[str]:
        # tier 1 — chunk title metadata
        hit = self._title_entity_index.get(title_norm)
        if hit:
            return set(hit)

        # tier 2 — exact entity id match
        if title_norm in self.hypergraph.nodes:
            return {title_norm}

        # tier 3 — token-subset containment, long entities only
        title_tokens = set(title_norm.split())
        if not title_tokens:
            return set()
        return {
            n.entity_id
            for n in self.hypergraph.nodes.values()
            if len(n.entity_id) >= MIN_FUZZY_ENTITY_LEN
            and set(n.entity_id.split()) <= title_tokens
        }

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """
        Graph statistics.

        max_arity and self_loops are diagnostics: an arity above ~6 means a
        sentence was dumped into one fact, and any self-loop means entity
        de-duplication regressed upstream. Both silently inflate connectivity.
        """
        degrees   = [d for _, d in self.G.degree()]
        arities   = [len(set(e.entities)) for e in self.hypergraph.edges.values()]
        return {
            "num_nodes":      self.G.number_of_nodes(),
            "num_edges":      self.G.number_of_edges(),
            "num_hyperedges": self.hypergraph.num_edges(),
            "avg_degree":     round(sum(degrees) / max(len(degrees), 1), 2),
            "max_arity":      max(arities) if arities else 0,
            "self_loops":     nx.number_of_selfloops(self.G),
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