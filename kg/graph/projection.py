"""
kg/graph/projection.py

Hypergraph -> NetworkX. One function, called once at PathFinder construction.
"""

import logging

import networkx as nx # type: ignore

logger = logging.getLogger(__name__)


def build_nx_graph(hypergraph, directed: bool = False) -> nx.Graph:
    """
    Project hyperedges into a binary NX graph via clique expansion.
    [e1, e2, e3] -> (e1<->e2), (e1<->e3), (e2<->e3).
    Multiple hyperedges on the same pair accumulate in edge_ids.

    Two rules that are NOT optional:

    1. Entities are de-duplicated per hyperedge first. HypergraphBuilder
       rejects facts with fewer than 2 DISTINCT normalized entities but stores
       the un-deduped list, so an edge like [a, b, a] survives and would add a
       self-loop on `a`. Those inflate edge counts (one sample reported
       density 1.07, impossible for a simple graph) and let shortest_path
       traverse an entity to itself.

    2. is_gold on a projected edge is the OR across all hyperedges covering
       that pair. Keeping only the first writer's value let a gold hyperedge
       be masked by an earlier non-gold one.

    Note the cost: one arity-k hyperedge becomes k(k-1)/2 binary edges. That
    quadratic is why MAX_FACT_ARITY exists in kg/extractors/config.py — an
    arity-32 edge once contributed 496 projected edges by itself.
    """
    G = nx.DiGraph() if directed else nx.Graph()

    for edge in hypergraph.edges.values():
        # dict.fromkeys preserves first-seen order while de-duplicating
        entities = list(dict.fromkeys(edge.entities))

        for eid in entities:
            if eid not in G:
                node = hypergraph.nodes.get(eid)
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