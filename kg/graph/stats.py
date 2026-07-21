"""
kg/graph/stats.py

Graph health metrics. A free function, not a method — it reads the graph and
returns numbers, holds no state, and is the thing you call when a sample's
results look wrong and you want to know whether the GRAPH is wrong.
"""

import networkx as nx


def graph_stats(G, hypergraph, directed: bool = False) -> dict:
    """
    Args:
        G:          Projected NX graph.
        hypergraph: Source hypergraph, for arity and hyperedge count.
        directed:   Suppresses connected-components, undefined for DiGraph.

    Diagnostics worth watching, and what a bad value means:

        self_loops > 0        Entity de-duplication regressed in
                              projection.build_nx_graph. Silently inflates
                              connectivity and lets a path visit an entity
                              twice.

        density > 1.0         Impossible for a simple graph. Always a
                              self-loop bug.

        max_arity at the cap  If the arity histogram is also non-monotonic at
                              the ceiling, facts are being CLAMPED by the
                              schema rather than split by the model. See
                              MAX_FACT_ARITY in kg/extractors/config.py.

        avg_degree very high  A few huge hyperedges dominating via clique
                              expansion. Cross-check max_arity.
    """
    degrees = [d for _, d in G.degree()]
    arities = [len(set(e.entities)) for e in hypergraph.edges.values()]

    return {
        "num_nodes":      G.number_of_nodes(),
        "num_edges":      G.number_of_edges(),
        "num_hyperedges": hypergraph.num_edges(),
        "avg_degree":     round(sum(degrees) / max(len(degrees), 1), 2),
        "max_arity":      max(arities) if arities else 0,
        "self_loops":     nx.number_of_selfloops(G),
        "num_components": nx.number_connected_components(G)
                          if not directed else "N/A (directed)",
        "density":        round(nx.density(G), 4),
    }