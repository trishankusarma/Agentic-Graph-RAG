"""
kg/detection/path_policy.py

Audit-specific path selection: shortest CLEAN path over every (from, to) pair
between two entity sets.

This is policy, not graph traversal — PathFinder finds a path between two
named entities; this decides which pair of entities to try and which result
counts as "clean enough" to accept. Shared by ChainBuilder (mid-chain
segments) and TerminalResolver (the answer hop) so the two can't quietly
diverge on what "reachable" means. A prior version had this logic copy-pasted
into both files.
"""

from typing import Optional

from kg.reasoning_models.graph_models import PathResult


def best_path(
    path_finder,
    from_ents: set[str],
    to_ents:   set[str],
) -> Optional[PathResult]:
    """
    Shortest path with no broken hops, tried over every (from, to) pair.

    Cost is |from_ents| x |to_ents| shortest-path searches — which is why
    PathFinder.entities_for_title's tier 3 is deliberately strict. A loose
    title match here turns into thousands of BFS calls for one segment.
    """
    best: Optional[PathResult] = None
    for f in from_ents:
        for t in to_ents:
            if f == t:
                continue
            result = path_finder.edges_on_path(f, t)
            if result.found and not result.has_broken_hops():
                if best is None or result.num_hops() < best.num_hops():
                    best = result
    return best