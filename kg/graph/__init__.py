"""
kg/graph

    path_finder.py  PathFinder — facade, unchanged public API
    projection.py   build_nx_graph() — hypergraph -> NX, shape rules
    resolver.py     EntityResolver — free text -> node ids, plus explain()
    traversal.py    Traversal — path finding over resolved node ids
    stats.py        graph_stats() — health metrics and what bad values mean

Import PathFinder from here; reach into the components only when debugging.
"""

from .path_finder import PathFinder
from .projection import build_nx_graph
from .resolver import MIN_FUZZY_ENTITY_LEN, EntityResolver
from .stats import graph_stats
from .traversal import Traversal

__all__ = [
    "PathFinder",
    "EntityResolver",
    "Traversal",
    "build_nx_graph",
    "graph_stats",
    "MIN_FUZZY_ENTITY_LEN",
]