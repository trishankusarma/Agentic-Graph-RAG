from dataclasses import dataclass
from typing import Optional
from kg.hypergraph_builder import HyperEdge

@dataclass
class SupportingFact:
    """One gold article and its relevant sentence indices."""
    title:        str
    sentence_ids: list[int]

    def num_sentences(self) -> int:
        return len(self.sentence_ids)
    
@dataclass
class HopResult:
    """One hop in a reasoning chain."""
    src:            str
    dst:            str
    edge:           Optional[HyperEdge]
    is_broken:      bool  = False
    hop_confidence: float = 1.0

@dataclass
class PathResult:
    """Full shortest path between two graph entities."""
    src:   str
    dst:   str
    hops:  list[HopResult]
    found: bool

    def num_hops(self) -> int:
        return len(self.hops)

    def has_broken_hops(self) -> bool:
        return any(h.is_broken for h in self.hops)

    def broken_hops(self) -> list[HopResult]:
        return [h for h in self.hops if h.is_broken]