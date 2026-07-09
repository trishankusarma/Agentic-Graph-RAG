"""
kg/reasoning_models.py

Pure dataclasses for the reasoning chain representation.
No logic, no imports from other kg modules — just data shapes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
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


@dataclass
class Segment:
    """
    One stitched segment of the intended reasoning chain,
    e.g. src → gold_title[0], or gold_title[0] → gold_title[1].
    """
    from_node:   str
    to_node:     str
    path_result: PathResult
    is_broken:   bool
    is_fallback: bool = False

    @property
    def label(self) -> str:
        return f"{self.from_node} -> {self.to_node}"


@dataclass
class Chain:
    """
    Full stitched chain for one src→dst reasoning path.
    1 chain for bridge, N chains for comparison.
    """
    segments: list[Segment]

    def num_unhealed_breaks(self) -> float:
        """
        0           → fully clean
        1..N        → N healed via skip-ahead (degraded but connected)
        float('inf')→ unhealed break — chain genuinely disconnected
        """
        count = 0
        i = 0
        while i < len(self.segments):
            seg = self.segments[i]
            if seg.is_broken:
                healed = (
                    i + 1 < len(self.segments)
                    and self.segments[i + 1].is_fallback
                    and not self.segments[i + 1].is_broken
                )
                if not healed:
                    return float("inf")
                count += 1
                i += 2
            else:
                i += 1
        return count

    def is_clean(self) -> bool:
        return self.num_unhealed_breaks() == 0

    def broken_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_broken]


@dataclass
class BrokenHopReport:
    """Complete broken hop analysis for one QA sample."""
    sample_id:         str
    question:          str
    answer:            str
    hop_type:          str                   # "bridge" | "comparison"
    supporting_facts:  list[SupportingFact]  # gold articles + sentence ids
    question_entities: list[str]             # extracted from question via LLM
    reasoning_chains:  list[Chain]           # 1 for bridge, N for comparison
    is_answerable:     bool

    def summary(self) -> dict:
        total_broken = sum(len(c.broken_segments()) for c in self.reasoning_chains)
        return {
            "sample_id":           self.sample_id,
            "hop_type":            self.hop_type,
            "is_answerable":       self.is_answerable,
            "question_entities":   self.question_entities,
            "answer":              self.answer,
            "num_reasoning_chains": len(self.reasoning_chains),
            "total_broken_hops":   total_broken,
            "supporting_titles":   [f.title for f in self.supporting_facts],
        }