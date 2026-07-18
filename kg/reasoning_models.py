"""
kg/reasoning_models.py

Pure dataclasses for the reasoning chain representation.

Terminal handling
-----------------
The chain src → gold[0] → ... → gold[n] is a graph-connectivity question.
The final hop gold[n] → answer often is NOT, and conflating them makes the
audit meaningless:

  - Comparison answers ("yes"/"no") are computed predicates over two articles.
    They are not nodes in any knowledge graph and never will be.
  - Bridge answers that are values rather than entities ("3,677 seated",
    "from 1986 to 2013") are likewise unreachable as nodes, even when the text
    stating them sits right there in the retrieved context.

Only ENTITY_UNREACHED is a genuine broken terminal hop. That, plus mid-chain
breaks, is the population a repair action can act on.

Trivial terminals
-----------------
HotpotQA frequently makes the answer one of the supporting article titles
("David Weissman", "Animorphs", "Kansas Song"). The terminal then resolves to
an entity the chain already passed through, so ENTITY_REACHED is recorded
without any reasoning having been tested. TerminalResult.is_trivial marks
these so they can be excluded when reporting genuine terminal success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kg.hypergraph_builder import HyperEdge


class TerminalStatus(str, Enum):
    """How the answer relates to the graph."""

    ENTITY_REACHED   = "entity_reached"     # answer is a node, path found
    ENTITY_UNREACHED = "entity_unreached"   # answer is a node, no path — REAL broken hop
    NOT_AN_ENTITY    = "not_an_entity"      # answer never extracted as a node
    PREDICATE        = "predicate"          # yes/no comparison — no terminal hop exists


class FailureMode(str, Enum):
    """
    Single-label classification of one sample, for histogramming.

    Repairable by a graph action:  BROKEN_MID_CHAIN, BROKEN_TERMINAL,
                                   PREDICATE_BROKEN
    Not a graph failure:           ANSWER_NOT_ENTITY, PREDICATE_OK
    Upstream failure:              NO_QUESTION_ENTITIES
    """

    CONNECTED            = "connected"
    HEALED               = "healed"                 # connected via skip-ahead
    BROKEN_MID_CHAIN     = "broken_mid_chain"       # a gold title is unreachable
    BROKEN_TERMINAL      = "broken_terminal"        # answer is a node, no path to it
    ANSWER_NOT_ENTITY    = "answer_not_entity"      # chain fine, answer isn't a node
    PREDICATE_OK         = "predicate_ok"           # comparison, both articles reached
    PREDICATE_BROKEN     = "predicate_broken"       # comparison, an article unreachable
    NO_QUESTION_ENTITIES = "no_question_entities"   # entity extraction returned nothing


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

    Segments cover the src → gold-titles portion only. The answer hop lives in
    TerminalResult, because its failure means something categorically different.
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
class TerminalResult:
    """
    The final answer hop, judged separately from chain connectivity.

    Args:
        status:          See TerminalStatus.
        answer:          Raw answer string from the dataset.
        answer_entities: Graph entity ids the answer resolved to.
        path_result:     Path from the last gold title to the answer entity,
                         when one was attempted.
        text_grounded:   Whether the answer appears verbatim in a retrieved
                         sentence. Distinguishes "never retrieved" from
                         "retrieved but not extracted as an entity" — the
                         latter is an extraction-recall problem, not a missing
                         edge, and needs a different repair.
        is_trivial:      The answer entity is itself a waypoint the chain
                         already passed through, so ENTITY_REACHED was granted
                         without testing any reasoning. Exclude from genuine
                         terminal-success counts.
    """
    status:          TerminalStatus
    answer:          str
    answer_entities: list[str]            = field(default_factory=list)
    path_result:     Optional[PathResult] = None
    text_grounded:   bool                 = False
    is_trivial:      bool                 = False

    @property
    def is_broken_hop(self) -> bool:
        """True only for a genuine unreachable-entity terminal."""
        return self.status is TerminalStatus.ENTITY_UNREACHED

    @property
    def is_genuine_success(self) -> bool:
        """Reached, and not because the answer was already a waypoint."""
        return (
            self.status is TerminalStatus.ENTITY_REACHED
            and not self.is_trivial
        )

    @property
    def label(self) -> str:
        triv = " (trivial)" if self.is_trivial else ""
        return f"terminal[{self.status.value}{triv}] -> {self.answer[:40]}"


@dataclass
class Chain:
    """
    Full stitched chain for one src → gold-titles reasoning path,
    plus its terminal verdict.

    1 chain for bridge, N chains for comparison.
    """
    segments: list[Segment]
    terminal: Optional[TerminalResult] = None

    # ---- segment connectivity (answer hop excluded) ------------------- #

    def num_unhealed_breaks(self) -> float:
        """
        0            → fully clean
        1..N         → N breaks healed via skip-ahead (degraded but connected)
        float('inf') → unhealed break — chain genuinely disconnected
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

    def is_connected(self) -> bool:
        """All gold titles reachable, skip-ahead heals permitted."""
        return self.num_unhealed_breaks() < float("inf")

    def is_clean(self) -> bool:
        """All gold titles reachable with no degradation."""
        return self.num_unhealed_breaks() == 0

    def num_heals(self) -> int:
        n = self.num_unhealed_breaks()
        return 0 if n == float("inf") else int(n)

    def broken_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_broken]

    # ---- combined verdict --------------------------------------------- #

    def is_answerable(self) -> bool:
        """
        Chain connected AND the terminal is not a genuine broken hop.

        NOT_AN_ENTITY and PREDICATE do not count against answerability — the
        graph did its job; the answer simply isn't the kind of thing that lives
        in it.
        """
        if not self.is_connected():
            return False
        if self.terminal is None:
            return True
        return not self.terminal.is_broken_hop

    def failure_mode(self) -> FailureMode:
        if not self.is_connected():
            if self.terminal and self.terminal.status is TerminalStatus.PREDICATE:
                return FailureMode.PREDICATE_BROKEN
            return FailureMode.BROKEN_MID_CHAIN

        status = self.terminal.status if self.terminal else None

        if status is TerminalStatus.ENTITY_UNREACHED:
            return FailureMode.BROKEN_TERMINAL
        if status is TerminalStatus.NOT_AN_ENTITY:
            return FailureMode.ANSWER_NOT_ENTITY
        if status is TerminalStatus.PREDICATE:
            return FailureMode.PREDICATE_OK
        if self.num_heals() > 0:
            return FailureMode.HEALED
        return FailureMode.CONNECTED


@dataclass
class BrokenHopReport:
    """Complete broken hop analysis for one QA sample."""
    sample_id:         str
    question:          str
    answer:            str
    hop_type:          str                   # "bridge" | "comparison"
    supporting_facts:  list[SupportingFact]
    question_entities: list[str]
    reasoning_chains:  list[Chain]           # 1 for bridge, N for comparison
    is_answerable:     bool
    failure_mode:      FailureMode
    parse_failures:    int = 0               # chunks whose extraction was unusable

    # ------------------------------------------------------------------ #

    def primary_chains(self) -> list[Chain]:
        """Chains that count toward the verdict (excludes unresolved audit rows)."""
        return [c for c in self.reasoning_chains if c.terminal is not None]

    def terminals(self) -> list[TerminalResult]:
        return [c.terminal for c in self.primary_chains() if c.terminal]

    def terminal_statuses(self) -> list[TerminalStatus]:
        return [t.status for t in self.terminals()]

    def has_trivial_terminal(self) -> bool:
        return any(t.is_trivial for t in self.terminals())

    def has_genuine_terminal(self) -> bool:
        return any(t.is_genuine_success for t in self.terminals())

    def is_text_grounded(self) -> bool:
        return any(t.text_grounded for t in self.terminals())

    def is_clean(self) -> bool:
        """Answerable with no skip-ahead degradation anywhere."""
        return self.is_answerable and all(
            c.is_clean() for c in self.primary_chains()
        )

    def num_heals(self) -> int:
        return sum(c.num_heals() for c in self.reasoning_chains)

    def is_repairable(self) -> bool:
        """
        Whether a graph repair action could plausibly change the verdict.

        Excludes samples that failed for reasons no edge insertion can fix
        (answer not an entity, no question entities extracted).
        """
        return self.failure_mode in (
            FailureMode.BROKEN_MID_CHAIN,
            FailureMode.BROKEN_TERMINAL,
            FailureMode.PREDICATE_BROKEN,
        )

    def summary(self) -> dict:
        total_broken = sum(len(c.broken_segments()) for c in self.reasoning_chains)
        return {
            "sample_id":            self.sample_id,
            "hop_type":             self.hop_type,
            "failure_mode":         self.failure_mode.value,
            "is_answerable":        self.is_answerable,
            "is_clean":             self.is_clean(),
            "is_repairable":        self.is_repairable(),
            "question_entities":    self.question_entities,
            "answer":               self.answer,
            "terminal":             [s.value for s in self.terminal_statuses()],
            "trivial_terminal":     self.has_trivial_terminal(),
            "genuine_terminal":     self.has_genuine_terminal(),
            "text_grounded":        self.is_text_grounded(),
            "num_reasoning_chains": len(self.reasoning_chains),
            "total_broken_hops":    total_broken,
            "num_heals":            self.num_heals(),
            "parse_failures":       self.parse_failures,
            "supporting_titles":    [f.title for f in self.supporting_facts],
        }