"""
kg/reasoning_models/chains.py

Segment, TerminalResult, Chain — the shape of one reasoning attempt.

Reconstructed to match what report.py already depends on (c.terminal,
c.is_clean(), c.num_heals(), t.is_trivial, t.status, etc.) — diff this against
your actual file before replacing it; only the src_unresolved addition below
is new, everything else should already match.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import FailureMode, TerminalStatus
from .graph_models import PathResult


@dataclass
class Segment:
    """One stitched leg of a chain: from_node -> to_node."""
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
    Verdict on the answer hop: gold[-1] -> answer.

    status:          See TerminalStatus.
    answer_entities: Graph node ids the answer resolved to, if any.
    path_result:     Path from the last gold title to the answer, when found.
    text_grounded:   Whether the answer string appears verbatim in retrieved
                      text — separates "never retrieved" from "retrieved but
                      not extracted as a node."
    is_trivial:      The answer entity is itself a waypoint already visited —
                      ENTITY_REACHED granted for a hop of length zero, so it
                      shouldn't count as a genuine terminal success.
    """
    status:          TerminalStatus
    answer:          str
    answer_entities: list[str]            = field(default_factory=list)
    path_result:     Optional[PathResult] = None
    text_grounded:   bool                 = False
    is_trivial:      bool                 = False

    @property
    def is_broken_hop(self) -> bool:
        return self.status is TerminalStatus.ENTITY_UNREACHED

    @property
    def is_genuine_success(self) -> bool:
        return self.status is TerminalStatus.ENTITY_REACHED and not self.is_trivial

    @property
    def label(self) -> str:
        triv = " (trivial)" if self.is_trivial else ""
        return f"terminal[{self.status.value}{triv}] -> {self.answer[:40]}"


@dataclass
class Chain:
    """
    One src -> gold[0] -> ... -> gold[n] -> answer attempt.
    """
    segments:       list[Segment]
    terminal:       Optional[TerminalResult] = None
    src_unresolved: bool                     = False

    # ---- segment connectivity (answer hop excluded) ------------------- #

    def num_unhealed_breaks(self) -> float:
        """
        0            -> fully clean
        1..N         -> N breaks healed via skip-ahead
        float('inf') -> unhealed break — chain genuinely disconnected
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
        return self.num_unhealed_breaks() < float("inf")

    def is_clean(self) -> bool:
        return self.num_unhealed_breaks() == 0

    def num_heals(self) -> int:
        n = self.num_unhealed_breaks()
        return 0 if n == float("inf") else int(n)

    def broken_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_broken]

    # ---- combined verdict --------------------------------------------- #

    def is_answerable(self) -> bool:
        if not self.is_connected():
            return False
        if self.terminal is None:
            return True
        return not self.terminal.is_broken_hop

    def failure_mode(self) -> FailureMode:
        # Checked FIRST, ahead of connectivity: an unresolved src is always
        # disconnected too (see num_unhealed_breaks above), so without this
        # check it would silently fall through to BROKEN_MID_CHAIN below and
        # the two causes would be indistinguishable again.
        if self.src_unresolved:
            return FailureMode.SRC_UNRESOLVED

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