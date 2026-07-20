from dataclasses import dataclass, field
from typing import Optional
from .graph_models import PathResult
from .enums import TerminalStatus, FailureMode

@dataclass
class Segment:
    """
    One stitched segment of the intended reasoning chain, e.g. src → gold_title[0], or gold_title[0] → gold_title[1].

    Segments cover the src → gold-titles portion only. The answer hop lives in TerminalResult, because its failure means something categorically different.
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
    status:          TerminalStatus         # See TerminalStatus.
    answer:          str                    # Raw answer string from the dataset.
    answer_entities: list[str]            = field(default_factory=list)     # Graph entity ids the answer resolved to.
    path_result:     Optional[PathResult] = None    # Path from the last gold title to the answer entity, when one was attempted.
    text_grounded:   bool                 = False   # Whether the answer appears verbatim in a retrieved sentence. 
    # Distinguishes "never retrieved" from "retrieved but not extracted as an entity" — the latter is an extraction-recall problem, 
    # not a missing edge, and needs a different repair.
    is_trivial:      bool                 = False   # The answer entity is itself a waypoint the chain. already passed through, so ENTITY_REACHED was granted 
        # without testing any reasoning. Exclude from genuine. Exclude from genuine terminal-success counts.

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

        NOT_AN_ENTITY and PREDICATE do not count against answerability — the graph did its job; the answer simply isn't the
        kind of thing that lives in it.
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