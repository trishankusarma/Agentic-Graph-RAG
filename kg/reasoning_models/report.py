from dataclasses import dataclass

from .chains import Chain, TerminalResult
from .enums import FailureMode, REPAIRABLE_MODES, TerminalStatus
from .graph_models import SupportingFact


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
        """
        return self.failure_mode in REPAIRABLE_MODES

    def chain_terminal_details(self) -> list[dict]:
        return [
            {
                "status":       c.terminal.status.value,
                "is_trivial":   c.terminal.is_trivial,
                "is_genuine":   c.terminal.is_genuine_success,
                "is_connected": c.is_connected(),
            }
            for c in self.primary_chains()
        ]

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
            "chain_terminals":      self.chain_terminal_details(),
            "trivial_terminal":     self.has_trivial_terminal(),
            "genuine_terminal":     self.has_genuine_terminal(),
            "text_grounded":        self.is_text_grounded(),
            "num_reasoning_chains": len(self.reasoning_chains),
            "total_broken_hops":    total_broken,
            "num_heals":            self.num_heals(),
            "parse_failures":       self.parse_failures,
            "supporting_titles":    [f.title for f in self.supporting_facts],
        }