"""
kg/detection/terminal.py

Classifies the terminal hop: gold[-1] -> answer.

Judged separately from chain connectivity — see
kg.reasoning_models.enums.TerminalStatus for the full reasoning. This is the
file to read if a terminal verdict looks wrong; it owns the entire
predicate / not-an-entity / unreached / trivial taxonomy in one place.
"""

from kg.data_loader import HotpotSample
from kg.reasoning_models.chains import TerminalResult
from kg.reasoning_models.enums import TerminalStatus

from .path_policy import best_path
from .utils import PREDICATE_ANSWERS, answer_in_text, normalize


class TerminalResolver:
    """Resolves and classifies one sample's answer against the graph."""

    def __init__(self, path_finder):
        self.path_finder = path_finder

    def resolve(
        self,
        sample:      HotpotSample,
        from_ents:   set[str],
        gold_titles: list[str],
    ) -> TerminalResult:
        """
        Args:
            from_ents:   Entity set of the last gold title — where the chain
                         stood before the answer hop. Empty if that title was
                         never reached.
            gold_titles: Needed only for the "answer equals a gold title
                         verbatim" trivial check.
        """
        answer      = sample.answer
        answer_norm = normalize(answer)
        grounded    = answer_in_text(sample)

        # predicate — no terminal hop exists at all
        if answer_norm in PREDICATE_ANSWERS:
            return TerminalResult(
                status=TerminalStatus.PREDICATE,
                answer=answer,
                text_grounded=grounded,
            )

        answer_entities = self.path_finder.resolve_entity(answer)

        # answer was never extracted as a node — not a connectivity failure
        if not answer_entities:
            return TerminalResult(
                status=TerminalStatus.NOT_AN_ENTITY,
                answer=answer,
                text_grounded=grounded,
            )

        trivial = self._is_trivial(answer_norm, answer_entities, from_ents, gold_titles)

        if not from_ents:
            return TerminalResult(
                status=TerminalStatus.ENTITY_UNREACHED,
                answer=answer,
                answer_entities=sorted(answer_entities),
                text_grounded=grounded,
                is_trivial=trivial,
            )

        # The answer entity IS the waypoint: no hop to make, none tested.
        if trivial and (answer_entities & from_ents):
            return TerminalResult(
                status=TerminalStatus.ENTITY_REACHED,
                answer=answer,
                answer_entities=sorted(answer_entities),
                text_grounded=grounded,
                is_trivial=True,
            )

        best = best_path(self.path_finder, from_ents, answer_entities)
        if best is None:
            return TerminalResult(
                status=TerminalStatus.ENTITY_UNREACHED,
                answer=answer,
                answer_entities=sorted(answer_entities),
                text_grounded=grounded,
                is_trivial=trivial,
            )

        return TerminalResult(
            status=TerminalStatus.ENTITY_REACHED,
            answer=answer,
            answer_entities=sorted(answer_entities),
            path_result=best,
            text_grounded=grounded,
            is_trivial=trivial,
        )

    @staticmethod
    def _is_trivial(
        answer_norm:     str,
        answer_entities: set[str],
        from_ents:       set[str],
        gold_titles:     list[str],
    ) -> bool:
        """
        Whether reaching the answer tested any reasoning at all.

        HotpotQA often makes the answer one of the supporting article titles
        ("David Weissman", "Animorphs"). The terminal then resolves to an
        entity the chain already passed through, granting ENTITY_REACHED for
        a hop of length zero. Counting those as genuine successes inflates
        the metric — a corpus can look "answerable" almost entirely on
        trivial terminals.
        """
        if answer_entities & from_ents:
            return True
        return any(normalize(t) == answer_norm for t in gold_titles)