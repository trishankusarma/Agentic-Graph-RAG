"""
kg/detection/detector.py

Orchestrates one sample's broken-hop check by delegating to:
    QuestionEntityCache  — question -> entities             (prefetch.py)
    Pairing              — entity -> gold title              (pairing.py)
    TerminalResolver      — gold[-1] -> answer                (terminal.py)
    ChainBuilder          — src -> gold[0] -> ... -> gold[n]  (chain_builder.py)
    enums.worst_mode      — chain verdicts -> one FailureMode

This file should be orchestration ONLY. If you're about to add graph logic or
terminal-classification logic here, it belongs in one of the files above
instead — that discipline is the entire point of the split.
"""

import logging
from typing import Optional

from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.reasoning_models.chains import Chain
from kg.reasoning_models.enums import FailureMode, worst_mode
from kg.reasoning_models.graph_models import SupportingFact
from kg.reasoning_models.report import BrokenHopReport

from .chain_builder import ChainBuilder
from .pairing import Pairing
from .prefetch import QuestionEntityCache, prefetch_entities
from .terminal import TerminalResolver
from .utils import normalize

logger = logging.getLogger(__name__)


class BrokenHopDetector:
    """
    Detects broken hops in HotpotQA reasoning chains.

    Args:
        path_finder: PathFinder instance (owns graph + traversal).
        extractor:   BaseExtractor (extracts question entities via LLM).
        entity_map:  Optional dict sample_id -> question entities, usually
                     pre-populated by kg.detection.prefetch.prefetch_entities()
                     and shared by reference across every detector in a run.
    """

    def __init__(
        self,
        path_finder,
        extractor:  BaseExtractor,
        entity_map: Optional[dict[str, list[str]]] = None,
    ):
        self.path_finder   = path_finder
        self.entities      = QuestionEntityCache(extractor, entity_map)
        self.terminal       = TerminalResolver(path_finder)
        self.pairing        = Pairing(path_finder)
        self.chain_builder  = ChainBuilder(path_finder, self.terminal)

    @staticmethod
    def prefetch_entities(extractor, samples, max_workers=32, into=None):
        """
        Backwards-compatible pass-through, so graph_store.py's
        `GraphStore.prefetch_entities(...)` call site keeps working unchanged.
        New code should import prefetch_entities directly from
        kg.detection.prefetch instead of going through the detector class.
        """
        return prefetch_entities(extractor, samples, max_workers, into)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def check(
        self,
        sample:         HotpotSample,
        parse_failures: int = 0,
    ) -> BrokenHopReport:
        """
        Args:
            parse_failures: Chunks in this sample whose extraction was
                            unusable. Samples with dropped chunks should not
                            be pooled with clean ones when computing repair
                            rates.
        """
        question_entities = self.entities.get(sample)
        gold_titles        = list(sample.gold_sentences.keys())

        supporting_facts = [
            SupportingFact(title=title, sentence_ids=sids)
            for title, sids in sample.gold_sentences.items()
        ]

        logger.info(
            f"[{sample.hop_type}] Q: {sample.question[:60]}... | "
            f"entities: {question_entities} | answer: {sample.answer}"
        )

        if not question_entities:
            chains = [self._empty_chain(sample, gold_titles)]
            return self._report(
                sample, supporting_facts, question_entities, chains,
                FailureMode.NO_QUESTION_ENTITIES, parse_failures,
            )
        
        if not any(self.path_finder.resolve_entity(e) for e in question_entities):
            chains = [self._empty_chain(sample, gold_titles)]
            return self._report(
                sample, supporting_facts, question_entities, chains,
                FailureMode.SRC_UNRESOLVED, parse_failures,
            )

        if sample.hop_type == "bridge":
            chains = self._check_bridge(sample, question_entities, gold_titles)
        elif sample.hop_type == "comparison":
            chains = self._check_comparison(sample, question_entities, gold_titles)
        else:
            logger.warning(f"Unknown hop_type '{sample.hop_type}' — {sample.sample_id}")
            chains = []

        primary = [c for c in chains if c.terminal is not None]
        mode = (
            worst_mode(c.failure_mode() for c in primary)
            if primary else FailureMode.BROKEN_MID_CHAIN
        )

        return self._report(
            sample, supporting_facts, question_entities, chains, mode, parse_failures,
        )

    def _empty_chain(self, sample: HotpotSample, gold_titles: list[str]) -> Chain:
        """One broken segment + a resolved terminal — used for the two
        early-exit cases where there's no real chain to build at all."""
        return Chain(
            segments=[self.chain_builder.broken_segment("src", "dst")],
            terminal=self.terminal.resolve(sample, set(), gold_titles),
        )

    def _report(
        self, sample, supporting_facts, question_entities, chains,
        mode: FailureMode, parse_failures: int,
    ) -> BrokenHopReport:
        primary = [c for c in chains if c.terminal is not None]
        return BrokenHopReport(
            sample_id         = sample.sample_id,
            question          = sample.question,
            answer            = sample.answer,
            hop_type          = sample.hop_type,
            supporting_facts  = supporting_facts,
            question_entities = question_entities,
            reasoning_chains  = chains,
            is_answerable     = bool(primary) and all(
                c.is_answerable() for c in primary
            ),
            failure_mode      = mode,
            parse_failures    = parse_failures,
        )

    # ------------------------------------------------------------------ #
    # Bridge / comparison
    # ------------------------------------------------------------------ #

    def _check_bridge(
        self,
        sample:            HotpotSample,
        question_entities: list[str],
        gold_titles:       list[str],
    ) -> list[Chain]:
        """One candidate chain per question entity as src; keep the best."""
        candidates = [
            self.chain_builder.build(sample, normalize(e), gold_titles)
            for e in question_entities
        ]
        return [min(candidates, key=self._chain_rank)]

    def _check_comparison(
        self,
        sample:            HotpotSample,
        question_entities: list[str],
        gold_titles:       list[str],
    ) -> list[Chain]:
        """
        Pair each question entity to its gold title, build one chain per
        pair, keep the best per title. Always returns exactly len(gold_titles)
        primary chains, plus any unresolved entities for audit visibility.
        """
        if not gold_titles:
            return [self._empty_chain(sample, gold_titles)]

        pairs = self.pairing.pair_entities_to_titles(question_entities, gold_titles)

        chains_by_title: dict[str, list[Chain]] = {t: [] for t in gold_titles}
        unresolved: list[Chain] = []

        for entity, title in pairs:
            if title is None:
                # terminal=None marks this as an audit row, not a verdict row
                unresolved.append(Chain(
                    segments=[self.chain_builder.broken_segment(entity, "???")],
                    terminal=None,
                ))
                continue
            chains_by_title[title].append(
                self.chain_builder.build(sample, normalize(entity), [title])
            )

        chains: list[Chain] = []
        for title in gold_titles:
            candidates = chains_by_title[title]
            if candidates:
                chains.append(min(candidates, key=self._chain_rank))
            else:
                chains.append(Chain(
                    segments=[self.chain_builder.broken_segment("???", title)],
                    terminal=self.terminal.resolve(sample, set(), gold_titles),
                ))

        chains.extend(unresolved)
        return chains

    @staticmethod
    def _chain_rank(chain: Chain) -> tuple:
        """
        Sort key: fewest unhealed breaks, then a reached terminal, then prefer
        a non-trivial terminal so a chain that actually traversed something
        wins over one that landed on its own waypoint.

        Stays here rather than in ChainBuilder: this ranks candidate chains
        against each other, a detector-level strategy choice, not part of
        building any single chain.
        """
        terminal_penalty = 0
        trivial_penalty  = 0
        if chain.terminal is not None:
            if chain.terminal.is_broken_hop:
                terminal_penalty = 1
            if chain.terminal.is_trivial:
                trivial_penalty = 1
        return (chain.num_unhealed_breaks(), terminal_penalty, trivial_penalty)