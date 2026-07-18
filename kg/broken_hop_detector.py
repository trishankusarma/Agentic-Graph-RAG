"""
kg/broken_hop_detector.py

Broken hop detection for HotpotQA samples.
Owns: chain stitching, segment building, entity-title pairing, terminal
resolution, bridge/comparison logic.

Chain shape:
    segments:  src → gold[0] → ... → gold[n]     (pure graph connectivity)
    terminal:  gold[n] → answer                  (judged separately)

See reasoning_models for why the terminal is judged separately and what
"trivial" means.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.path_finder import PathFinder
from kg.reasoning_models import (
    BrokenHopReport, Chain, FailureMode, PathResult,
    Segment, SupportingFact, TerminalResult, TerminalStatus,
)

logger = logging.getLogger(__name__)

# Answers that are computed predicates, not entities.
PREDICATE_ANSWERS = {"yes", "no"}


class BrokenHopDetector:
    """
    Detects broken hops in HotpotQA reasoning chains.

    Args:
        path_finder: PathFinder instance (owns graph + traversal)
        extractor:   BaseExtractor instance (extracts question entities via LLM)
        entity_map:  Optional dict sample_id → question entities, pre-computed
                     concurrently via prefetch_entities().
    """

    def __init__(
        self,
        path_finder: PathFinder,
        extractor:   BaseExtractor,
        entity_map:  Optional[dict[str, list[str]]] = None,
    ):
        self.path_finder = path_finder
        self.extractor   = extractor
        self.entity_map  = entity_map if entity_map is not None else {}

    # ------------------------------------------------------------------ #
    # Entity prefetch
    # ------------------------------------------------------------------ #

    @staticmethod
    def prefetch_entities(
        extractor:   BaseExtractor,
        samples:     list[HotpotSample],
        max_workers: int                            = 32,
        into:        Optional[dict[str, list[str]]] = None,
    ) -> dict[str, list[str]]:
        """
        Extract question entities for every sample concurrently, up front.
        Returns the (possibly pre-existing) dict so it can be shared by reference.
        """
        result = into if into is not None else {}
        todo   = [s for s in samples if s.sample_id not in result]
        if not todo:
            return result

        workers = max(1, min(max_workers, len(todo)))
        logger.info(
            f"Prefetching question entities for {len(todo)} samples "
            f"({workers} workers)..."
        )

        def _one(sample: HotpotSample) -> list[str]:
            try:
                return extractor.extract_entities(sample.question)
            except Exception as e:
                logger.error(f"entity extraction failed for {sample.sample_id}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sample, entities in zip(todo, pool.map(_one, todo)):
                result[sample.sample_id] = entities

        return result

    def _entities_for(self, sample: HotpotSample) -> list[str]:
        cached = self.entity_map.get(sample.sample_id)
        if cached is not None:
            return cached
        entities = self.extractor.extract_entities(sample.question)
        self.entity_map[sample.sample_id] = entities
        return entities

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def check(
        self,
        sample:         HotpotSample,
        parse_failures: int = 0,
    ) -> BrokenHopReport:
        """
        Detect broken hops for one HotpotQA sample.

        Args:
            sample:         The HotpotQA sample.
            parse_failures: Chunks in this sample whose extraction was unusable.
                            Samples with dropped chunks should not be pooled
                            with clean ones when computing repair rates.
        """
        question_entities = self._entities_for(sample)
        gold_titles       = list(sample.gold_sentences.keys())

        supporting_facts = [
            SupportingFact(title=title, sentence_ids=sids)
            for title, sids in sample.gold_sentences.items()
        ]

        logger.info(
            f"[{sample.hop_type}] Q: {sample.question[:60]}... | "
            f"entities: {question_entities} | answer: {sample.answer}"
        )

        if not question_entities:
            chains = [Chain(
                segments=[self._broken_segment("src", "dst")],
                terminal=self._terminal_for(sample, set(), gold_titles),
            )]
            return self._report(
                sample, supporting_facts, question_entities, chains,
                FailureMode.NO_QUESTION_ENTITIES, parse_failures,
            )

        if sample.hop_type == "bridge":
            chains = self._check_bridge(sample, question_entities, gold_titles)
        elif sample.hop_type == "comparison":
            chains = self._check_comparison(sample, question_entities, gold_titles)
        else:
            logger.warning(f"Unknown hop_type '{sample.hop_type}' — {sample.sample_id}")
            chains = []

        primary = [c for c in chains if c.terminal is not None]
        mode    = self._aggregate_failure_mode(primary)

        return self._report(
            sample, supporting_facts, question_entities, chains,
            mode, parse_failures,
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

    @staticmethod
    def _aggregate_failure_mode(primary: list[Chain]) -> FailureMode:
        """
        Worst mode across chains, by severity. A comparison sample where one
        article is unreachable is broken even if the other is fine.
        """
        if not primary:
            return FailureMode.BROKEN_MID_CHAIN

        order = [
            FailureMode.BROKEN_MID_CHAIN,
            FailureMode.PREDICATE_BROKEN,
            FailureMode.BROKEN_TERMINAL,
            FailureMode.ANSWER_NOT_ENTITY,
            FailureMode.HEALED,
            FailureMode.PREDICATE_OK,
            FailureMode.CONNECTED,
        ]
        modes = {c.failure_mode() for c in primary}
        for mode in order:
            if mode in modes:
                return mode
        return FailureMode.CONNECTED

    # ------------------------------------------------------------------ #
    # Terminal resolution
    # ------------------------------------------------------------------ #

    def _terminal_for(
        self,
        sample:      HotpotSample,
        from_ents:   set[str],
        gold_titles: list[str],
    ) -> TerminalResult:
        """
        Classify the answer and, when it is a reachable entity, path to it.

        from_ents is the entity set of the last gold title — the point the
        chain reached before the answer hop.
        """
        answer      = sample.answer
        answer_norm = self._normalize(answer)
        grounded    = self._answer_in_text(sample)

        # predicate — no terminal hop exists
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

        trivial = self._is_trivial_terminal(
            answer_norm, answer_entities, from_ents, gold_titles
        )

        if not from_ents:
            return TerminalResult(
                status=TerminalStatus.ENTITY_UNREACHED,
                answer=answer,
                answer_entities=sorted(answer_entities),
                text_grounded=grounded,
                is_trivial=trivial,
            )

        # The answer entity IS the waypoint: no hop to make, and none tested.
        if trivial and (answer_entities & from_ents):
            return TerminalResult(
                status=TerminalStatus.ENTITY_REACHED,
                answer=answer,
                answer_entities=sorted(answer_entities),
                text_grounded=grounded,
                is_trivial=True,
            )

        best = self._best_path(from_ents, answer_entities)
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

    def _is_trivial_terminal(
        self,
        answer_norm:     str,
        answer_entities: set[str],
        from_ents:       set[str],
        gold_titles:     list[str],
    ) -> bool:
        """
        Whether reaching the answer tested any reasoning at all.

        HotpotQA often makes the answer one of the supporting article titles
        ("David Weissman", "Animorphs", "Kansas Song"). The terminal then
        resolves to an entity the chain already passed through, and
        ENTITY_REACHED is granted for a hop of length zero. Counting those as
        terminal successes inflates the metric.
        """
        if answer_entities & from_ents:
            return True
        return any(self._normalize(t) == answer_norm for t in gold_titles)

    @staticmethod
    def _answer_in_text(sample: HotpotSample) -> bool:
        """
        Whether the answer string appears verbatim in any retrieved sentence.

        Separates "never retrieved" from "retrieved but not extracted as an
        entity" — the second is an extraction-recall failure, not a missing
        edge, and wants a different repair action.

        Word-boundary matched, and predicates are excluded outright: plain
        substring made "no" match inside "not", "known" and "Nolan", so every
        comparison sample reported as text-grounded regardless of content.
        """
        needle = sample.answer.lower().strip()
        if not needle or needle in PREDICATE_ANSWERS:
            return False
        pattern = r"\b" + re.escape(needle) + r"\b"
        return any(
            re.search(pattern, sentence.lower())
            for chunk in sample.chunks
            for sentence in chunk.sentences
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
        """
        Build one candidate chain per question entity as src, return the best.
        Ranked by segment connectivity first, then terminal reachability.
        """
        candidates = [
            self._build_chain(sample, self._normalize(e), gold_titles)
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
        Pair each question entity to its gold title via 3-tier matching.
        Build one chain per pair, pick best per title.
        Always returns exactly len(gold_titles) primary chains, plus any
        unresolved entities appended for audit visibility.
        """
        if not gold_titles:
            return [Chain(
                segments=[self._broken_segment("src", "dst")],
                terminal=self._terminal_for(sample, set(), gold_titles),
            )]

        pairs = self._pair_entities_to_titles(question_entities, gold_titles)

        chains_by_title: dict[str, list[Chain]] = {t: [] for t in gold_titles}
        unresolved: list[Chain] = []

        for entity, title in pairs:
            if title is None:
                # terminal=None marks this as an audit row, not a verdict row
                unresolved.append(Chain(
                    segments=[self._broken_segment(entity, "???")],
                    terminal=None,
                ))
                continue
            chains_by_title[title].append(
                self._build_chain(sample, self._normalize(entity), [title])
            )

        chains: list[Chain] = []
        for title in gold_titles:
            candidates = chains_by_title[title]
            if candidates:
                chains.append(min(candidates, key=self._chain_rank))
            else:
                chains.append(Chain(
                    segments=[self._broken_segment("???", title)],
                    terminal=self._terminal_for(sample, set(), gold_titles),
                ))

        chains.extend(unresolved)
        return chains

    @staticmethod
    def _chain_rank(chain: Chain) -> tuple:
        """
        Sort key: fewest unhealed breaks, then a reached terminal, then prefer
        a non-trivial terminal so a chain that actually traversed something
        wins over one that landed on its own waypoint.
        """
        terminal_penalty = 0
        trivial_penalty  = 0
        if chain.terminal is not None:
            if chain.terminal.is_broken_hop:
                terminal_penalty = 1
            if chain.terminal.is_trivial:
                trivial_penalty = 1
        return (chain.num_unhealed_breaks(), terminal_penalty, trivial_penalty)

    # ------------------------------------------------------------------ #
    # Chain stitching
    # ------------------------------------------------------------------ #

    def _build_chain(
        self,
        sample:      HotpotSample,
        src:         str,
        gold_titles: list[str],
    ) -> Chain:
        """
        Stitch src → gold[0] → ... → gold[n], then resolve the terminal.
        On segment failure tries one skip-ahead fallback.
        """
        waypoints: list[tuple[str, set[str]]] = [("src", {src})]
        for title in gold_titles:
            waypoints.append((title, self.path_finder.entities_for_title(title)))

        segments: list[Segment] = []
        i = 0
        while i < len(waypoints) - 1:
            from_node, from_ents = waypoints[i]
            to_node,   to_ents   = waypoints[i + 1]

            seg = self._segment_between(from_node, from_ents, to_node, to_ents)

            if seg.is_broken and i + 2 < len(waypoints):
                skip_node, skip_ents = waypoints[i + 2]
                fallback = self._segment_between(
                    from_node, from_ents, skip_node, skip_ents
                )
                fallback.is_fallback = True

                if not fallback.is_broken:
                    segments.append(seg)       # record the break
                    segments.append(fallback)  # record the successful skip
                    i += 2
                    continue

            segments.append(seg)
            i += 1

        # terminal departs from the last gold title's entities
        last_ents = waypoints[-1][1] if len(waypoints) > 1 else {src}
        terminal  = self._terminal_for(sample, last_ents, gold_titles)

        return Chain(segments=segments, terminal=terminal)

    def _segment_between(
        self,
        from_node: str,
        from_ents: set[str],
        to_node:   str,
        to_ents:   set[str],
    ) -> Segment:
        """Shortest clean path between two waypoint entity sets."""
        if not from_ents or not to_ents:
            return self._broken_segment(from_node, to_node)

        best = self._best_path(from_ents, to_ents)
        if best is None:
            f0, t0 = next(iter(from_ents)), next(iter(to_ents))
            return Segment(
                from_node=from_node, to_node=to_node,
                path_result=PathResult(src=f0, dst=t0, hops=[], found=False),
                is_broken=True,
            )

        return Segment(
            from_node=from_node, to_node=to_node,
            path_result=best, is_broken=False,
        )

    def _best_path(
        self,
        from_ents: set[str],
        to_ents:   set[str],
    ) -> Optional[PathResult]:
        """
        Shortest clean path over all (from, to) pairs.

        Cost is |from_ents| x |to_ents| shortest-path searches, which is why
        PathFinder.entities_for_title tier 3 is deliberately strict.
        """
        best: Optional[PathResult] = None
        for f in from_ents:
            for t in to_ents:
                if f == t:
                    continue
                result = self.path_finder.edges_on_path(f, t)
                if result.found and not result.has_broken_hops():
                    if best is None or result.num_hops() < best.num_hops():
                        best = result
        return best

    @staticmethod
    def _broken_segment(from_node: str, to_node: str) -> Segment:
        return Segment(
            from_node=from_node, to_node=to_node,
            path_result=PathResult(src="", dst="", hops=[], found=False),
            is_broken=True,
        )

    def _pair_entities_to_titles(
        self,
        question_entities: list[str],
        gold_titles:       list[str],
    ) -> list[tuple[str, Optional[str]]]:
        """
        3-tier pairing: exact string → substring → graph neighbor.
        Titles consumed greedily — each matched at most once.
        """
        remaining = list(gold_titles)
        pairs: list[tuple[str, Optional[str]]] = []

        for entity in question_entities:
            entity_norm = self._normalize(entity)
            matched = None

            # tier 1 — exact
            for title in remaining:
                if self._normalize(title) == entity_norm:
                    matched = title
                    break

            # tier 2 — substring
            if matched is None:
                for title in remaining:
                    title_norm = self._normalize(title)
                    if entity_norm in title_norm or title_norm in entity_norm:
                        matched = title
                        break

            # tier 3 — graph neighbor
            if matched is None:
                neighbors = self.path_finder.neighbors(entity_norm)
                for title in remaining:
                    if neighbors.intersection(
                        self.path_finder.entities_for_title(title)
                    ):
                        matched = title
                        break

            if matched is not None:
                remaining.remove(matched)
            pairs.append((entity, matched))

        return pairs

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()