"""
kg/broken_hop_detector.py

Broken hop detection for HotpotQA samples.
Owns: chain stitching, segment building, entity-title pairing, bridge/comparison logic.

Depends on PathFinder for graph traversal and BaseExtractor for question entity extraction.
Does NOT own the NX graph or hypergraph — those live in PathFinder.
"""

import logging
from typing import Optional

from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.path_finder import PathFinder
from kg.reasoning_models import (
    BrokenHopReport, Chain, HopResult, PathResult,
    Segment, SupportingFact,
)

logger = logging.getLogger(__name__)


class BrokenHopDetector:
    """
    Detects broken hops in HotpotQA reasoning chains.

    Args:
        path_finder: PathFinder instance (owns graph + traversal)
        extractor:   BaseExtractor instance (extracts question entities via LLM)
    """

    def __init__(self, path_finder: PathFinder, extractor: BaseExtractor):
        self.path_finder = path_finder
        self.extractor   = extractor

    def check(self, sample: HotpotSample) -> BrokenHopReport:
        """
        Detect broken hops for one HotpotQA sample.

        src  → extracted from question via extractor.extract_entities()
        dst  → sample.answer (normalized)
        gold → sample.gold_sentences keys (article titles, in order)

        bridge:
            1 chain: src → gold[0] → gold[1] → ... → dst
        comparison:
            N chains: src_i → gold_title_i → dst  (one per paired entity/title)
        """
        dst               = self._normalize(sample.answer)
        question_entities = self.extractor.extract_entities(sample.question)
        gold_titles       = list(sample.gold_sentences.keys())

        supporting_facts = [
            SupportingFact(title=title, sentence_ids=sids)
            for title, sids in sample.gold_sentences.items()
        ]

        logger.info(
            f"[{sample.hop_type}] Q: {sample.question[:60]}... | "
            f"entities: {question_entities} | dst: {sample.answer}"
        )

        if sample.hop_type == "bridge":
            chains, is_answerable = self._check_bridge(
                question_entities, gold_titles, dst
            )
        elif sample.hop_type == "comparison":
            chains, is_answerable = self._check_comparison(
                question_entities, gold_titles, dst
            )
        else:
            logger.warning(f"Unknown hop_type '{sample.hop_type}' — {sample.sample_id}")
            chains, is_answerable = [], False

        return BrokenHopReport(
            sample_id         = sample.sample_id,
            question          = sample.question,
            answer            = sample.answer,
            hop_type          = sample.hop_type,
            supporting_facts  = supporting_facts,
            question_entities = question_entities,
            reasoning_chains  = chains,
            is_answerable     = is_answerable,
        )

    def _check_bridge(
        self,
        question_entities: list[str],
        gold_titles:       list[str],
        dst:               str,
    ) -> tuple[list[Chain], bool]:
        """
        Build one candidate chain per question entity as src,
        rank by num_unhealed_breaks, return the best.
        """
        if not question_entities:
            return [self._empty_chain(dst)], False

        candidates = [
            self._build_chain(self._normalize(e), gold_titles, dst)
            for e in question_entities
        ]
        best = min(candidates, key=lambda c: c.num_unhealed_breaks())
        return [best], best.is_clean()

    def _check_comparison(
        self,
        question_entities: list[str],
        gold_titles:       list[str],
        dst:               str,
    ) -> tuple[list[Chain], bool]:
        """
        Pair each question entity to its gold title via 3-tier matching.
        Build one chain per pair, pick best per title.
        Always returns exactly len(gold_titles) primary chains.
        """
        if not question_entities or not gold_titles:
            return [self._empty_chain(dst)], False

        pairs = self._pair_entities_to_titles(question_entities, gold_titles)

        chains_by_title: dict[str, list[Chain]] = {t: [] for t in gold_titles}
        unresolved: list[Chain] = []

        for entity, title in pairs:
            if title is None:
                unresolved.append(Chain(segments=[Segment(
                    from_node=entity, to_node="???",
                    path_result=PathResult(
                        src=self._normalize(entity), dst=dst, hops=[], found=False
                    ),
                    is_broken=True,
                )]))
                continue
            chain = self._build_chain(self._normalize(entity), [title], dst)
            chains_by_title[title].append(chain)

        chains: list[Chain] = []
        for title in gold_titles:
            candidates = chains_by_title[title]
            if candidates:
                chains.append(min(candidates, key=lambda c: c.num_unhealed_breaks()))
            else:
                chains.append(Chain(segments=[Segment(
                    from_node="???", to_node=title,
                    path_result=PathResult(src="", dst=dst, hops=[], found=False),
                    is_broken=True,
                )]))

        chains.extend(unresolved)   # audit visibility — not counted for is_answerable

        is_answerable = all(
            c.num_unhealed_breaks() < float("inf")
            for c in chains[:len(gold_titles)]
        )
        return chains, is_answerable
    
    def _build_chain(
        self,
        src:         str,
        gold_titles: list[str],
        dst:         str,
    ) -> Chain:
        """
        Stitch: src → gold[0] → gold[1] → ... → dst.
        On segment failure tries one skip-ahead fallback.
        """
        waypoints: list[tuple[str, set[str]]] = [("src", {src})]
        for title in gold_titles:
            waypoints.append((title, self.path_finder.entities_for_title(title)))
        waypoints.append(("dst", {dst}))

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

        return Chain(segments=segments)

    def _segment_between(
        self,
        from_node: str,
        from_ents: set[str],
        to_node:   str,
        to_ents:   set[str],
    ) -> Segment:
        """
        Find shortest clean path between two waypoint entity sets.
        Tries all (from_entity, to_entity) pairs, picks shortest clean result.
        """
        if not from_ents or not to_ents:
            return Segment(
                from_node=from_node, to_node=to_node,
                path_result=PathResult(src="", dst="", hops=[], found=False),
                is_broken=True,
            )

        best: Optional[PathResult] = None
        for f in from_ents:
            for t in to_ents:
                if f == t:
                    continue
                result = self.path_finder.edges_on_path(f, t)
                if result.found and not result.has_broken_hops():
                    if best is None or result.num_hops() < best.num_hops():
                        best = result

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
    def _empty_chain(dst: str) -> Chain:
        return Chain(segments=[Segment(
            from_node="src", to_node="dst",
            path_result=PathResult(src="", dst=dst, hops=[], found=False),
            is_broken=True,
        )])

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()