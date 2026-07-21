"""
kg/graph_store.py

Thin facade composing PathFinder + BrokenHopDetector.
This is the single public entry point for main.py.

Usage:
    extractor = OpenAIBackend(model="qwen3-14b", pool_size=32)
    builder   = HypergraphBuilder(extractor=extractor, max_workers=32)
    graph     = builder.build(loader.get_all_chunks(samples))

    store = GraphStore(graph, extractor)
    store.index_samples(samples)
    report = store.check_broken_hops(sample)
"""

import logging
from collections import Counter
from typing import Optional

from kg.detection.detector import BrokenHopDetector
from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.hypergraph_builder import KnowledgeHypergraph
from kg.graph import PathFinder
from kg.reasoning_models import BrokenHopReport

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Composes PathFinder and BrokenHopDetector into a single interface.

    Args:
        hypergraph: KnowledgeHypergraph from HypergraphBuilder.build()
        extractor:  BaseExtractor — same instance used for KG construction.
        directed:   Pass True to use DiGraph (undirected is better for paths).
        entity_map: Optional shared dict sample_id → question entities.
    """

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        extractor:  BaseExtractor,
        directed:   bool                           = False,
        entity_map: Optional[dict[str, list[str]]] = None,
    ):
        self.extractor   = extractor
        self.path_finder = PathFinder(hypergraph, directed=directed)
        self.detector    = BrokenHopDetector(
            self.path_finder, extractor, entity_map=entity_map
        )

    @staticmethod
    def prefetch_entities(
        extractor:   BaseExtractor,
        samples:     list[HotpotSample],
        max_workers: int                            = 32,
        into:        Optional[dict[str, list[str]]] = None,
    ) -> dict[str, list[str]]:
        """Concurrently extract question entities for all samples up front."""
        return BrokenHopDetector.prefetch_entities(
            extractor, samples, max_workers=max_workers, into=into
        )

    def index_samples(self, samples) -> None:
        """Wire chunk→title metadata into PathFinder."""
        self.path_finder.index_samples(samples)

    def check_broken_hops(self, sample: HotpotSample) -> BrokenHopReport:
        """
        Detect broken hops for one sample.

        Parse failures for this sample's chunks are counted and attached, so a
        sample whose chunks were dropped by truncation can be excluded from
        pooled repair statistics rather than silently counted as broken.
        """
        parse_failures = 0
        if hasattr(self.extractor, "failure_count_for"):
            parse_failures = self.extractor.failure_count_for(
                [c.chunk_id for c in sample.chunks]
            )
        return self.detector.check(sample, parse_failures=parse_failures)

    def stats(self) -> dict:
        """Graph statistics from PathFinder."""
        return self.path_finder.stats()


def aggregate(reports: list[BrokenHopReport]) -> dict:
    """
    Corpus-level rollup.

    The headline number is repairable/total, not answerable/total — samples
    that failed because the answer is a predicate or was never an entity are
    not broken hops, and no edge insertion can move them. Pooling those with
    genuine connectivity breaks was what produced the misleading 2/20 in the
    first full run.
    """
    total = len(reports)
    if not total:
        return {}

    clean_inputs = [r for r in reports if r.parse_failures == 0]

    return {
        "total_samples":     total,
        "answerable":        sum(1 for r in reports if r.is_answerable),
        "clean":             sum(1 for r in reports if r.is_clean()),
        "repairable":        sum(1 for r in reports if r.is_repairable()),
        "failure_modes":     dict(Counter(r.failure_mode.value for r in reports)),
        "terminal_statuses": dict(Counter(
            s.value for r in reports for s in r.terminal_statuses()
        )),
        "text_grounded":     sum(
            1 for r in reports if r.summary()["text_grounded"]
        ),
        "samples_with_parse_failures": total - len(clean_inputs),
        "answerable_excl_parse_failures":
            f"{sum(1 for r in clean_inputs if r.is_answerable)}/{len(clean_inputs)}",
    }


if __name__ == "__main__":
    from kg.data_loader import HotpotQALoader
    from kg.extractors.qwen_extractor import OpenAIBackend
    from kg.hypergraph_builder import HypergraphBuilder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    POOL = 32

    loader  = HotpotQALoader(
        split="validation", chunk_size=5, overlap=1, max_samples=200,
    )
    samples = loader.load()

    extractor = OpenAIBackend(
        model="qwen3-14b", api_url="http://localhost:8000", pool_size=POOL,
    )

    entity_map    = GraphStore.prefetch_entities(extractor, samples, max_workers=POOL)
    extract_cache: dict = {}

    reports:        list[BrokenHopReport] = []
    relation_counts: Counter              = Counter()
    arity_counts:    Counter              = Counter()

    for sample in samples:
        builder = HypergraphBuilder(
            extractor=extractor, max_workers=POOL, extract_cache=extract_cache,
        )
        graph = builder.build(sample.chunks)

        # prompt-contamination + arity diagnostics, pooled across samples
        relation_counts.update(e.relation for e in graph.edges.values())
        arity_counts.update(len(e.entities) for e in graph.edges.values())

        store = GraphStore(graph, extractor=extractor, entity_map=entity_map)
        store.index_samples([sample])

        logger.info("\n=== Graph Stats ===")
        for k, v in store.stats().items():
            logger.info(f"  {k}: {v}")

        logger.info("\n=== Broken Hop Audit ===")
        report = store.check_broken_hops(sample)
        reports.append(report)

        logger.info(f"\n  Q: {sample.question[:80]}")
        for k, v in report.summary().items():
            logger.info(f"  {k:<24}: {v}")

        for i, chain in enumerate(report.reasoning_chains):
            status = "clean" if chain.is_clean() else (
                "connected" if chain.is_connected() else "BROKEN"
            )
            logger.info(f"\n  chain_{i+1} ({status}):")
            for seg in chain.segments:
                flag     = "⚠ BROKEN" if seg.is_broken else "✓"
                fb       = " (skip-ahead)" if seg.is_fallback else ""
                hops_str = ""
                if seg.path_result.found:
                    rels = [
                        h.edge.relation if h.edge else "???"
                        for h in seg.path_result.hops
                    ]
                    hops_str = f"  via [{', '.join(rels)}]"
                logger.info(f"    {seg.label}  {flag}{fb}{hops_str}")

            if chain.terminal is not None:
                t    = chain.terminal
                mark = "⚠ BROKEN" if t.is_broken_hop else "✓"
                gnd  = " [text-grounded]" if t.text_grounded else ""
                logger.info(f"    {t.label}  {mark}{gnd}")

    logger.info("\n\n=== Corpus Rollup ===")
    for k, v in aggregate(reports).items():
        logger.info(f"  {k:<32}: {v}")

    logger.info("\n=== Edge arity distribution ===")
    logger.info("  (all-2 means the hypergraph is really just a graph)")
    for arity, count in sorted(arity_counts.items()):
        logger.info(f"  arity {arity}: {count}")

    logger.info("\n=== Top relation labels (prompt contamination check) ===")
    logger.info("  (a demo label dominating a mixed corpus means the prompt leaks)")
    for label, count in relation_counts.most_common(15):
        logger.info(f"  {count:>4}  {label}")