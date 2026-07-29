"""
kg/graph_store.py

Thin facade composing PathFinder + BrokenHopDetector.

Usage:
    extractor = OpenAIBackend(model="qwen3-14b", pool_size=64)
    builder   = HypergraphBuilder(extractor=extractor, max_workers=64)
    graph     = builder.build(loader.get_all_chunks(samples))

    store = GraphStore(graph, extractor)
    store.index_samples(samples)
    report = store.check_broken_hops(sample)
"""

import logging
from typing import Optional

from kg.detection import BrokenHopDetector, prefetch_entities
from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.graph import PathFinder
from kg.hypergraph_builder import KnowledgeHypergraph
from kg.reasoning_models import BrokenHopReport

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        extractor:  BaseExtractor,
        directed:   bool                           = False,
        entity_map: Optional[dict[str, list[str]]] = None,
    ):
        self.extractor   = extractor
        self.path_finder = PathFinder(hypergraph, directed=directed, use_semantic=False)
        self.detector    = BrokenHopDetector(
            self.path_finder, extractor, entity_map=entity_map
        )

    @staticmethod
    def prefetch_entities(extractor, samples, max_workers=32, into=None):
        return prefetch_entities(extractor, samples, max_workers=max_workers, into=into)

    def index_samples(self, samples) -> None:
        self.path_finder.index_samples(samples)

    def check_broken_hops(self, sample: HotpotSample) -> BrokenHopReport:
        parse_failures = self.extractor.failure_count_for(
            [c.chunk_id for c in sample.chunks]
        )
        return self.detector.check(sample, parse_failures=parse_failures)

    def stats(self) -> dict:
        return self.path_finder.stats()


def aggregate(reports: list[BrokenHopReport]) -> dict:
    from collections import Counter

    total = len(reports)
    if not total:
        return {}
    clean_inputs = [r for r in reports if r.parse_failures == 0]
    return {
        "total_samples":                   total,
        "answerable":                      sum(1 for r in reports if r.is_answerable),
        "clean":                           sum(1 for r in reports if r.is_clean()),
        "repairable":                      sum(1 for r in reports if r.is_repairable()),
        "genuine_terminal":                sum(1 for r in reports if r.has_genuine_terminal()),
        "trivial_terminal":                sum(1 for r in reports if r.has_trivial_terminal()),
        "failure_modes":                   dict(Counter(r.failure_mode.value for r in reports)),
        "terminal_statuses":               dict(Counter(
            s.value for r in reports for s in r.terminal_statuses()
        )),
        "text_grounded":                   sum(1 for r in reports if r.is_text_grounded()),
        "samples_with_parse_failures":     total - len(clean_inputs),
        "answerable_excl_parse_failures":
            f"{sum(1 for r in clean_inputs if r.is_answerable)}/{len(clean_inputs)}",
    }


if __name__ == "__main__":
    import json
    from collections import Counter

    from kg.data_loader import HotpotQALoader
    from kg.extractors import OpenAIBackend, contamination_rate
    from kg.hypergraph_builder import HypergraphBuilder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Match this to your vLLM launch flag: --max-num-seqs 64
    POOL = 64

    loader = HotpotQALoader(
        split="validation", chunk_size=5, overlap=1, max_samples=200,
        cache_path="data/data_loader_cache.jsonl",
    )
    samples = loader.load()

    extractor = OpenAIBackend(
        model="qwen3-14b", api_url="http://localhost:8001", pool_size=POOL,
    )

    entity_map = prefetch_entities(extractor, samples, max_workers=POOL)
    all_chunks    = loader.get_all_chunks(samples)
    extract_cache: dict = {}
    warmup_builder = HypergraphBuilder(
        extractor=extractor, max_workers=POOL, extract_cache=extract_cache,
    )
    warmup_builder.warmup(all_chunks)
    logger.info(
        f"Cache: {len(all_chunks)} total chunks -> "
        f"{len(extract_cache)} unique extracted"
    )

    reports:         list[BrokenHopReport] = []
    relation_counts: Counter               = Counter()
    arity_counts:    Counter               = Counter()
    contam_samples:  list[float]           = []

    for sample in samples:
        # max_workers barely matters here now — every chunk is a cache hit
        # after warmup, so build() runs single-threaded in practice.
        builder = HypergraphBuilder(
            extractor=extractor, max_workers=POOL, extract_cache=extract_cache,
        )
        graph = builder.build(sample.chunks)

        relation_counts.update(e.relation for e in graph.edges.values())
        arity_counts.update(len(e.entities) for e in graph.edges.values())
        contam_samples.append(contamination_rate(graph))

        store = GraphStore(graph, extractor=extractor, entity_map=entity_map)
        store.index_samples([sample])

        report = store.check_broken_hops(sample)
        reports.append(report)

    with open("data/reports.jsonl", "w") as f:
        for r in reports:
            f.write(json.dumps(r.summary()) + "\n")

    logger.info(f"Extractor stats: {extractor.stats()}")

    logger.info("\n=== Corpus Rollup ===")
    for k, v in aggregate(reports).items():
        logger.info(f"  {k:<32}: {v}")

    logger.info(f"\n  avg contamination_rate: {sum(contam_samples)/len(contam_samples):.1%}")

    logger.info("\n=== Edge arity distribution ===")
    for arity, count in sorted(arity_counts.items()):
        logger.info(f"  arity {arity}: {count}")

    logger.info("\n=== Top relation labels ===")
    for label, count in relation_counts.most_common(15):
        logger.info(f"  {count:>4}  {label}")