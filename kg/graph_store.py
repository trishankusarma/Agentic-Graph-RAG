"""
kg/graph_store.py

Thin facade composing PathFinder + BrokenHopDetector.
This is the single public entry point for main.py.

Usage:
    extractor = OpenAIBackend(model="qwen3-14b", api_url="http://localhost:8000")
    builder   = HypergraphBuilder(extractor=extractor, cache_path="...")
    graph     = builder.build(loader.get_all_chunks(samples))

    store = GraphStore(graph, extractor)
    store.index_samples(samples)          # wire chunk→title metadata

    report = store.check_broken_hops(sample)
    stats  = store.stats()
"""

import logging

from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor
from kg.hypergraph_builder import KnowledgeHypergraph
from kg.broken_hop_detector import BrokenHopDetector
from kg.reasoning_models import BrokenHopReport
from kg.path_finder import PathFinder

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Composes PathFinder and BrokenHopDetector into a single interface.

    Args:
        hypergraph: KnowledgeHypergraph from HypergraphBuilder.build()
        extractor:  BaseExtractor — same instance used for KG construction.
                    Used here for question entity extraction only.
        directed:   Pass True to use DiGraph (rare; undirected is better for path finding).
    """

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        extractor:  BaseExtractor,
        directed:   bool = False,
    ):
        self.path_finder = PathFinder(hypergraph, directed=directed)
        self.detector    = BrokenHopDetector(self.path_finder, extractor)

    def index_samples(self, samples) -> None:
        """
        Wire chunk→title metadata into PathFinder.
        Call this after init for accurate entity-title resolution.
        """
        self.path_finder.index_samples(samples)

    def check_broken_hops(self, sample: HotpotSample) -> BrokenHopReport:
        """Detect broken hops for one HotpotQA sample."""
        return self.detector.check(sample)

    def stats(self) -> dict:
        """Graph statistics from PathFinder."""
        return self.path_finder.stats()

if __name__ == "__main__":
    from kg.data_loader import HotpotQALoader
    from kg.hypergraph_builder import HypergraphBuilder
    from kg.extractors.ollama_extractor import OllamaBackend
    from kg.extractors.qwen_extractor import OpenAIBackend
    loader  = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=20,
    )
    samples = loader.load()

    extractor = OpenAIBackend(model="qwen3-14b", api_url="http://localhost:8000")

    for sample in samples:
        builder   = HypergraphBuilder(
            extractor=extractor,
        )
        graph = builder.build(sample.chunks)

        store = GraphStore(graph, extractor=extractor)
        store.index_samples([sample])

        print("\n=== Graph Stats ===")
        for k, v in store.stats().items():
            print(f"  {k}: {v}")

        print("\n=== Broken Hop Audit ===")
        report = store.check_broken_hops(sample)
        print(f"\n  Q: {sample.question[:80]}")
        for k, v in report.summary().items():
            print(f"  {k:<24}: {v}")

        for i, chain in enumerate(report.reasoning_chains):
            status = "clean" if chain.is_clean() else "BROKEN"
            print(f"\n  chain_{i+1} ({status}):")
            for seg in chain.segments:
                flag     = "⚠ BROKEN" if seg.is_broken else "✓"
                fb       = " (skip-ahead)" if seg.is_fallback else ""
                hops_str = ""
                if seg.path_result.found:
                    rels     = [
                        h.edge.relation if h.edge else "???"
                        for h in seg.path_result.hops
                    ]
                    hops_str = f"  via [{', '.join(rels)}]"
                print(f"    {seg.label}  {flag}{fb}{hops_str}")