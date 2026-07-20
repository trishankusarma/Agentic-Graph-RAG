"""
kg/detection/prefetch.py

Question-entity extraction. Only ever touches `extractor` and the samples,
never path_finder or the graph — that's what makes it safe to run before any
graph exists, which is exactly how graph_store.py's __main__ uses it:

    entity_map = prefetch_entities(extractor, samples, max_workers=32)
    # ... build graph ...
    detector = BrokenHopDetector(path_finder, extractor, entity_map)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from kg.data_loader import HotpotSample
from kg.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


def prefetch_entities(
    extractor:   BaseExtractor,
    samples:     list[HotpotSample],
    max_workers: int                            = 32,
    into:        Optional[dict[str, list[str]]] = None,
) -> dict[str, list[str]]:
    """
    Extract question entities for every sample concurrently, up front.

    Returns the (possibly pre-existing) `into` dict, so the caller keeps the
    same reference across every builder/detector in a run.
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


class QuestionEntityCache:
    """
    Read-through cache in front of extractor.extract_entities.

    entity_map is normally pre-populated by prefetch_entities() and shared by
    reference across every detector in a run. get() only makes a live
    (blocking) call when a sample was missed by the prefetch — e.g. a new
    sample added after warm-up.
    """

    def __init__(
        self,
        extractor:  BaseExtractor,
        entity_map: Optional[dict[str, list[str]]] = None,
    ):
        self.extractor  = extractor
        self.entity_map = entity_map if entity_map is not None else {}

    def get(self, sample: HotpotSample) -> list[str]:
        """this is the class's whole public API,
        a leading underscore on the only method anyone calls was backwards."""
        cached = self.entity_map.get(sample.sample_id)
        if cached is not None:
            return cached
        entities = self.extractor.extract_entities(sample.question)
        self.entity_map[sample.sample_id] = entities
        return entities