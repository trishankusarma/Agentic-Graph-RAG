"""
kg/hypergraph_builder.py

Orchestrates hypergraph construction from text chunks.
Extraction is fully delegated to a BaseExtractor subclass —
swap OllamaBackend for OpenAIBackend with zero changes here.

Pipeline:
    chunks -> extractor.extract(chunk) -> facts -> _add_fact_to_graph()
                                                  -> KnowledgeHypergraph
"""
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

from kg.data_loader import Chunk
from kg.extractors.base import BaseExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class HyperEdge:
    edge_id:        str
    entities:       list[str]     # normalized entity ids
    relation:       str
    sentence_index: int
    sentence:       str           # source sentence from the chunk
    chunk_id:       str
    sample_id:      str
    confidence:     float
    is_gold:        bool = False


@dataclass
class EntityNode:
    entity_id:  str          # normalized: lowercased, stripped
    label:      str          # original surface form (first seen)
    chunks:     list[str] = field(default_factory=list)     # chunk_ids
    edges:      list[str] = field(default_factory=list)     # edge_ids


@dataclass
class KnowledgeHypergraph:
    nodes: dict[str, EntityNode]    # entity_id -> EntityNode
    edges: dict[str, HyperEdge]     # edge_id   -> HyperEdge

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def summary(self) -> dict:
        gold_edges = sum(1 for e in self.edges.values() if e.is_gold)
        avg_arity = (
            sum(len(e.entities) for e in self.edges.values()) / max(len(self.edges), 1)
        )
        return {
            "num_nodes":      self.num_nodes(),
            "num_edges":      self.num_edges(),
            "gold_edges":     gold_edges,
            "avg_edge_arity": round(avg_arity, 2),
        }

    def get_edges_for_entity(self, entity_id: str) -> list[HyperEdge]:
        node = self.nodes.get(entity_id)
        if not node:
            return []
        return [self.edges[eid] for eid in node.edges if eid in self.edges]

    def get_neighbors(self, entity_id: str) -> set[str]:
        """All entities reachable from entity_id via a single hyperedge."""
        neighbors = set()
        for edge in self.get_edges_for_entity(entity_id):
            neighbors.update(edge.entities)
        neighbors.discard(entity_id)
        return neighbors


def normalize_entity(label: str) -> str:
    return label.lower().strip()


def add_fact_to_graph(
    graph: "KnowledgeHypergraph", fact: dict, chunk: Chunk
) -> Optional["HyperEdge"]:
    """
    Add one extracted fact as a HyperEdge + update entity nodes. Returns the
    new HyperEdge, or None if the fact was degenerate or already present.
    """
    raw_entities   = fact["entities"]
    relation       = fact["relation"].strip().lower()
    sentence_index = fact["sentence_index"]
    confidence     = fact["confidence"]

    norm_entities = [normalize_entity(e) for e in raw_entities]

    # skip degenerate facts (self-loops, duplicate entities)
    if len(set(norm_entities)) < 2:
        return None

    # deterministic edge id from content
    edge_content = f"{sorted(norm_entities)}|{relation}|{chunk.chunk_id}"
    edge_id = "edge_" + hashlib.md5(edge_content.encode()).hexdigest()[:12]

    if edge_id in graph.edges:
        return None  # deduplicate — caller can tell "already known" from None

    edge = HyperEdge(
        edge_id        = edge_id,
        entities       = norm_entities,
        relation       = relation,
        sentence_index = sentence_index,
        sentence       = chunk.sentences[sentence_index],
        chunk_id       = chunk.chunk_id,
        sample_id      = chunk.sample_id,
        confidence     = confidence,
        is_gold        = sentence_index in chunk.gold_sentence_offsets,
    )
    graph.edges[edge_id] = edge

    for raw, norm in zip(raw_entities, norm_entities):
        if norm not in graph.nodes:
            graph.nodes[norm] = EntityNode(entity_id=norm, label=raw)
        node = graph.nodes[norm]
        if chunk.chunk_id not in node.chunks:
            node.chunks.append(chunk.chunk_id)
        if edge_id not in node.edges:
            node.edges.append(edge_id)

    return edge


def add_repaired_fact_to_graph(
    graph: "KnowledgeHypergraph",
    fact: dict,
    chunk: Chunk,
    path_finder,
    verbose: bool = False,
) -> tuple[Optional["HyperEdge"], list[tuple[str, str]]]:
    """
    Like add_fact_to_graph, but snaps each entity onto an EXISTING graph node when
    one unambiguously refers to the same thing. Returns (edge, snaps) where snaps is
    a list of (raw_surface_form, existing_node_id) pairs that were merged.
    """
    snaps: list[tuple[str, str]] = []
    resolved_entities = []

    for raw_entity in fact["entities"]:
        norm = normalize_entity(raw_entity)

        if norm in graph.nodes:
            resolved_entities.append(norm)          # already exact, nothing to do
            continue

        existing = path_finder.resolve_entity(raw_entity)
        if len(existing) == 1:
            node_id = next(iter(existing))
            resolved_entities.append(node_id)
            if node_id != norm:
                snaps.append((raw_entity, node_id))
                if verbose:
                    logger.info(f"  snapped {raw_entity!r} -> existing node {node_id!r}")
        else:
            # zero matches (genuinely new entity) or several (ambiguous) — keep the
            # extractor's own form rather than guessing
            resolved_entities.append(norm)

    snapped_fact = {**fact, "entities": resolved_entities}
    edge = add_fact_to_graph(graph, snapped_fact, chunk)
    return edge, snaps


class HypergraphBuilder:
    """
    Builds a KnowledgeHypergraph from text chunks.

    Args:
        extractor:     Any BaseExtractor subclass.
        cache_path:    If set, saves/loads the FINISHED hypergraph as JSON.
                       Unrelated to extract_cache below — this is a whole-graph
                       cache, keyed by nothing (one path = one graph).
        max_workers:   Concurrent LLM calls FOR THIS build() CALL'S chunk list
                       only. Should match the extractor's HTTP pool_size.
        extract_cache: Optional dict shared across builder instances, mapping
                       chunk-content hash -> fact list. Pass the SAME dict to
                       every builder in a run — required for both cross-sample
                       dedup (HotpotQA distractor paragraphs repeat heavily
                       across questions) and for warmup() to have any effect
                       on later build() calls.
    """

    def __init__(
        self,
        extractor:     BaseExtractor,
        cache_path:    Optional[str]  = None,
        max_workers:   int            = 32,
        extract_cache: Optional[dict] = None,
    ):
        self.extractor   = extractor
        self.cache_path  = cache_path
        self.max_workers = max_workers

        # Shared by reference when injected — never mutate cached fact dicts.
        self._extract_cache = extract_cache if extract_cache is not None else {}
        self._cache_lock    = threading.Lock()
        self._done          = 0
        self._total         = 0

    # ------------------------------------------------------------------ #
    # Warmup — the speed fix. Extraction only, no graph construction.
    # ------------------------------------------------------------------ #

    def warmup(self, chunks: list[Chunk]) -> dict:
        """
        Populate the extract cache for every chunk, WITHOUT building a graph.

        Call this once, up front, over the FULL flattened chunk list across
        every sample in the run — see the module docstring for why. Safe to
        call more than once or with overlapping chunk sets; extraction is
        already memoized by content hash.

        Returns the shared extract_cache dict (same object as
        self._extract_cache), so the caller can inspect hit rate:

            cache = builder.warmup(all_chunks)
            print(f"{len(all_chunks)} chunks -> {len(cache)} unique")
        """
        if not self.extractor.is_available():
            raise RuntimeError(
                f"Extractor {type(self.extractor).__name__} is not available. "
                "Check your backend and model."
            )

        self._total = len(chunks)
        self._done  = 0
        workers = max(1, min(self.max_workers, len(chunks)))

        logger.info(f"Warming extract cache for {len(chunks)} chunks ({workers} workers)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self._safe_extract, chunks))

        logger.info(
            f"Warmup complete — {len(chunks)} chunks, "
            f"{len(self._extract_cache)} unique cached"
        )
        return self._extract_cache

    # ------------------------------------------------------------------ #
    # Build — graph construction. Fast if warmup() already ran.
    # ------------------------------------------------------------------ #

    def build(self, chunks: list[Chunk]) -> KnowledgeHypergraph:
        """Process all chunks and return a KnowledgeHypergraph."""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading hypergraph from cache: {self.cache_path}")
            return self._load_from_cache(self.cache_path)
        
        if not self.extractor.is_available():
            raise RuntimeError(
                f"Extractor {type(self.extractor).__name__} is not available. "
                "Check your backend and model."
            )

        graph = KnowledgeHypergraph(nodes={}, edges={})

        self._total = len(chunks)
        self._done  = 0
        workers = max(1, min(self.max_workers, len(chunks)))

        logger.info(f"Extracting from {len(chunks)} chunks ({workers} workers)...")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map preserves input order -> deterministic edge insertion
            results = list(pool.map(self._safe_extract, chunks))

        for chunk, facts in zip(chunks, results):
            for fact in facts:
                self._add_fact_to_graph(graph, fact, chunk)

        logger.info(f"Hypergraph built: {graph.summary()}")

        if self.cache_path:
            self._save_to_cache(graph, self.cache_path)
            logger.info(f"Hypergraph cached to {self.cache_path}")

        return graph

    def _safe_extract(self, chunk: Chunk) -> list[dict]:
        """
        Extract facts for one chunk, memoized on sentence content.

        Runs on a worker thread. Never raises — a dead chunk yields no edges
        rather than killing the whole pool.
        """
        key = self._chunk_key(chunk)

        with self._cache_lock:
            cached = self._extract_cache.get(key)
        if cached is not None:
            self._tick()
            return cached

        try:
            facts = self.extractor.extract(chunk)
        except Exception as e:
            logger.error(f"extract failed for {chunk.chunk_id}: {e}")
            self._tick()
            return []

        with self._cache_lock:
            self._extract_cache.setdefault(key, facts)
        self._tick()
        return facts

    @staticmethod
    def _chunk_key(chunk: Chunk) -> str:
        """
        Content hash over the sentence LIST, not the joined text.

        Joining with a space would let ["a b", "c"] and ["a", "b c"] collide,
        and a cache hit across those would silently misalign sentence_index.
        """
        joined = "\x00".join(chunk.sentences)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    def _tick(self) -> None:
        """Progress counter — pool.map blocks, so without this a long run is silent."""
        with self._cache_lock:
            self._done += 1
            done = self._done
        if done % 50 == 0 or done == self._total:
            logger.info(f"  extracted {done}/{self._total} chunks")

    def _add_fact_to_graph(
        self, graph: KnowledgeHypergraph, fact: dict, chunk: Chunk
    ) -> Optional["HyperEdge"]:
        return add_fact_to_graph(graph, fact, chunk)

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()

    def _load_from_cache(self, path: str) -> KnowledgeHypergraph:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = {nid: EntityNode(**n) for nid, n in data["nodes"].items()}
        edges = {eid: HyperEdge(**e) for eid, e in data["edges"].items()}
        return KnowledgeHypergraph(nodes=nodes, edges=edges)

    def _save_to_cache(self, graph: KnowledgeHypergraph, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in graph.nodes.items()},
            "edges": {eid: asdict(e) for eid, e in graph.edges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)