"""
kg/hypergraph_builder.py

Orchestrates hypergraph construction from text chunks.
Extraction is fully delegated to a BaseExtractor subclass —
swap OllamaBackend for OpenAIBackend with zero changes here.

Pipeline:
    chunks → extractor.extract(chunk) → facts → _add_fact_to_graph()
                                                → KnowledgeHypergraph

Concurrency model:
    LLM calls run in a ThreadPoolExecutor (vLLM batches them server-side).
    Graph mutation stays single-threaded, after the pool drains, in chunk
    order — so the resulting graph is deterministic regardless of thread
    scheduling.
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


class HypergraphBuilder:
    """
    Builds a KnowledgeHypergraph from text chunks.

    Args:
        extractor:     Any BaseExtractor subclass.
        cache_path:    If set, saves/loads the finished hypergraph as JSON.
        max_workers:   Concurrent LLM calls. Should match the extractor's HTTP
                       pool_size — if the pool is smaller, threads block waiting
                       for a connection and the extra workers buy nothing.
        extract_cache: Optional dict shared across builder instances, mapping
                       chunk-content hash → fact list. HotpotQA distractor
                       paragraphs repeat heavily across samples, so when you
                       build one graph per sample this avoids re-extracting the
                       same text. Pass the SAME dict to every builder; an
                       instance-local cache would be discarded each iteration.
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

    def build(self, chunks: list[Chunk]) -> KnowledgeHypergraph:
        """Process all chunks and return a KnowledgeHypergraph."""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading hypergraph from cache: {self.cache_path}")
            return self._load_from_cache(self.cache_path)

        # Checked here rather than in __init__: a cache hit above needs no
        # backend at all, and constructing one builder per sample would
        # otherwise fire a health-check round trip per sample.
        if not self.extractor.is_available():
            raise RuntimeError(
                f"Extractor {type(self.extractor).__name__} is not available. "
                "Check your backend and model."
            )

        graph = KnowledgeHypergraph(nodes={}, edges={})

        self._total  = len(chunks)
        self._done   = 0
        workers      = max(1, min(self.max_workers, len(chunks)))

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
        if done % 25 == 0 or done == self._total:
            logger.info(f"  extracted {done}/{self._total} chunks")

    def _add_fact_to_graph(
        self, graph: KnowledgeHypergraph, fact: dict, chunk: Chunk
    ) -> None:
        """
        Add one extracted fact as a HyperEdge + update entity nodes.

        Single-threaded by contract — called only after the pool has drained.

        Note on the shared extract cache: is_gold is computed from the chunk
        being processed right now, not from whichever chunk first populated the
        cache entry. Two samples sharing identical paragraph text but different
        supporting facts therefore still get correct gold labels.
        """
        raw_entities   = fact["entities"]
        relation       = fact["relation"].strip().lower()
        sentence_index = fact["sentence_index"]
        confidence     = fact["confidence"]

        norm_entities = [self._normalize(e) for e in raw_entities]

        # skip degenerate facts (self-loops, duplicate entities)
        if len(set(norm_entities)) < 2:
            return

        # deterministic edge id from content
        edge_content = f"{sorted(norm_entities)}|{relation}|{chunk.chunk_id}"
        edge_id = "edge_" + hashlib.md5(edge_content.encode()).hexdigest()[:12]

        if edge_id in graph.edges:
            return  # deduplicate

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

        # upsert entity nodes
        for raw, norm in zip(raw_entities, norm_entities):
            if norm not in graph.nodes:
                graph.nodes[norm] = EntityNode(entity_id=norm, label=raw)
            node = graph.nodes[norm]
            if chunk.chunk_id not in node.chunks:
                node.chunks.append(chunk.chunk_id)
            if edge_id not in node.edges:
                node.edges.append(edge_id)

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
        """asdict() rather than hand-listing fields — adding a field to
        HyperEdge would otherwise silently drop it from the cache."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in graph.nodes.items()},
            "edges": {eid: asdict(e) for eid, e in graph.edges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    from kg.data_loader import HotpotQALoader
    from kg.extractors.qwen_extractor import OpenAIBackend

    logger.info("1. Loading a slice of the HotpotQA dataset")
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=5,
    )
    samples = loader.load()

    all_chunks = loader.get_all_chunks(samples)
    logger.info(f"2. Running extraction on {len(all_chunks)} chunks")

    POOL = 32
    extractor = OpenAIBackend(
        model="qwen3-14b",
        api_url="http://localhost:8000",
        pool_size=POOL,
    )
    builder = HypergraphBuilder(
        extractor=extractor,
        cache_path="data/hyper_graph_builder_qwen.json",
        max_workers=POOL,
    )
    graph = builder.build(all_chunks)

    logger.info("== Hypergraph Summary ==")
    for k, v in graph.summary().items():
        print(f" {k}: {v}")

    logger.info("== Sample edges ==")
    for edge in list(graph.edges.values())[:10]:
        print(f"\n  edge_id  : {edge.edge_id}")
        print(f"  entities : {edge.entities}")
        print(f"  relation : {edge.relation}")
        print(f"  sentence : {edge.sentence[:100]}")
        print(f"  is_gold  : {edge.is_gold}")