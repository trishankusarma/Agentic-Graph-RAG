"""
kg/hypergraph_builder.py
 
Orchestrates hypergraph construction from text chunks.
Extraction is fully delegated to a BaseExtractor subclass —
swap OllamaExtractor for QwenExtractor with zero changes here.
 
Pipeline:
    chunks → extractor.extract(chunk) → facts → _add_fact_to_graph()
                                                → KnowledgeHypergraph
"""
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
 
from kg.data_loader import Chunk
from kg.extractors.base import BaseExtractor
from kg.extractors.ollama_extractor import OllamaBackend
from kg.extractors.qwen_extractor import OpenAIBackend
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class HyperEdge:
    edge_id:    str
    entities:   list[str]     # normalized entity ids
    relation:   str
    sentence_index: int
    sentence:   str           # source sentence from the chunk
    chunk_id:   str
    sample_id:  str
    confidence: float
    is_gold:    bool = False

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
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "gold_edges": gold_edges,
            "avg_edge_arity": round(avg_arity, 2),
        }

    def get_edges_for_entity(self, entity_id: str) -> list[HyperEdge]:
        node = self.nodes.get(entity_id)
        if not node:
            return []
        return [self.edges[eid] for eid in node.edges if eid in self.edges]
    
    def get_neighbors(self, entity_id: str) -> set[str]:
        """All entities reachable from entity_id via a single hyperEdge"""
        neighbors = set()
        for edge in self.get_edges_for_entity(entity_id):
            neighbors.update(edge.entities)
        neighbors.discard(entity_id)
        return neighbors

class HypergraphBuilder:
    """
    Builds a KnowledgeHypergraph from text chunks.
 
    Args:
        extractor:  Any BaseExtractor subclass (OllamaExtractor, QwenExtractor, ...)
        cache_path: if set, saves/loads the hypergraph as JSON
    """
    def __init__(
            self,
            extractor: BaseExtractor,
            cache_path: Optional[str] = None
    ):
        self.extractor  = extractor
        self.cache_path = cache_path
 
        if not self.extractor.is_available():
            raise RuntimeError(
                f"Extractor {type(extractor).__name__} is not available. "
                "Check your backend and model."
            )
    
    def build(self, chunks: list[Chunk]) -> KnowledgeHypergraph:
        """Process all chunks and return a KnowledgeHypergraph."""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading hypergraph from cache: {self.cache_path}")
            return self._load_from_cache(self.cache_path)
 
        graph = KnowledgeHypergraph(nodes={}, edges={})
 
        logger.info(f"Extracting hyperedges from {len(chunks)} chunks ...")
        for i, chunk in enumerate(chunks):
            facts = self.extractor.extract(chunk)
            for fact in facts:
                self._add_fact_to_graph(graph, fact, chunk)
 
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  processed {i+1}/{len(chunks)} chunks | "
                    f"nodes={graph.num_nodes()} edges={graph.num_edges()}"
                )
 
        logger.info(f"Hypergraph built: {graph.summary()}")

        if self.cache_path:
            self._save_to_cache(graph, self.cache_path)
            logger.info(f"Hypergraph cached to {self.cache_path}")
        
        return graph
    
    def _add_fact_to_graph(self, graph: KnowledgeHypergraph, fact: dict, chunk: Chunk)->None:
        """Add one extracted fact as a HyperEdge + update entity nodes."""
        raw_entities = fact["entities"]
        relation     = fact["relation"].strip().lower()
        sentence_index     = fact["sentence_index"]
        confidence = fact["confidence"]

        # normalize entities
        norm_entities = [self._normalize(e) for e in raw_entities]

        # skip if duplicate entities in the same fact
        if len(set(norm_entities)) < 2:
            return # skip degenrate facts

        # deterministic edge id from content
        edge_content = f"{sorted(norm_entities)}|{relation}|{chunk.chunk_id}"
        edge_id = "edge_" + hashlib.md5(edge_content.encode()).hexdigest()[:12]

        if edge_id in graph.edges:
            return # deduplicate
        
        # create hyperedge
        edge = HyperEdge(
            edge_id   = edge_id,
            entities  = norm_entities,
            relation  = relation,
            sentence_index = sentence_index,
            sentence  = chunk.sentences[sentence_index],
            chunk_id  = chunk.chunk_id,
            sample_id = chunk.sample_id,
            confidence = confidence,
            is_gold   = sentence_index in chunk.gold_sentence_offsets,
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
    
    def _save_to_cache(self, graph: KnowledgeHypergraph, path: str)->None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "nodes": {
                nid: {
                    "entity_id": n.entity_id, "label": n.label, 
                    "chunks": n.chunks, "edges": n.edges
                }
                for nid, n in graph.nodes.items()
            },
            "edges": {
                eid: {
                    "edge_id": e.edge_id, "entities": e.entities,
                    "relation": e.relation, "sentence_index": e.sentence_index,
                    "sentence": e.sentence, "chunk_id": e.chunk_id,
                    "sample_id": e.sample_id, "confidence": e.confidence,
                    "is_gold": e.is_gold
                }
                for eid, e in graph.edges.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    from kg.data_loader import HotpotQALoader
    from kg.extractors.ollama_extractor import OllamaBackend
    from kg.extractors.qwen_extractor import OpenAIBackend

    # Step 1: loading a tiny slice
    logger.info("1. Loading a slice of hotpot qa dataset")
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=1
    )

    samples = loader.load()

    # Step 2: building a hypergraph
    all_chunks = loader.get_all_chunks(samples)
    logger.info(f"2. Running extraction on {len(all_chunks)} chunks")

    # extractor = OllamaBackend(model="deepseek-r1:32b")
    extractor = OpenAIBackend(model="qwen2.5-32b", api_url="http://localhost:8000")
    builder = HypergraphBuilder(
        extractor=extractor,
        cache_path="data/hyper_graph_builder_qwen.json",
    )
    graph = builder.build(all_chunks)

    logger.info("== Hypergraph Summary ==")
    for k, v in graph.summary().items():
        print(f" {k}: {v}")
    
    # show edges for a known entity
    logger.info("== Sample edges ==")
    for edge in list(graph.edges.values())[:10]:
        print(f"\n  edge_id  : {edge.edge_id}")
        print(f"  entities : {edge.entities}")
        print(f"  relation : {edge.relation}")
        print(f"  sentence : {edge.sentence[:100]}")
        print(f"  is_gold  : {edge.is_gold}")