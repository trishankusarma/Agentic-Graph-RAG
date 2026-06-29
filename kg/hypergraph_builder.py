"""
kg/hypergraph_builder.py

Builds a knowledge hypergraph G = (V, E_H) from tex chunks.

Each hyperedge is an n-ary relational fact:
    {
        "edge_id":   str,
        "entities":  [str, ...], # 2+ participating entities
        "relation":  str,        # semantic relation label
        "sentence":  str,        # source sentence
        "chunk_id":  str,        # back-reference to chunk
        "sample_id": str,
        "is_gold":   bool
    }

Node (entity) structure:
    {
        "entity_id":  str,              # normalized lowercase
        "label":      str,              # original surface form
        "chunks":     [chunk_id, ...]   # all chunks this entity appears in
        "edges":      [edge_id, ...]    # all hyperedges this entity participates in
    }

Pipeline per chunk:
    chunk text
        -> LLM extraction prompt
        -> JSON list of (entities, relation, sentence) triples
        -> deduplicate entities (normalize)
        -> build hyperedge objects
        -> update node index
"""
import logging
import os
import json
import requests
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from .data_loader import Chunk

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hyper-parameters
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_URL = "http://localhost:11434"

@dataclass
class HyperEdge:
    edge_id:    str
    entities:   list[str]     # normalized entity ids
    relation:   str
    sentence:   str           # source sentence from the chunk
    chunk_id:   str
    sample_id:  str
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

# LLM Extraction prompt
EXTRACTION_SYSTEM_PROMPT = """You are a Knowledge Graph extraction engine.

Given a passage of text, extract all n-ary relational facts

Each fact must have:
-"entities": a list of 2 or more entity strings that participate in this relation
-"relation": a short label for the relationship (e.g. "directed_by", "nationality", "founded_in", "spounse_of")
-"sentence": the exact sentence from the passage this fact comes from

Rules:
- Entities must be proper nouns or named concepts (people, places, orgs, works, dates, nationalities)
- Extract all facts you can find, including implicit ones
- Each sentence can yield multiple facts
- Relation labels must be snake_case and concise
- Do not include generic facts like "is a person" or "exists"
- Return ONLY a JSON array. No explanation, no markdown fences.

Example output:
[
  {
    "entities": ["Christopher Nolan", "The Dark Knight", "2008"],
    "relation": "directed_in_year",
    "sentence": "Christopher Nolan directed The Dark Knight in 2008."
  },
  {
    "entities": ["Christopher Nolan", "British-American"],
    "relation": "nationality",
    "sentence": "Christopher Nolan is a British-American filmmaker."
  }
]
"""

class HypergraphBuilder:
    """
    Extracts n-ary relational facts from text chunks via LLM and builds a KnowledgeHypergraph

    Args:
        model:          Ollama model name (default: deepseek-r1:32b)
        ollama_url:     Ollama API base URL
        max_tokens:     max tokens for LLM response
        retry_limit:    number of retries on parse failure
        retry_delay:    seconds between retries
        batch_delay:    seconds between chunk calls (rate limit buffer)
        cache_path:     if set, saves/loads the hypergraph as JSON
    """
    def __init__(
            self,
            model:      str = MODEL_NAME,
            ollama_url: str = OLLAMA_URL,
            max_tokens: int = 4096,
            retry_limit: int = 2,
            retry_delay: float = 1.0,
            batch_delay: float = 0.0, # No rate limit on local
            cache_path: Optional[str] = None,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.max_tokens = max_tokens
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.batch_delay = batch_delay
        self.cache_path = cache_path
        self._verify_ollama()
    
    def build(self, chunks: list[Chunk]) -> KnowledgeHypergraph:
        """
        Main entry: Process all chunks and return a Knowledge Hypergraph
        Loads from cache if available
        """
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading hypergraph from cache: {self.cache_path}")
            return self._load_from_cache(self.cache_path)
        
        graph = KnowledgeHypergraph(nodes={}, edges={})

        logger.info(f"Extracting hyperedges from {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            facts = self._extract_facts(chunk)
            for fact in facts:
                self._add_fact_to_graph(graph, fact, chunk)
            
            if (i+1) % 10 == 0:
                logger.info(f" processed {i+1}/{len(chunks)} chunks | "
                            f"nodes={graph.num_nodes()} edges={graph.num_edges()}")
            
            if self.batch_delay > 0:
                time.sleep(self.batch_delay)
            
        logger.info(f"Hypergraph built: {graph.summary()}")

        if self.cache_path:
            self._save_to_cache(graph, self.cache_path)
            logger.info(f"Hypergraph cached to {self.cache_path}")
        
        return graph
    
    def _extract_facts(self, chunk: Chunk) -> list[dict]:
        """
        Call LLM on a single chunk. Return list of raw fact dicts:
            [{"entities": [...], "relation": str, "sentence": str}, ...]
        Retries on JSON parse failures
        """
        prompt = f"Extract all relational facts from this passage:\n\n{' '.join(chunk.sentences).strip()}"

        for attempt in range(self.retry_limit + 1):
            try:
                raw = self._call_ollama(prompt)
                facts = json.loads(raw)

                # basic validation
                valid = []
                for f in facts:
                    if (
                        isinstance(f, dict)
                        and isinstance(f.get("entities"), list)
                        and len(f["entities"]) >= 2
                        and isinstance(f.get("relation"), str)
                        and f["relation"].strip()
                        and isinstance(f.get("sentence"), str)
                        and f["sentence"].strip()
                    ):
                        valid.append(f)
                return valid
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt < self.retry_limit:
                    logger.warning(f"Parse failed for chunk {chunk.chunk_id} "
                                   F"(attempt {attempt+1}): {e} - retrying")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Giving up on chunk {chunk.chunk_id}: {e}")
                    return []
    
    def _add_fact_to_graph(self, graph: KnowledgeHypergraph, fact: dict, chunk: Chunk)->None:
        """Add one extracted fact as a HyperEdge + update entity nodes."""
        raw_entities = fact["entities"]
        relation     = fact["relation"].strip().lower()
        sentence     = fact["sentence"]

        # normalize entities
        norm_entities = [self._normalize(e) for e in raw_entities]

        # skip if duplicate entities in the same fact
        if len(set(norm_entities)) < 2:
            return

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
            sentence  = sentence,
            chunk_id  = chunk.chunk_id,
            sample_id = chunk.sample_id,
            is_gold   = chunk.is_gold,
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

    def _verify_ollama(self) -> None:
        """Check Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            if not any(self.model in m for m in models):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama"
                    f"Run: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama ready - model: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama at "
                f"{self.ollama_url}. Is it running? -> Ollama serve"
            )
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama /api/chat endpoint with the extraction prompt
        Returns raw text response
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.0 # deterministic extraction
            }
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    
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
                    "relation": e.relation, "sentence": e.sentence,
                    "chunk_id": e.chunk_id, "sample_id": e.sample_id,
                    "is_gold": e.is_gold
                }
                for eid, e in graph.edges.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    from kg.data_loader import HotpotQALoader

    # Step 1: loading a tiny slice
    logger.info("1. Loading a slice of hotpot qa dataset")
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=3,
    )

    samples = loader.load()

    # Step 2: building a hypergraph
    gold_chunks = loader.get_gold_chunks(samples)
    logger.info(f"2. Running extraction on {len(gold_chunks)} gold chunks")

    builder = HypergraphBuilder(
        model=MODEL_NAME,
        ollama_url=OLLAMA_URL
    )
    graph = builder.build(gold_chunks)

    logger.info("== Hypergraph Summary ==")
    for k, v in graph.summary().items():
        print(f" {k}: {v}")
    
    # show edges for a known entity
    logger.info("== Sample edges ==")
    for edge in list(graph.edges.values())[:3]:
        print(f"\n  edge_id  : {edge.edge_id}")
        print(f"  entities : {edge.entities}")
        print(f"  relation : {edge.relation}")
        print(f"  sentence : {edge.sentence[:100]}")
        print(f"  is_gold  : {edge.is_gold}")