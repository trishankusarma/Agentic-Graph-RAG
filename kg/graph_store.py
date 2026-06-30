"""
kg/graph_store.py

Wraps KnowledgeHypergraph into a NetworkX graph for:
 - multi-hop path finding between entities
 - connectivity checks
 - broken hop detection (missing edges in a reasoning chain)

Hyperedge -> NetworkX projection:
    A hyperedge connecting [e1, e2, e3] via relation R is expanded into
    a clique of binary edges: (e1 <-> e2), (e1 <-> e3), (e2 <-> e3) each carrying
    the original edge_id, relation and is_gold as attributes

    This projection lets us use standard graph algorithms (BFS, shortest path) while
    preserving full hyperedge metadata for retrieval

Key operations:
    path_between(src, dst)          -> shortest entity path [e1, e2,... , en]
    edges_on_path(src, dst)         -> hyperedges that cover each hop
    check_broken_hops(gold_titles)  -> which hops in a reasoning chain are missing
    neighbors(entity)               -> all entities one hop away

Broken hop detection - two types:

    bridge:
        Q mentions one entity -> hops through intermediate -> reaches answer
        src = entity extracted from question
        dst = answer string
        path = src -> [intermediate entities] -> dst
        broken if: no path exists from src to dst
    
    comparison:
        Q mentions two entities, asks to compare a shared property
        src_1, src_2 = twp entities extracted from question
        dst = answer string (shared property value)
        path1 = src_1 -> dst
        path2 = src_2 -> dst
        broken if: either path_1 or path_2 is missing
"""

import logging
import requests
import json
import sys
from dataclasses import dataclass
from typing import Optional
import networkx as nx

from .data_loader import HotpotSample
from .hypergraph_builder import HyperEdge, KnowledgeHypergraph

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hyper-parameters
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_URL = "http://localhost:11434"
MAX_TOKENS = 2048

@dataclass
class HopResult:
    """One hop in a reasoning chain."""
    src:        str             # source entity (normalized)
    dst:        str             # destination entity (normalized)
    edge:       HyperEdge       # hyperedge covering this hop
    is_broken:  bool = False    # True of no edge exists for this hop

@dataclass
class PathResult:
    """Full path between two entities"""
    src:        str
    dst:        str
    hops:       list[HopResult]
    found:      bool            # False if no path exists at all

    def num_hops(self) -> int:
        return len(self.hops)

    def has_broken_hops(self) -> bool:
        return any(h.is_broken for h in self.hops)
    
    def broken_hops(self) -> list[HopResult]:
        return [h for h in self.hops if h.is_broken]
    
@dataclass
class Segment:
    """
    One stitched segment of the intended reasoning chain
    (e.g. src->gold_title[0], or gold_title[0] -> gold_title[1])
    """
    label:      str         # human-readable, eg "src -> Oberoi family"
    from_node:  str         # description of the from-endpoint (title or "src"/"dst")
    to_node:    str         # description of the to-endpoint
    path:       PathResult  # the actual shortest-path result from entity sets
    is_broken:  bool        # True if no path found between the endpoints
    is_fallback: bool = False # True if this segment used a skip-ahead fallback 

@dataclass
class Chain:
    """
    Full stitched chain for one src->dst reasoning path
    (1 chain for bridge, 1 per entity for comparison)
    """
    segments: list[Segment]

    def broken_count(self) -> int:
        """Count of segments that were genuinely broken (not healed by fallback)"""
        # a broken segment immediately followed by a successful falback = healed, will increment count to 1
        # else un-healed will return -1 won't consider
        count = 0
        i = 0

        while i < len(self.segments):
            seg = self.segments[i]
            if seg.is_broken:
                healed = (
                    i+1 < len(self.segments)
                    and self.segments[i+1].is_fallback
                    and not self.segments[i+1].is_broken
                )
                if not healed:
                    return sys.maxsize
                count += 1
                i += 2
            else:
                i += 1
        return count

    def is_clean(self) -> bool:
        return self.broken_count() == 0
    
    def broken_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_broken]
    
@dataclass
class BrokenHopReport:
    """Broken hop analysis for one QA sample"""
    sample_id:      str
    question:       str
    answer:         str
    hop_type:       str               # "bridge" | "comparison"
    gold_titles:    list[str]         # expected reasoning chain
    question_entities: list[str]      # extracted from question via LLM 
    chains:          list[Chain]      
    is_answerable:  bool              # False if any hop is broken

    def summary(self) -> dict:
        total_broken = sum(len(c.broken_segments()) for c in self.chains)
        return{
            "sample_id"      : self.sample_id,
            "hop_type"       : self.hop_type,
            "is_answerable"  : self.is_answerable,
            "question_entities" : self.question_entities,
            "answer"         : self.answer,
            "num_chains"       : len(self.chains),
            "total_broken_hops" : total_broken,
            "gold_titles"    : self.gold_titles
        }

ENTITY_EXTRACTION_PROMPT = """You are a named entity extractor.

Given a question, extract the key named entities that the question is ABOUT.
These are the entities that would be the starting points for graph traversal.

Rules:
- For bridge questions (who/what/where did X do?): return the main subject entity
- For comparison questions (did X and Y share property X?): return BOTH X and Y
- Return proper nouns only (people, places, orgs, works, dates)
- Return ONLY a JSON array of entity strings, nothing else

Examples:
Q: "Who directed the film starring Shirley Temple as Corliss Archer?"
→ ["Shirley Temple"]
 
Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
→ ["Scott Derrickson", "Ed Wood"]
 
Q: "What year was the director of Inception born?"
→ ["Inception"]
"""

class QuestionEntityExtractor:
    """Extracts named entities from questions via Ollama"""
    def __init__(
        self,
        model: str = MODEL_NAME,
        ollama_url: str = OLLAMA_URL,
        max_tokens: int = MAX_TOKENS
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_tokens = max_tokens
    
    def extract(self, question: str) -> list[str]:
        """Extract named entities from a question. Returns list of entity strings."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Q: {question}"}
            ],
            "stream": False,
            "options": {"num_predict": self.max_tokens, "temperature": 0.0},
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()

            # add immediately after it:
            # strip DeepSeek-R1 <think> block
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            # strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
            return []
        except Exception as e:
            logger.warning(f"Entity extraction failed for '{question[:60]}': {e}")
            return []

class GraphStore:
    """
    Projects a KnowledgeHyperGraph into NetworkX and provides graph traversal + broken hop detection

    Args:
        hypergraph: KnowledgeHypergraph from hypergraph_builder.py
        ollama_url: Ollama base URL for question qntity extraction
        model:      Ollama model for entity extraction
        directed: if True, use DiGraph (respects extraction order):
                  if False, use Graph (undirected, better for path finding)  
    """

    def __init__(
            self, 
            hypergraph: KnowledgeHypergraph, 
            model: str = MODEL_NAME,
            ollama_url: str = OLLAMA_URL,
            directed: bool = False
        ):
        self.hypergraph = hypergraph
        self.directed   = directed
        self.extractor  = QuestionEntityExtractor(model=model, ollama_url=ollama_url)
        self.G          = self._build_nx_graph()
        logger.info(
            f"GraphStore ready -- "
            f"{self.G.number_of_nodes()} nodes"
            f"{self.G.number_of_edges()} projected edges"
            f"(from {hypergraph.num_edges()} hyperedges)"
        )
    
    def _build_nx_graph(self) -> nx.Graph:
        """
        Project Hyperedges into a binary NetworkX graph.
        Each hyperedge [e1, e2, ..., en] -> clique of binary edges
        all carrying the hyperedge metadata as attributes
        """
        G = nx.DiGraph() if self.directed else nx.Graph()

        for edge in self.hypergraph.edges.values():
            entities = edge.entities    # already normalized

            # add all nodes with their label
            for eid in entities:
                if eid not in G:
                    node = self.hypergraph.nodes.get(eid)
                    G.add_node(eid, label=node.label if node else eid)
            
            # clique expansion: every pair in the hyperedge gets a binary edge
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    src, dst = entities[i], entities[j]

                    # If edge already exists, append this hyperedge_id to it
                    # (multiple hyperedges can connect the same pair)
                    if G.has_edge(src, dst):
                        G[src][dst]["edge_ids"].append(edge.edge_id)
                    else:
                        G.add_edge(
                            src, dst,
                            edge_ids = [edge.edge_id],
                            relation = edge.relation,
                            is_gold = edge.is_gold
                        )
        return G

    def path_between(self, src: str, dst: str) -> list[str]:
        """
        Shortest entity path from src to dst.
        Returns list of normalized entity strings or [] if unreachable.
        """
        if src not in self.G or dst not in self.G:
            return []
        
        try:
            return nx.shortest_path(self.G, src, dst)
        except nx.NetworkXNoPath:
            return []
    
    def edges_on_path(self, src: str, dst: str) -> PathResult:
        """
        Find stortest path and return full HopResult chain with
        the covering hyperedge for each hop.
        """
        src = self._normalize(src)
        dst = self._normalize(dst)
        path = self.path_between(src, dst)

        if not path:
            return PathResult(src=src, dst=dst, hops=[], found=False)
        
        hops = []
        for i in range(len(path) - 1):
            hop_src = path[i]
            hop_dst = path[i+1]
            edge = self._best_edge_for_hop(hop_src, hop_dst)
            hops.append(HopResult(
                src = hop_src,
                dst = hop_dst,
                edge = edge,
                is_broken = edge is None
            ))

        return PathResult(src=src, dst=dst, hops=hops, found=True)
    
    def _best_edge_for_hop(self, src, dst) -> HyperEdge | None:
        """
        Given a binary hop (src -> dst) in the projected graph,
        return the best covering hyperedge (prefer gold edge).
        """
        if not self.G.has_edge(src, dst):
            return None
        
        edge_ids = self.G[src][dst]["edge_ids"]
        edges = [
            self.hypergraph.edges[eid] for eid in edge_ids if eid in self.hypergraph.edges
        ]

        if not edges:
            return None
        
        # prefer gold edges, then highest arity (richest fact)
        edges.sort(key=lambda e: (e.is_gold, len(e.entities)), reverse=True)
        return edges[0]
    
    def check_broken_hops(self, sample: HotpotSample) -> BrokenHopReport:
        """
        Detect broken hops for a HotpotQA sample.

        Architecture:
            src -> always from LLM extraction on the question
            dst -> sample.answer
            gold title entities -> used to Validate the path passes through expected articles

            bridge: 
                1 src entity from question -> path to answer
                path should pass through gold_titles entities[i]( bridge article)
                broken if : no path from src -> dst
            
            comparison:
                2 src entities from question (one per entity being compared)
                each must independently reach dst (shared property)
                path_i should pass through gold_entities[j]
                broken if: either path missing
        """
        dst = self._normalize(sample.answer)
        question_entities = self.extractor.extract(sample.question)

        logger.info(
            f"[{sample.hop_type}] Q: {sample.question[:60]}... | "
            f"src_entities: {question_entities} | dst: {sample.answer}"
        )

        if sample.hop_type == "bridge":
            chains, is_answerable = self._check_bridge(
                question_entities, sample.gold_titles, dst
            )
        elif sample.hop_type == "comparison":
            chains, is_answerable = self._check_comparison(
                question_entities, sample.gold_titles, dst
            )
        else:
            logger.warning(f"Unknown hop_type '{sample.hop_type}' for {sample.sample_id}")
            chains, is_answerable = [], False
        
        return BrokenHopReport(
            sample_id = sample.sample_id,
            question = sample.question,
            answer = sample.answer,
            hop_type = sample.hop_type,
            gold_titles = sample.gold_titles,
            question_entities = question_entities,
            chains = chains,
            is_answerable = is_answerable
        )
    
    def _check_bridge(
            self,
            question_entities: list[str],
            gold_titles: list[str], 
            dst: str, 
    ) -> tuple[list[Chain], bool]:
        """
        Bridge: one chain stitched as src -> gold[0] -> gold[1] -> ... -> dst
        src is picked from question_entities (trying out each, taking first clean chain)
        """
        if not question_entities:
            empty_chain = Chain(
                segments=[Segment(
                    label="src -> dst", from_node="src", to_node="dst",
                    path=PathResult(src="", dst=dst, hops=[], found=False),
                    is_broken=True
                )]
            )
            return [empty_chain], False

        candidates = []
        for entity in question_entities:

            src = self._normalize(entity)
            chain = self._build_chain(src, gold_titles, dst)
            candidates.append(chain)
        
        # rank by broken count ascending - fewest broken wins
        best = min(candidates, key = lambda c: c.broken_count())
        is_answerable = best.is_clean()
        
        return [best], is_answerable
    
    def _build_chain(
            self,
            src : str, 
            gold_titles: list[str], 
            dst: str
    ) -> Chain:
        """
        Stitched the exact intened chain: src -> gold[0] -> gold[1] -> ... -> dst

        Each waypoint (src, gold[0], ..., gold[L-1], dst) is connected to the next via shortest path between their entity sets. If a direct segment
        (waypoint[i] -> waypoint[i+1]) is broken, try a skip-ahead fallback to waypoint[i+2] as a diagnostic — tells us if it's a single missing
        edge or a deeper disconnection.
        """
        # build the waypoint sequence: src, gold_title_0, gold_title_1, .... dst
        waypoints: list[tuple[str, set[str]]] = [("src", {src})]
        for title in gold_titles:
            waypoints.append((title, self._entities_for_title(title)))
        waypoints.append(("dst", {dst}))

        segments: list[Segment] = []
        i = 0
        while i < len(waypoints) -1:
            from_label, from_ents = waypoints[i]
            to_label, to_ents = waypoints[i+1]

            seg = self._segment_between(from_label, from_ents, to_label, to_ents)

            if seg.is_broken and i+2 < len(waypoints):
                # fallback: try skipping ahead one waypoint as a diagnostic
                skip_label, skip_ents = waypoints[i+2]
                fallback_seg = self._segment_between(
                    from_label, from_ents, skip_label, skip_ents
                )
                fallback_seg.is_fallback = True
                
                if not fallback_seg.is_broken:
                    # skip-ahead worked — the missing piece is specifically
                    # waypoint[i+1], record original broken segment AND
                    # note the successful skip, then continue from i+2
                    segments.append(seg)          # record the break
                    segments.append(fallback_seg) # record the skip that worked
                    i += 2
                    continue
            
            segments.append(seg)
            i += 1
        
        return Chain(segments=segments)
    
    def _segment_between(
            self,
            from_label: str, 
            from_ents: set[str], 
            to_label: str, 
            to_ents: set[str]
    ) -> Segment:
        """
        Find shortest path between two way point entity sets
        Tries all (from_entity, to_entity) pairs, take the shortest path found across all combinations
        """
        label = f"{from_label} -> {to_label}"

        if not from_ents or not to_ents:
            return Segment(
                label=label, from_node=from_label, to_node=to_label,
                path=PathResult(src="", dst="", hops=[], found=False),
                is_broken=True,
            )

        best: Optional[PathResult] = None
        for f in from_ents:
            for t in to_ents:
                if f == t:
                    continue
                result = self.edges_on_path(f, t)
                if result.found and not result.has_broken_hops():
                    if best is None or result.num_hops() < best.num_hops():
                        best = result
        
        if best is None:
            f0, t0 = next(iter(from_ents)), next(iter(to_ents))
            return Segment(
                label = label, from_node = from_label, to_node = to_label,
                path = PathResult(src=f0, dst = t0, hops=[], found=False),
                is_broken = True
            )
        
        return Segment(
            label=label, from_node=from_label, 
            to_node=to_label, path=best, is_broken=False
        )
    
    def _check_comparison(
            self,
            question_entities: list[str],
            gold_titles: list[str], 
            dst: str, 
    ) -> tuple[list[PathResult], bool]:
        """
        Comparison: one chain per (question_entity, gold_title) pair.
        chain_i = src_i -> matched_title_i -> dst
        """
        if not question_entities or not gold_titles:
            empty_chain = Chain(
                segments=[Segment(
                    label="src -> dst", from_node="src", to_node="dst",
                    path=PathResult(src="", dst=dst, hops=[], found=False),
                    is_broken=True
                )]
            )
            return [empty_chain], False
        
        pairs = self._pair_entities_to_titles(question_entities, gold_titles)

        # group chains by title
        chains_by_title: dict[str, list[Chain]] = {t: [] for t in gold_titles}
        unresolved: list[Chain] = []

        for entity, title in pairs:
            if title is None:
                # Cound not resolve which gold title this entity maps to
                unresolved.append(Chain(segments=[Segment(
                    label=f"{entity} -> ??? (unresolved title)",
                    from_node=entity, to_node="?",
                    path=PathResult(src=self._normalize(entity), dst=dst, hops=[], found=False),
                    is_broken=True,
                )]))
                continue

            src = self._normalize(entity)
            chain = self._build_chain(src, [title], dst)
            chains_by_title[title].append(chain)
        
        # pick the best chain per title; if a title has no candidate, it's broken
        chains = []
        for title in gold_titles:
            candidates = chains_by_title[title]
            if candidates:
                chains.append(min(candidates, key=lambda c: c.broken_count()))
            else:
                chains.append(Chain(segments=[Segment(
                    label=f"??? -> {title} (no entity matched)",
                    from_node="?", to_node=title,
                    path=PathResult(src="", dst=dst, hops=[], found=False),
                    is_broken=True,
                )]))
    
        chains.extend(unresolved) # keep for audit visibility, doesn't affect answerability

        is_answerable = all(
            c.broken_count() < sys.maxsize
            for c in chains[:len(gold_titles)]   # only the 2 title-bound chains matter
        )
        return chains, is_answerable

    def _pair_entities_to_titles(
        self,
        question_entities: list[str],
        gold_titles: list[str], 
    ) -> list[tuple[str, Optional[str]]]:
        """
        Pair each question entity to its corresponding gold title

        Tier 1 — exact match: normalized entity == normalized title
        Tier 2 — substring match: entity is a substring of title or vice versa
                 (e.g. "Jonathan Stark" <-> "Jonathan Stark (tennis)")
        Tier 3 — graph match: entity is a graph-neighbor of an entity
                 belonging to the title's article (e.g. "Hole" <-> "Courtney Love",
                 connected via a frontwoman_of / member_of hyperedge)

        Titles are consumed greedily — once matched, not reused for another entity.
        Returns list of (entity, title_or_None) in question_entities order.
        """
        remaining_titles = list(gold_titles)
        pairs: list[tuple[str, Optional[str]]] = []

        for entity in question_entities:
            entity_norm = self._normalize(entity)
            matched = None

            # tier 1 - exact match
            for title in remaining_titles:
                if self._normalize(title) == entity_norm:
                    matched = title
                    break
            
            # tier 2 - substring match
            if matched is None:
                for title in remaining_titles:
                    title_norm = self._normalize(title)
                    if entity_norm in title_norm or title_norm in entity_norm:
                        matched = title
                        break
            
            # tier 3 - graph neighbor match
            if matched is None:
                entity_neighbors = self.neighbors(entity_norm)
                for title in remaining_titles:
                    title_ents = self._entities_for_title(title)
                    if entity_neighbors.intersection(title_ents):
                        matched = title
                        break
            
            if matched is not None:
                remaining_titles.remove(matched)
            pairs.append((entity, matched))

        return pairs

    
    def neighbors(self, entity: str) -> set[str]:
        """All entities one hop away."""
        entity = self._normalize(entity)
        if entity not in self.G:
            return set()
        return set(self.G.neighbors(entity))

    def _entities_for_title(self, title: str) -> set[str]:
        """
        Returns all entity node ids whose chunks came from a given wikipedia article title
        chunk_id format:  {sample_id}_{title}_{i}
        """
        title_norm = self._normalize(title)
        result = set()
        for node in self.hypergraph.nodes.values():
            for chunk_id in node.chunks:
                if f"_{title_norm}_" in chunk_id.lower():
                    result.add(node.entity_id)
        return result
    
    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()
    
    def stats(self) -> dict:
        degrees = [d for _, d in self.G.degree()]
        return {
            "num_nodes":      self.G.number_of_nodes(),
            "num_edges":      self.G.number_of_edges(),
            "num_hyperedges": self.hypergraph.num_edges(),
            "avg_degree":     round(sum(degrees) / max(len(degrees), 1), 2),
            "num_components": nx.number_connected_components(self.G)
                              if not self.directed else "N/A (directed)",
            "density":        round(nx.density(self.G), 4),
        }
    
if __name__ == "__main__":
    from .data_loader import HotpotQALoader
    from .hypergraph_builder import HypergraphBuilder

    loader  = HotpotQALoader(split="validation", chunk_size=5, overlap=1, max_samples=5)
    samples = loader.load()
 
    builder = HypergraphBuilder(
        model=MODEL_NAME,
        cache_path="data/hypergraph_cache.json",
    )
    graph = builder.build(loader.get_gold_chunks(samples))
 
    store = GraphStore(graph, model=MODEL_NAME)

    print("\n=== Graph Stats ===")
    for k, v in store.stats().items():
        print(f"  {k}: {v}")
    
    print("\n=== Broken Hop Audit ===")
    for sample in samples:
        report = store.check_broken_hops(sample)
        print(f"\n  Q: {sample.question[:80]}")
        for k, v in report.summary().items():
            print(f"  {k:<22}: {v}")
 
        for i, chain in enumerate(report.chains):
            print(f"\n  chain_{i+1} ({'clean' if chain.is_clean() else 'BROKEN'}):")
            for seg in chain.segments:
                status = "⚠ BROKEN" if seg.is_broken else "✓"
                fb     = " (fallback skip-ahead)" if seg.is_fallback else ""
                hops_str = ""
                if seg.path.found:
                    rels = [h.edge.relation if h.edge else "???" for h in seg.path.hops]
                    hops_str = f"  via [{', '.join(rels)}]"
                print(f"    {seg.label}  {status}{fb}{hops_str}")