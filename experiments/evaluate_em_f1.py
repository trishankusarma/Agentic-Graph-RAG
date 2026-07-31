"""
miscellenous/evaluate_em_f1.py

Turns connectivity measurements into EM/F1 — numbers comparable to published
work (Graph-R1 reports 57.03 EM / 62.69 F1 on HotpotQA at Qwen2.5-7B).

Everything measured so far has been `answerable`: "a path exists in the graph
between the question entities and the answer string". That is a proxy. It does
not say a model would produce the right answer, and no paper reports it, so it
cannot be compared to anything. This script closes that gap.

Four conditions:

    closed_book   question only, no retrieval. The floor — how much does the
                  model already know? HotpotQA entities are Wikipedia-famous,
                  so this is NOT zero, and any graph result has to beat it to
                  mean anything.

    text_rag      top-k chunk sentences by semantic similarity. Standard RAG,
                  no graph. This is the baseline the graph must beat to justify
                  its existence — without it, a good graph number proves
                  nothing, because the same sentences fed directly might do as
                  well or better.

    graph         entity-anchored traversal: resolve question entities to
                  nodes, walk outward collecting hyperedges, rank by semantic
                  similarity to the question. Uses graph STRUCTURE, not just
                  embeddings, which is the whole premise.

    graph_repair  same, after running targeted repair on broken segments.

Also reports the correlation between `answerable` and EM. That validates (or
refutes) the diagnostic instrument this whole project is built on: if
answerable samples score far higher EM than unanswerable ones, connectivity is
a legitimate proxy and can be used to measure graph quality cheaply. If not,
the instrument has been measuring the wrong thing.
"""

import json
import logging
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from kg.data_loader import HotpotQALoader
from kg.detection import BrokenHopDetector, prefetch_entities
from kg.embeddings import EmbeddingBackend, SemanticIndex
from kg.extractors import OpenAIBackend
from kg.graph import PathFinder
from kg.hypergraph_builder import HypergraphBuilder, add_repaired_fact_to_graph

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

API_URL         = "http://localhost:8001"
MODEL           = "qwen3-14b"
MAX_SAMPLES     = 200
WARMUP_POOL     = 32
PER_SAMPLE_POOL = 4
GEN_WORKERS     = 8
TOP_K           = 8
MAX_HOPS        = 2
OVERRETRIEVE    = 20   # hyperedges fetched per top_k slot before deduping;
                       # 5 was not enough (6.6 avg facts vs text_rag's 8.0)
USE_SEMANTIC    = True # MUST match the resolver config the graph was measured
                       # under. With False, src_unresolved samples get an empty
                       # frontier in retrieve_graph AND are skipped entirely by
                       # repair_broken_segments, silently answering closed-book
                       # while labelled `graph`.
OUTPUT_PATH     = "data/em_f1_results.json"

CONDITIONS = ["closed_book", "text_rag", "graph", "graph_repair",
              "graph_guided", "graph_guided_repair"]

GUIDED_HOPS  = 5   # guided search stays narrow, so depth is affordable here in
                   # a way it is not for the unguided 2-hop collect-everything
GUIDED_BEAM  = 4   # edges followed per hop; BEAM * HOPS should exceed TOP_K so
                   # the budget can actually be filled
GUIDED_MIN_SIM = 0.15  # stop expanding a branch below this; untuned


# --------------------------------------------------------------------------- #
# HotpotQA official metrics — same normalization as Graph-R1's
# verl/utils/reward_score/qa_em_and_format.py, so numbers are comparable
# --------------------------------------------------------------------------- #

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM_PROMPT = """You answer multi-hop questions.

Rules:
- Answer with the SHORTEST possible span: a name, date, number, or short phrase.
- Do NOT explain, do NOT write a sentence, do NOT restate the question.
- For yes/no questions answer exactly "yes" or "no".
- If the facts given are insufficient, still give your single best guess.

Return only the answer."""


def build_answer_prompt(question: str, facts: list[str]) -> str:
    if not facts:
        return f"Question: {question}\n\nAnswer:"
    joined = "\n".join(f"- {f}" for f in facts)
    return f"Facts:\n{joined}\n\nQuestion: {question}\n\nAnswer:"


def retrieve_text(backend, sample, top_k: int) -> list[str]:
    """
    Standard RAG: top-k raw sentences by semantic similarity. No graph.

    Deduped by sentence TEXT, not by (chunk, index) id: the loader runs with
    overlap=1, so consecutive chunks share a sentence and the same string
    appears under two different ids. Without this, text_rag's k slots could be
    partly filled by repeats — inflating its apparent context while giving the
    model no extra information, and unfairly disadvantaging it against the
    graph conditions.
    """
    pairs = []
    seen_text = set()
    for c in sample.chunks:
        for i, s in enumerate(c.sentences):
            key = s.strip()
            if key and key not in seen_text:
                seen_text.add(key)
                pairs.append((f"{c.chunk_id}:{i}", s))
    if not pairs:
        return []
    index = SemanticIndex.build(backend, pairs)
    qvec = backend.encode([sample.question])[0]
    hits = index.search(qvec, top_k=top_k, min_similarity=0.0)
    lookup = dict(pairs)
    return [lookup[hit_id] for hit_id, _score in hits]


def retrieve_graph(backend, sample, graph, path_finder, question_entities, top_k: int) -> list[str]:
    """
    Entity-anchored retrieval: start at the question's entities, walk outward
    up to MAX_HOPS collecting incident hyperedges, then rank those by semantic
    similarity to the question.

    This is the condition that actually uses graph structure. Pure semantic
    top-k (text_rag above) would find sentences about the question's entities
    but has no mechanism to follow a bridge to a second article — which is the
    entire premise of multi-hop GraphRAG.
    """
    frontier = set()
    for entity in question_entities:
        frontier |= path_finder.resolve_entity(entity)
    if not frontier:
        return []

    visited = set(frontier)
    edge_ids = set()
    for _hop in range(MAX_HOPS):
        next_frontier = set()
        for node_id in frontier:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            edge_ids.update(node.edges)
            neighbours = graph.get_neighbors(node_id)
            next_frontier |= (neighbours - visited)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    pairs = [
        (eid, graph.edges[eid].sentence)
        for eid in edge_ids
        if eid in graph.edges
    ]
    if not pairs:
        return []

    index = SemanticIndex.build(backend, pairs)
    qvec = backend.encode([sample.question])[0]

    # Over-retrieve, THEN dedupe down to top_k.
    #
    # Several hyperedges routinely share one source sentence (a 5-sentence
    # chunk yielding ~8 facts per sentence means heavy overlap), so asking for
    # top_k hyperedges and deduping afterwards collapsed 8 hits into ~3 unique
    # sentences. Measured: graph conditions averaged 3.2 facts against
    # text_rag's 8.0 — a 2.5x context deficit that made the graph-vs-text
    # comparison meaningless, since the graph condition was answering from far
    # less evidence rather than worse evidence.
    #
    # text_rag has no such problem because it indexes sentences directly, so
    # its hits are unique by construction. The 5x over-retrieve is a heuristic
    # margin: if avg facts still comes out below top_k, raise it.
    hits = index.search(qvec, top_k=top_k * OVERRETRIEVE, min_similarity=0.0)
    lookup = dict(pairs)
    seen, facts = set(), []
    for hit_id, _score in hits:
        sentence = lookup[hit_id]
        if sentence not in seen:
            seen.add(sentence)
            facts.append(sentence)
            if len(facts) >= top_k:
                break
    return facts


def score_all_edges(backend, graph, question):
    """
    Embed every hyperedge sentence ONCE per sample and dot against the question.

    Guided traversal needs a similarity score at every hop; rebuilding an index
    per hop would mean 200 samples x 5 hops of redundant encoding. Scores are
    fixed given (graph, question), so compute them once.
    """
    items = list(graph.edges.items())
    if not items:
        return {}, {}
    sentences = [edge.sentence for _eid, edge in items]
    vectors = backend.encode(sentences)
    qvec = backend.encode([question])[0]
    sims = vectors @ qvec          # both L2-normalized -> cosine
    edge_score = {eid: float(sims[i]) for i, (eid, _e) in enumerate(items)}
    edge_sentence = {eid: edge.sentence for eid, edge in items}
    return edge_score, edge_sentence


def retrieve_graph_guided(backend, sample, graph, path_finder, question_entities,
                          top_k, max_hops=GUIDED_HOPS, beam=GUIDED_BEAM,
                          min_sim=GUIDED_MIN_SIM):
    """
    Beam search over the graph, steered by question similarity at every step.

    Contrast with retrieve_graph, which does two disconnected passes: expand
    blindly to MAX_HOPS collecting every incident edge, then rank the whole pile
    once. There, similarity never influences WHERE the walk goes — so a repaired
    edge cannot redirect anything, which is why graph and graph_repair came out
    identical to four decimal places.

    Here each hop picks the `beam` highest-scoring unseen edges from the current
    frontier, follows them, and expands from THEIR entities. A new edge can
    therefore change which branch is taken, giving repair a causal path to the
    retrieved context for the first time.

    The tradeoff is deliberate: at equal depth this returns a SUBSET of what the
    unguided version collects, so it should not win at max_hops=2. Its case is
    depth — an unguided 5-hop expansion would collect most of the graph and
    swamp the ranking, while a beam of 4 stays bounded.
    """
    frontier = set()
    for entity in question_entities:
        frontier |= path_finder.resolve_entity(entity)
    if not frontier:
        return []

    edge_score, edge_sentence = score_all_edges(backend, graph, sample.question)
    if not edge_score:
        return []

    visited_nodes = set(frontier)
    seen_edges = set()
    collected = []          # (score, edge_id)

    for _hop in range(max_hops):
        candidates = []
        for node_id in frontier:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            for eid in node.edges:
                if eid in seen_edges or eid not in edge_score:
                    continue
                candidates.append((edge_score[eid], eid))
        if not candidates:
            break

        candidates.sort(reverse=True)
        chosen = [(sc, eid) for sc, eid in candidates[:beam] if sc >= min_sim]
        if not chosen:
            break

        next_frontier = set()
        for score, eid in chosen:
            seen_edges.add(eid)
            collected.append((score, eid))
            for entity_id in graph.edges[eid].entities:
                if entity_id not in visited_nodes:
                    visited_nodes.add(entity_id)
                    next_frontier.add(entity_id)
        frontier = next_frontier
        if not frontier:
            break

    collected.sort(reverse=True)
    seen_text, facts = set(), []
    for _score, eid in collected:
        sentence = edge_sentence[eid]
        if sentence not in seen_text:
            seen_text.add(sentence)
            facts.append(sentence)
            if len(facts) >= top_k:
                break
    return facts


def repair_broken_segments(sample, graph, path_finder, extractor, report):
    """Targeted re-extraction on every broken gap. Mutates graph. Returns count."""
    added = 0
    for chain in report.reasoning_chains:
        if chain.terminal is None:
            continue
        gaps = []
        for seg in chain.segments:
            if seg.is_broken:
                src = (report.question_entities[0] if seg.from_node == "src"
                       else seg.from_node)
                gaps.append((src, seg.to_node))
        if chain.terminal.is_broken_hop:
            titles = list(sample.gold_sentences.keys())
            if titles:
                gaps.append((titles[-1], sample.answer))

        for src_name, dst_name in gaps:
            if not src_name or not dst_name:
                continue
            src_ents = path_finder.resolve_entity(src_name) or \
                       path_finder.entities_for_title(src_name)
            if not src_ents:
                continue
            chunk_ids = set()
            for eid in src_ents:
                node = graph.nodes.get(eid)
                if node:
                    chunk_ids.update(node.chunks)
            chunks = [c for c in sample.chunks if c.chunk_id in chunk_ids][:3]
            for chunk in chunks:
                try:
                    facts = extractor.extract_targeted(
                        chunk, src=src_name, dst=dst_name,
                        goal=f"connect {src_name} to {dst_name}",
                    )
                except Exception:
                    continue
                for fact in facts:
                    edge, _ = add_repaired_fact_to_graph(graph, fact, chunk, path_finder)
                    if edge is not None:
                        added += 1
    return added


# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()

    loader = HotpotQALoader(split="validation", chunk_size=5, overlap=1,
                            max_samples=MAX_SAMPLES,
                            cache_path="data/data_loader_cache.jsonl")
    samples = loader.load()
    print(f"{len(samples)} samples, {sum(len(s.chunks) for s in samples)} chunks")

    extractor = OpenAIBackend(model=MODEL, api_url=API_URL, pool_size=WARMUP_POOL)
    if not extractor.is_available():
        raise RuntimeError(f"not reachable at {API_URL}")

    backend = EmbeddingBackend()
    entity_map = prefetch_entities(extractor, samples, max_workers=WARMUP_POOL)

    extract_cache: dict = {}
    print("Warming extract cache...")
    HypergraphBuilder(extractor=extractor, max_workers=WARMUP_POOL,
                      extract_cache=extract_cache).warmup(
        [c for s in samples for c in s.chunks])
    print(f"  {len(extract_cache)} unique chunks  ({time.time()-t0:.0f}s)")

    # ---- build per-sample retrieval contexts ------------------------------- #

    print("\nBuilding graphs + retrieving...")
    per_sample = []
    for i, sample in enumerate(samples, 1):
        graph = HypergraphBuilder(extractor=extractor, max_workers=PER_SAMPLE_POOL,
                                  extract_cache=extract_cache).build(sample.chunks)
        pf = PathFinder(graph, use_semantic=USE_SEMANTIC)
        pf.index_samples([sample])
        detector = BrokenHopDetector(pf, extractor, entity_map=entity_map)
        report = detector.check(sample)
        q_entities = entity_map.get(sample.sample_id, [])

        contexts = {
            "closed_book": [],
            "text_rag":    retrieve_text(backend, sample, TOP_K),
            "graph":        retrieve_graph(backend, sample, graph, pf, q_entities, TOP_K),
            "graph_guided": retrieve_graph_guided(
                backend, sample, graph, pf, q_entities, TOP_K),
        }

        # repair mutates the graph, so this must come after the `graph` context
        n_added = 0
        if report.summary()["failure_mode"] in (
            "broken_mid_chain", "broken_terminal", "predicate_broken"
        ):
            n_added = repair_broken_segments(sample, graph, pf, extractor, report)
        pf_after = PathFinder(graph, use_semantic=USE_SEMANTIC)
        pf_after.index_samples([sample])
        contexts["graph_repair"] = retrieve_graph(
            backend, sample, graph, pf_after, q_entities, TOP_K)
        contexts["graph_guided_repair"] = retrieve_graph_guided(
            backend, sample, graph, pf_after, q_entities, TOP_K)

        per_sample.append({
            "sample": sample,
            "contexts": contexts,
            "failure_mode": report.summary()["failure_mode"],
            "answerable": report.is_answerable,
            "repair_edges": n_added,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(samples)}  ({time.time()-t0:.0f}s)")

    total_repair_edges = sum(r["repair_edges"] for r in per_sample)
    n_repaired = sum(1 for r in per_sample if r["repair_edges"] > 0)
    print(f"\nrepair: {total_repair_edges} edges added across {n_repaired} samples")
    if total_repair_edges == 0:
        print("  WARNING: repair added nothing — graph_repair will be IDENTICAL")
        print("  to graph, and its row is meaningless. Check that resolve_entity")
        print("  finds src_ents (needs use_semantic=True for src_unresolved cases).")

    # ---- generate + score --------------------------------------------------- #

    def answer_one(args):
        sample, facts = args
        raw = extractor._call_with_retry(
            ANSWER_SYSTEM_PROMPT,
            build_answer_prompt(sample.question, facts),
            label=f"ans:{sample.sample_id}", schema=None, max_tokens=64,
        )
        return (raw or "").strip().split("\n")[0].strip()

    results = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        print(f"\nGenerating [{condition}]...")
        payload = [(r["sample"], r["contexts"][condition]) for r in per_sample]
        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as pool:
            predictions = list(pool.map(answer_one, payload))
        for record, pred in zip(per_sample, predictions):
            gold = record["sample"].answer
            results[condition].append({
                "sample_id":    record["sample"].sample_id,
                "question":     record["sample"].question,
                "gold":         gold,
                "pred":         pred,
                "em":           em_score(pred, gold),
                "f1":           f1_score(pred, gold),
                "n_facts":      len(record["contexts"][condition]),
                "answerable":   record["answerable"],
                "failure_mode": record["failure_mode"],
            })

    # ---- report -------------------------------------------------------------- #

    print("\n" + "=" * 66)
    print(f"{'condition':<16}{'EM':>8}{'F1':>8}{'avg facts':>12}")
    print("=" * 66)
    fact_counts = {}
    for condition in CONDITIONS:
        rows = results[condition]
        em = 100 * sum(r["em"] for r in rows) / len(rows)
        f1 = 100 * sum(r["f1"] for r in rows) / len(rows)
        nf = sum(r["n_facts"] for r in rows) / len(rows)
        fact_counts[condition] = nf
        print(f"{condition:<16}{em:>8.2f}{f1:>8.2f}{nf:>12.1f}")

    # A graph-vs-text comparison is only meaningful at matched context budgets.
    # The previous run had graph at 3.2 facts vs text_rag at 8.0 — a 2.5x
    # deficit that confounded the entire result.
    retrieval_conditions = [c for c in CONDITIONS if c != "closed_book"]
    budgets = [fact_counts[c] for c in retrieval_conditions]
    if budgets and (max(budgets) - min(budgets)) > 1.0:
        print(f"\n  WARNING: fact budgets differ by more than 1.0 "
              f"({min(budgets):.1f} to {max(budgets):.1f}) — the graph-vs-text "
              f"comparison is confounded by context size, not quality.")
    else:
        print(f"\n  fact budgets matched within 1.0 — comparison is fair")

    # If graph conditions still fall short after over-retrieving, the cause may
    # be STRUCTURAL rather than a tuning problem: a sample whose 2-hop
    # neighbourhood spans fewer than top_k distinct source sentences cannot
    # reach the budget at any multiplier. That is itself a finding — it means
    # graph retrieval is inherently narrower than text retrieval here, not just
    # configured worse.
    for condition in ("graph", "graph_repair", "graph_guided", "graph_guided_repair"):
        short = [r for r in results[condition] if r["n_facts"] < TOP_K]
        if not short:
            continue
        avg_short = sum(r["n_facts"] for r in short) / len(short)
        em_short = 100 * sum(r["em"] for r in short) / len(short)
        full = [r for r in results[condition] if r["n_facts"] >= TOP_K]
        em_full = (100 * sum(r["em"] for r in full) / len(full)) if full else float("nan")
        print(f"\n  [{condition}] {len(short)}/{len(results[condition])} samples "
              f"below the {TOP_K}-fact budget (avg {avg_short:.1f})")
        print(f"      EM on those: {em_short:.2f}   vs {em_full:.2f} on full-budget samples")

    # Does `answerable` predict EM? This validates the whole diagnostic.
    print("\n" + "=" * 66)
    print("Does connectivity predict correctness?  (graph_repair condition)")
    print("=" * 66)
    rows = results["graph_repair"]
    for label, subset in (
        ("answerable",     [r for r in rows if r["answerable"]]),
        ("not answerable", [r for r in rows if not r["answerable"]]),
    ):
        if not subset:
            continue
        em = 100 * sum(r["em"] for r in subset) / len(subset)
        f1 = 100 * sum(r["f1"] for r in subset) / len(subset)
        print(f"  {label:<16} n={len(subset):>4}   EM {em:>6.2f}   F1 {f1:>6.2f}")

    print("\nBy failure mode:")
    modes = sorted({r["failure_mode"] for r in rows})
    for mode in modes:
        subset = [r for r in rows if r["failure_mode"] == mode]
        em = 100 * sum(r["em"] for r in subset) / len(subset)
        f1 = 100 * sum(r["f1"] for r in subset) / len(subset)
        print(f"  {mode:<20} n={len(subset):>4}   EM {em:>6.2f}   F1 {f1:>6.2f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "config": {"max_samples": MAX_SAMPLES, "top_k": TOP_K,
                       "max_hops": MAX_HOPS, "model": MODEL},
            "summary": {
                c: {
                    "em": 100 * sum(r["em"] for r in results[c]) / len(results[c]),
                    "f1": 100 * sum(r["f1"] for r in results[c]) / len(results[c]),
                } for c in CONDITIONS
            },
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()