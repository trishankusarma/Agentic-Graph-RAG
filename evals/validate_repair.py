"""
miscellenous/validate_repair.py

THE decisive experiment: of the samples the detector calls `repairable`, how many
actually flip to answerable once repair runs?

This gates the whole repair-as-an-action direction. Read the result honestly:
    >50%  repair thesis is strong
    20-50% real but modest — needs a tight framing
    <15%  the remaining broken hops are NOT extraction-recall failures, and the
          framing needs to change before more code gets written

Why the per-sample max_workers is deliberately tiny
---------------------------------------------------
HypergraphBuilder.build() does `min(max_workers, len(chunks))`. A 14-chunk sample
with max_workers=14 submits every chunk simultaneously with nothing queued behind
it, which is exactly the regime where vLLM returns well-formed garbage (measured:
164 nodes at 12 workers vs 23 nodes at 14 on the same sample). The corpus warmup
does not have this problem — 2840 chunks means the opening burst is ~1% of the
work — but per-sample builds do. After warmup every per-sample build is 100% cache
hits anyway, so a low worker count costs nothing.

Outputs data/repair_validation.json for post-hoc analysis.
"""

import json
import logging
import time
from collections import Counter

from kg.data_loader import HotpotQALoader
from kg.detection import BrokenHopDetector, prefetch_entities
from kg.extractors import OpenAIBackend
from kg.graph import PathFinder
from kg.hypergraph_builder import (
    HypergraphBuilder,
    add_repaired_fact_to_graph,
)
from kg.text import normalize

logging.basicConfig(
    level=logging.WARNING,          # per-sample INFO spam would bury the results
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_URL          = "http://localhost:8001"
MODEL            = "qwen3-14b"
MAX_SAMPLES      = 200
WARMUP_POOL      = 32     # safe: opening burst is ~1% of 2840 chunks
PER_SAMPLE_POOL  = 4      # see module docstring — must stay well under len(chunks)
MAX_CHUNKS_PER_REPAIR = 3
OUTPUT_PATH      = "data/repair_validation.json"

REPAIRABLE_MODES = {"broken_mid_chain", "broken_terminal", "predicate_broken"}
ANSWERABLE_MODES = {"connected", "healed", "predicate_ok", "answer_not_entity"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def resolve_name(path_finder, name):
    return (
        path_finder.resolve_entity(name)
        or path_finder.entities_for_title(name)
        or set()
    )


def candidate_chunks_for(graph, sample, *entity_sets):
    """Source passages attached to any of the given entities."""
    chunk_ids = set()
    for entity_set in entity_sets:
        for eid in entity_set:
            node = graph.nodes.get(eid)
            if node:
                chunk_ids.update(node.chunks)
    return [c for c in sample.chunks if c.chunk_id in chunk_ids]


def build_graph_and_detect(sample, extractor, extract_cache, entity_map):
    """Fresh graph + detector for one sample. All cache hits after warmup."""
    builder = HypergraphBuilder(
        extractor=extractor,
        max_workers=PER_SAMPLE_POOL,
        extract_cache=extract_cache,
    )
    graph = builder.build(sample.chunks)

    path_finder = PathFinder(graph, use_semantic=True)
    path_finder.index_samples([sample])
    detector = BrokenHopDetector(path_finder, extractor, entity_map=entity_map)
    return graph, path_finder, detector


def repair_targets(report, sample):
    """
    (kind, src_options, dst_options) triples, derived from what is ACTUALLY broken.

    Two distinct kinds of gap need different targets:
      - broken segment  : connect two waypoints on the chain
      - broken terminal : connect the last gold title to the answer itself

    FIX 1 — iterate EVERY primary chain, not just the best one.
    The previous version picked min(num_unhealed_breaks), i.e. the chain that is
    doing BEST. But a sample's failure_mode comes from worst_mode() across all
    chains, so for a comparison sample the best chain can be perfectly clean
    while its sibling is broken — and repair then targeted nothing at all.
    That produced 4 samples with `attempts: []` in the first validation run:
    counted as failures without ever having been attempted. The true denominator
    was 43, not 47.

    FIX 2 — skip self-loop targets (src == dst after normalization).
    Four samples produced targets like "Leo Varadkar" -> "Leo Varadkar", which
    arise when the ANSWER is literally the name of a gold article title. No
    extraction can connect an entity to itself, so these calls were pure waste.
    They are skipped here, but note that skipping them does NOT make those
    samples pass — the underlying issue is that the terminal reports
    ENTITY_UNREACHED when it should report ENTITY_REACHED (trivial). See
    diagnose_self_loops.py; the fix belongs in TerminalResolver, not here.
    """
    targets = []
    seen = set()

    for chain in report.reasoning_chains:
        if chain.terminal is None:      # audit row, not a verdict row
            continue

        for seg in chain.segments:
            if not seg.is_broken:
                continue
            # ChainBuilder labels the first waypoint literally "src" rather than
            # storing the resolved entity string — a real gap in the data model.
            src_options = (
                report.question_entities if seg.from_node == "src" else [seg.from_node]
            )
            dst_options = (
                report.question_entities if seg.to_node == "src" else [seg.to_node]
            )
            key = ("segment", tuple(src_options), tuple(dst_options))
            if key not in seen:
                seen.add(key)
                targets.append(("segment", src_options, dst_options))

        terminal = chain.terminal
        if terminal.is_broken_hop:
            gold_titles = list(sample.gold_sentences.keys())
            if gold_titles:
                key = ("terminal", gold_titles[-1], sample.answer)
                if key not in seen:
                    seen.add(key)
                    targets.append(("terminal", [gold_titles[-1]], [sample.answer]))

    # drop self-loops — nothing to extract, and they polluted the first run
    filtered = []
    for kind, src_options, dst_options in targets:
        if (
            len(src_options) == 1 and len(dst_options) == 1
            and normalize(src_options[0]) == normalize(dst_options[0])
        ):
            continue
        filtered.append((kind, src_options, dst_options))

    return filtered


def attempt_repair(sample, graph, path_finder, extractor, report):
    """
    Run targeted re-extraction for every broken gap. Mutates `graph`.
    Returns (n_new_edges, n_snaps, list_of_attempt_records).
    """
    new_edges = 0
    snaps = 0
    attempts = []

    for kind, src_options, dst_options in repair_targets(report, sample):
        src_name = next(
            (c for c in src_options if resolve_name(path_finder, c)), None
        )
        # dst is NOT required to resolve: for a terminal gap the answer is
        # frequently not in the graph at all, which is the whole point.
        dst_name = dst_options[0] if dst_options else None

        if src_name is None or dst_name is None:
            attempts.append({
                "kind": kind, "src": src_options, "dst": dst_options,
                "status": "unresolvable_endpoint", "new_edges": 0,
            })
            continue

        src_ents = resolve_name(path_finder, src_name)
        dst_ents = resolve_name(path_finder, dst_name)
        chunks = candidate_chunks_for(graph, sample, src_ents, dst_ents)
        chunks = chunks[:MAX_CHUNKS_PER_REPAIR]

        if not chunks:
            attempts.append({
                "kind": kind, "src": src_name, "dst": dst_name,
                "status": "no_source_chunks", "new_edges": 0,
            })
            continue

        found_here = 0
        for chunk in chunks:
            try:
                facts = extractor.extract_targeted(
                    chunk, src=src_name, dst=dst_name,
                    goal=f"connect {src_name} to {dst_name}",
                )
            except Exception as e:
                logger.warning(f"repair extraction failed: {e}")
                continue

            for fact in facts:
                edge, edge_snaps = add_repaired_fact_to_graph(
                    graph, fact, chunk, path_finder,
                )
                snaps += len(edge_snaps)
                if edge is not None:
                    new_edges += 1
                    found_here += 1

        attempts.append({
            "kind": kind, "src": src_name, "dst": dst_name,
            "status": "found_facts" if found_here else "nothing_new",
            "new_edges": found_here,
            "chunks_read": [c.chunk_id for c in chunks],
        })

    return new_edges, snaps, attempts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()

    loader = HotpotQALoader(
        split="validation", chunk_size=5, overlap=1,
        max_samples=MAX_SAMPLES, cache_path="data/data_loader_cache.jsonl",
    )
    samples = loader.load()
    print(f"Loaded {len(samples)} samples, {sum(len(s.chunks) for s in samples)} chunks")

    extractor = OpenAIBackend(model=MODEL, api_url=API_URL, pool_size=WARMUP_POOL)
    if not extractor.is_available():
        raise RuntimeError(f"Extractor not reachable at {API_URL}")

    entity_map = prefetch_entities(extractor, samples, max_workers=WARMUP_POOL)

    # One flat concurrent pass over every chunk, before any per-sample work.
    extract_cache: dict = {}
    print(f"\nWarming extract cache ({WARMUP_POOL} workers)...")
    HypergraphBuilder(
        extractor=extractor, max_workers=WARMUP_POOL, extract_cache=extract_cache,
    ).warmup([c for s in samples for c in s.chunks])
    print(f"Cache: {len(extract_cache)} unique chunks  ({time.time()-t0:.0f}s)")

    # ---- pass 1: find the repairable population ---------------------------- #

    print("\nPass 1 — detecting...")
    repairable = []
    all_modes = Counter()

    for i, sample in enumerate(samples, 1):
        graph, path_finder, detector = build_graph_and_detect(
            sample, extractor, extract_cache, entity_map,
        )
        report = detector.check(sample)
        mode = report.summary()["failure_mode"]
        all_modes[mode] += 1
        if mode in REPAIRABLE_MODES:
            repairable.append(sample)
        if i % 50 == 0:
            print(f"  {i}/{len(samples)}")

    print(f"\nfailure_modes: {dict(all_modes)}")
    print(f"repairable: {len(repairable)}/{len(samples)}")

    # ---- pass 2: repair each one ------------------------------------------- #

    print(f"\nPass 2 — repairing {len(repairable)} samples...")
    results = []
    transitions = Counter()

    for i, sample in enumerate(repairable, 1):
        graph, path_finder, detector = build_graph_and_detect(
            sample, extractor, extract_cache, entity_map,
        )
        report_before = detector.check(sample)
        mode_before = report_before.summary()["failure_mode"]

        new_edges, snaps, attempts = attempt_repair(
            sample, graph, path_finder, extractor, report_before,
        )

        # Re-detect against the MUTATED graph — PathFinder caches resolution,
        # so it must be rebuilt, not reused.
        path_finder_after = PathFinder(graph, use_semantic=True)
        path_finder_after.index_samples([sample])
        detector_after = BrokenHopDetector(
            path_finder_after, extractor, entity_map=entity_map,
        )
        report_after = detector_after.check(sample)
        mode_after = report_after.summary()["failure_mode"]

        transitions[f"{mode_before} -> {mode_after}"] += 1
        results.append({
            "sample_id":    sample.sample_id,
            "question":     sample.question,
            "answer":       sample.answer,
            "hop_type":     sample.hop_type,
            "mode_before":  mode_before,
            "mode_after":   mode_after,
            "answerable_before": report_before.is_answerable,
            "answerable_after":  report_after.is_answerable,
            "new_edges":    new_edges,
            "entity_snaps": snaps,
            "attempts":     attempts,
        })

        if i % 10 == 0:
            print(f"  {i}/{len(repairable)}")

    # ---- report ------------------------------------------------------------ #

    flipped   = [r for r in results if r["mode_after"] in ANSWERABLE_MODES]
    unchanged = [r for r in results if r["mode_after"] == r["mode_before"]]
    no_facts  = [r for r in results if r["new_edges"] == 0]

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"repairable samples attempted : {len(results)}")
    print(f"flipped to answerable        : {len(flipped)}  "
          f"({len(flipped)/max(len(results),1):.0%})")
    print(f"unchanged failure_mode       : {len(unchanged)}")
    print(f"repair found NO new facts    : {len(no_facts)}")

    print("\nTransitions:")
    for transition, count in transitions.most_common():
        print(f"  {count:>3}  {transition}")

    # Why did the non-flips fail? Separates "no facts found" (extraction can't
    # help) from "facts added but still broken" (resolution/traversal issue).
    stuck = [r for r in results if r["mode_after"] not in ANSWERABLE_MODES]
    if stuck:
        print(f"\nOf {len(stuck)} still-broken:")
        reasons = Counter()
        for r in stuck:
            if r["new_edges"] == 0:
                statuses = {a["status"] for a in r["attempts"]}
                reasons[f"no new facts ({'/'.join(sorted(statuses))})"] += 1
            else:
                reasons[f"added {r['new_edges']} edges, still broken"] += 1
        for reason, count in reasons.most_common():
            print(f"  {count:>3}  {reason}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "config": {
                "max_samples": MAX_SAMPLES, "warmup_pool": WARMUP_POOL,
                "per_sample_pool": PER_SAMPLE_POOL,
                "max_chunks_per_repair": MAX_CHUNKS_PER_REPAIR,
            },
            "failure_modes_all": dict(all_modes),
            "n_repairable": len(results),
            "n_flipped": len(flipped),
            "transitions": dict(transitions),
            "results": results,
        }, f, indent=2)

    print(f"\nWrote {OUTPUT_PATH}   ({time.time()-t0:.0f}s total)")
    print(f"extractor stats: {extractor.stats()}")


if __name__ == "__main__":
    main()