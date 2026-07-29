"""
miscellenous/diagnose_stuck.py

Of the 45 repairable samples, 18 had repair successfully insert new edges and
STILL reported broken. That is now the dominant failure bucket (bigger than
"no new facts found", which fell to 8), so it is where the remaining headroom
is — and it is NOT an extraction-recall problem, since the facts were found.

Hypothesis under test: the inserted edges name entities that already exist in
the graph under a different surface form, so the edge lands on an orphaned
duplicate node that the chain's waypoints never resolve to.

Evidence motivating it, from diagnose_self_loops.py on sample 5abc8d75:
    resolve_entity("Leo Varadkar")     -> {"leo varadkar"}
    entities_for_title("Leo Varadkar") -> {..., "leo eric varadkar", ...}
Two nodes, one person, disjoint sets, no path between them.
add_repaired_fact_to_graph only snaps on an UNAMBIGUOUS single match, so an
ambiguous or missed match silently creates yet another orphan.

Classification per sample:
    NEAR_MISS   an added entity shares tokens with a waypoint entity but is not
                equal -> canonicalization would fix it
    DISJOINT    added entities share nothing with the waypoints -> repair found
                a real but irrelevant fact
    OTHER_GAP   the repaired gap closed, but a different one is still open
                -> partial fix, needs more repair passes not canonicalization
"""

import json
from collections import Counter

from kg.data_loader import HotpotQALoader
from kg.detection import BrokenHopDetector, prefetch_entities
from kg.extractors import OpenAIBackend
from kg.graph import PathFinder
from kg.hypergraph_builder import HypergraphBuilder, add_repaired_fact_to_graph
from kg.text import normalize

VALIDATION_PATH = "data/repair_validation.json"
API_URL = "http://localhost:8001"
ANSWERABLE_MODES = {"connected", "healed", "predicate_ok", "answer_not_entity"}


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def resolve_name(pf, name):
    return pf.resolve_entity(name) or pf.entities_for_title(name) or set()


# --------------------------------------------------------------------------- #

with open(VALIDATION_PATH) as f:
    validation = json.load(f)

stuck = [
    r for r in validation["results"]
    if r["new_edges"] > 0 and r["mode_after"] not in ANSWERABLE_MODES
]
print(f"{len(stuck)} samples added edges but stayed broken\n")

wanted = {r["sample_id"] for r in stuck}
loader = HotpotQALoader(split="validation", chunk_size=5, overlap=1,
                        max_samples=200, cache_path="data/data_loader_cache.jsonl")
samples = {s.sample_id: s for s in loader.load() if s.sample_id in wanted}

extractor = OpenAIBackend(model="qwen3-14b", api_url=API_URL, pool_size=8)

# warm only these samples' chunks
extract_cache: dict = {}
HypergraphBuilder(extractor=extractor, max_workers=8, extract_cache=extract_cache).warmup(
    [c for s in samples.values() for c in s.chunks]
)

verdicts = Counter()

for record in stuck:
    sample = samples.get(record["sample_id"])
    if sample is None:
        continue

    graph = HypergraphBuilder(
        extractor=extractor, max_workers=4, extract_cache=extract_cache,
    ).build(sample.chunks)
    pf = PathFinder(graph)
    pf.index_samples([sample])
    entity_map = prefetch_entities(extractor, [sample])
    report_before = BrokenHopDetector(pf, extractor, entity_map=entity_map).check(sample)

    print("=" * 76)
    print(f"{record['sample_id']}  [{record['hop_type']}]")
    print(f"  Q: {record['question'][:72]}")
    print(f"  A: {record['answer']!r}")
    print(f"  {record['mode_before']} -> {record['mode_after']}")

    # --- replay the repair, capturing the ACTUAL entity ids inserted -------- #
    added_entities: set[str] = set()
    for attempt in record["attempts"]:
        if attempt["status"] != "found_facts":
            continue
        src_name, dst_name = attempt["src"], attempt["dst"]
        if isinstance(src_name, list):
            src_name = src_name[0]
        chunk_ids = set(attempt.get("chunks_read", []))
        for chunk in sample.chunks:
            if chunk.chunk_id not in chunk_ids:
                continue
            try:
                facts = extractor.extract_targeted(
                    chunk, src=str(src_name), dst=str(dst_name),
                    goal=f"connect {src_name} to {dst_name}",
                )
            except Exception:
                continue
            for fact in facts:
                edge, _snaps = add_repaired_fact_to_graph(graph, fact, chunk, pf)
                if edge is not None:
                    added_entities.update(edge.entities)
                    print(f"    ADDED: {edge.entities} --{edge.relation}-->")

    if not added_entities:
        print("    (replay added nothing — nondeterminism; skipping)")
        verdicts["replay_empty"] += 1
        print()
        continue

    # --- what is still broken, and what do its waypoints resolve to? ------- #
    pf_after = PathFinder(graph)
    pf_after.index_samples([sample])
    report_after = BrokenHopDetector(pf_after, extractor, entity_map=entity_map).check(sample)
    mode_after = report_after.summary()["failure_mode"]
    print(f"  after replay: {mode_after}")

    waypoint_entities: set[str] = set()
    still_broken = []
    for chain in report_after.reasoning_chains:
        if chain.terminal is None:
            continue
        for seg in chain.segments:
            if not seg.is_broken:
                continue
            still_broken.append((seg.from_node, seg.to_node))
            for label in (seg.from_node, seg.to_node):
                if label != "src":
                    waypoint_entities |= resolve_name(pf_after, label)
        if chain.terminal.is_broken_hop:
            still_broken.append(("<terminal>", sample.answer))
            waypoint_entities |= resolve_name(pf_after, sample.answer)

    print(f"  still-broken gaps: {still_broken}")

    # --- near-miss analysis ------------------------------------------------ #
    exact_overlap = added_entities & waypoint_entities
    near_misses = []
    for added in added_entities:
        for waypoint in waypoint_entities:
            if added == waypoint:
                continue
            score = token_jaccard(added, waypoint)
            if score > 0.3:
                near_misses.append((added, waypoint, score))
    near_misses.sort(key=lambda x: -x[2])

    if mode_after in ANSWERABLE_MODES:
        verdict = "FIXED_ON_REPLAY"
    elif near_misses:
        verdict = "NEAR_MISS"
        print("  NEAR MISSES (canonicalization would merge these):")
        for added, waypoint, score in near_misses[:5]:
            print(f"    {added!r}  ~  {waypoint!r}   (jaccard {score:.2f})")
    elif exact_overlap:
        verdict = "OTHER_GAP"
        print(f"  added entities DO match waypoints ({sorted(exact_overlap)[:3]}) "
              f"— a different gap is still open")
    else:
        verdict = "DISJOINT"
        print(f"  added entities share nothing with waypoints")
        print(f"    added:     {sorted(added_entities)[:6]}")
        print(f"    waypoints: {sorted(waypoint_entities)[:6]}")

    verdicts[verdict] += 1
    print(f"  VERDICT: {verdict}\n")

print("=" * 76)
print("SUMMARY")
print("=" * 76)
for verdict, count in verdicts.most_common():
    print(f"  {count:>3}  {verdict}")
print("""
NEAR_MISS  -> build-time entity canonicalization is the fix, and it is cheap
              relative to anything RL-shaped
OTHER_GAP  -> repair needs to run to a fixed point, not one pass per gap
DISJOINT   -> repair is finding real but irrelevant facts; the targeting prompt
              or the src/dst selection is at fault, not the graph
""")