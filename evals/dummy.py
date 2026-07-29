"""
miscellenous/check_repair_tool.py

Single-sample repair test that DIAGNOSES why a repair did or didn't take effect,
rather than only reporting the before/after failure_mode.

The previous version reported "2 new edges added, still broken_mid_chain", which is
ambiguous between several very different causes:
  - the new edges landed on duplicate/orphaned nodes the chain never looks at
  - the edges are correct but some other segment is still broken
  - the waypoint resolution itself is the problem, not extraction at all
This version prints enough to tell those apart in one run.
"""

from kg.data_loader import HotpotQALoader
from kg.detection import BrokenHopDetector, prefetch_entities
from kg.extractors import OpenAIBackend
from kg.graph import PathFinder
from kg.hypergraph_builder import HypergraphBuilder, add_repaired_fact_to_graph

TARGET_SAMPLE_ID = "5a8c7595554299585d9e36b6"
API_URL = "http://localhost:8001"


def resolve_name(path_finder, name):
    return (
        path_finder.resolve_entity(name)
        or path_finder.entities_for_title(name)
        or set()
    )


def gather_candidate_chunks(graph, sample, *entity_sets):
    chunk_ids = set()
    for entity_set in entity_sets:
        for eid in entity_set:
            node = graph.nodes.get(eid)
            if node:
                chunk_ids.update(node.chunks)
    return [c for c in sample.chunks if c.chunk_id in chunk_ids]


def show_waypoints(path_finder, chain_labels):
    """What node ids does each waypoint ACTUALLY resolve to? This is what the
    chain tests connectivity against — a repaired edge on any other node is
    invisible to it."""
    print("\nWaypoint resolution (what the chain actually checks):")
    for label in chain_labels:
        if label == "src":
            continue
        print(f"  {label!r} -> {sorted(resolve_name(path_finder, label))}")


# --------------------------------------------------------------------------- #

loader = HotpotQALoader(
    split="validation", chunk_size=5, overlap=1,
    cache_path="data/data_loader_cache.jsonl",
)
samples = loader.load()
sample = next((s for s in samples if s.sample_id == TARGET_SAMPLE_ID), None)
if sample is None:
    raise ValueError(f"{TARGET_SAMPLE_ID} not found.")

print(f"\nQuestion : {sample.question}")
print(f"Answer   : {sample.answer}")

extractor = OpenAIBackend(model="qwen3-14b", api_url=API_URL, pool_size=64)

extract_cache = {}
builder = HypergraphBuilder(
    extractor=extractor, max_workers=8, extract_cache=extract_cache,
)
graph = builder.build(sample.chunks)

path_finder = PathFinder(graph)
path_finder.index_samples([sample])
entity_map = prefetch_entities(extractor, [sample])
detector = BrokenHopDetector(path_finder, extractor, entity_map=entity_map)

report_before = detector.check(sample)
mode_before = report_before.summary()["failure_mode"]
print(f"\nCurrent failure mode : {mode_before}")

if mode_before not in ("broken_mid_chain", "broken_terminal"):
    print("\nThis sample is not currently a repair candidate.")
    raise SystemExit()

winning_chain = min(
    report_before.reasoning_chains, key=lambda c: c.num_unhealed_breaks()
)
broken_segments = [s for s in winning_chain.segments if s.is_broken]

print(f"\nBroken segments ({len(broken_segments)}):")
for seg in broken_segments:
    print(f"  {seg.from_node} -> {seg.to_node}")

show_waypoints(
    path_finder,
    [s.from_node for s in winning_chain.segments]
    + [s.to_node for s in winning_chain.segments],
)

# --------------------------------------------------------------------------- #
# Repair every broken segment
# --------------------------------------------------------------------------- #

total_new_edges = 0
total_snaps = 0

for seg in broken_segments:
    print("\n" + "=" * 78)
    print(f"Repairing segment: {seg.from_node} -> {seg.to_node}")

    src_candidates = (
        report_before.question_entities if seg.from_node == "src" else [seg.from_node]
    )
    dst_candidates = (
        report_before.question_entities if seg.to_node == "src" else [seg.to_node]
    )

    src_name, src_entities = None, set()
    for candidate in src_candidates:
        resolved = resolve_name(path_finder, candidate)
        if resolved:
            src_name, src_entities = candidate, resolved
            break

    dst_name, dst_entities = None, set()
    for candidate in dst_candidates:
        resolved = resolve_name(path_finder, candidate)
        if resolved:
            dst_name, dst_entities = candidate, resolved
            break

    if src_name is None or dst_name is None:
        print("  Could not resolve one endpoint — skipping.")
        continue

    print(f"  source      : {src_name!r} -> {sorted(src_entities)}")
    print(f"  destination : {dst_name!r} -> {sorted(dst_entities)}")

    candidate_chunks = gather_candidate_chunks(
        graph, sample, src_entities, dst_entities
    )
    print(f"  candidate chunks ({len(candidate_chunks)}):")
    for chunk in candidate_chunks:
        print(f"    {chunk.chunk_id} :: {chunk.title}")

    for chunk in candidate_chunks:
        facts = extractor.extract_targeted(
            chunk, src=src_name, dst=dst_name,
            goal=f"connect {src_name} to {dst_name}",
        )
        for fact in facts:
            edge, snaps = add_repaired_fact_to_graph(
                graph, fact, chunk, path_finder, verbose=False,
            )
            total_snaps += len(snaps)
            for raw, node_id in snaps:
                print(f"    SNAP: {raw!r} -> existing node {node_id!r}")
            if edge is not None:
                total_new_edges += 1
                print(
                    f"    NEW:  {edge.entities} --{edge.relation}--> ({chunk.title})"
                )

# --------------------------------------------------------------------------- #

print("\n" + "=" * 78)
print("AFTER REPAIR")
print("=" * 78)

path_finder = PathFinder(graph)
path_finder.index_samples([sample])
detector = BrokenHopDetector(path_finder, extractor, entity_map=entity_map)
report_after = detector.check(sample)
mode_after = report_after.summary()["failure_mode"]

print(f"\nNew edges added : {total_new_edges}")
print(f"Entity snaps    : {total_snaps}")
print(f"Failure before  : {mode_before}")
print(f"Failure after   : {mode_after}")

remaining = [
    seg
    for chain in report_after.reasoning_chains
    for seg in chain.segments
    if seg.is_broken
]
print(f"\nRemaining broken segments ({len(remaining)}):")
for seg in remaining:
    print(f"  {seg.from_node} -> {seg.to_node}")
if not remaining:
    print("  None")

# --------------------------------------------------------------------------- #
# Diagnosis — why did (or didn't) the repair take effect?
# --------------------------------------------------------------------------- #

print("\n" + "=" * 78)
print("DIAGNOSIS")
print("=" * 78)

if mode_after != mode_before and not remaining:
    print("\nRepair fixed the chain.")
elif total_new_edges == 0:
    print("\nNo new edges were added at all — the gap is NOT an extraction-recall")
    print("problem. Either the facts already exist (and the issue is resolution or")
    print("traversal), or the source text genuinely does not support the connection.")
else:
    print(f"\n{total_new_edges} edge(s) added but the chain is still broken.")
    if total_snaps:
        print(f"{total_snaps} entity snap(s) occurred, so duplicate-node creation was")
        print("actively prevented this run — if the chain is STILL broken, the")
        print("duplicate-node theory is not the (only) cause.")
    else:
        print("No entity snaps occurred, meaning every repaired entity either")
        print("already existed exactly, or resolved ambiguously/not at all.")

    # Show, for each still-broken segment, whether both endpoints' nodes are
    # actually adjacent in the graph — separates 'edge missing' from
    # 'edge exists but waypoint resolution points elsewhere'.
    for seg in remaining:
        if seg.from_node == "src":
            continue
        a = resolve_name(path_finder, seg.from_node)
        b = resolve_name(path_finder, seg.to_node)
        print(f"\n  segment {seg.from_node!r} -> {seg.to_node!r}")
        print(f"    from resolves to : {sorted(a)}")
        print(f"    to   resolves to : {sorted(b)}")
        adjacency = {
            x: sorted(graph.get_neighbors(x) & b) for x in a
        }
        for node_id, shared in adjacency.items():
            print(f"    neighbours of {node_id!r} inside target set: {shared or 'NONE'}")

print("\nAll node ids in graph:")
for nid in sorted(graph.nodes):
    print(f"  {nid}")