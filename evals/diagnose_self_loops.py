"""
miscellenous/diagnose_self_loops.py

Four repairable samples had repair targets where src == dst:
    Scotch Collie -> Scotch Collie
    Leo Varadkar  -> Leo Varadkar
    Wendell Berry -> Wendell Berry
    John Waters   -> John Waters

These are cases where the ANSWER is literally the name of a gold article title.
The terminal should resolve trivially — the chain reached that title, and the
answer is that title's name, so nothing further needs traversing. Instead they
came back ENTITY_UNREACHED.

The only way that happens in TerminalResolver.resolve():

    answer_entities = resolve_entity(answer)          # set A
    from_ents       = entities_for_title(gold[-1])    # set B
    if trivial and (answer_entities & from_ents): -> ENTITY_REACHED

...is if A and B are DISJOINT for what is textually the same string, and no
clean path connects them either. This script prints A and B side by side so the
cause is visible rather than inferred.

Hypothesis worth checking in the output: entities_for_title's tier 1 returns
entities extracted FROM that article's chunks (provenance), which need not
include a node named after the article itself. resolve_entity's tier 1 is an
exact node-id match. Those two can legitimately miss each other.
"""

from kg.data_loader import HotpotQALoader
from kg.detection import BrokenHopDetector, prefetch_entities
from kg.extractors import OpenAIBackend
from kg.graph import PathFinder
from kg.hypergraph_builder import HypergraphBuilder
from kg.text import normalize

SELF_LOOP_SAMPLES = [
    "5a7a0e1e5542990783324e1a",   # Scotch Collie
    "5abc8d75554299700f9d7900",   # Leo Varadkar
    "5adf65555542992d7e9f9334",   # Wendell Berry
    "5ac3165c5542995ef918c10a",   # John Waters
]

loader = HotpotQALoader(split="validation", chunk_size=5, overlap=1,
                        max_samples=200, cache_path="data/data_loader_cache.jsonl")
samples = {s.sample_id: s for s in loader.load()}
extractor = OpenAIBackend(model="qwen3-14b", api_url="http://localhost:8001", pool_size=8)

for sid in SELF_LOOP_SAMPLES:
    sample = samples.get(sid)
    if sample is None:
        print(f"{sid}: not found\n")
        continue

    graph = HypergraphBuilder(
        extractor=extractor, max_workers=4, extract_cache={},
    ).build(sample.chunks)
    pf = PathFinder(graph)
    pf.index_samples([sample])

    answer = sample.answer
    gold_titles = list(sample.gold_sentences.keys())
    last_title = gold_titles[-1] if gold_titles else None

    A = pf.resolve_entity(answer)
    B = pf.entities_for_title(last_title) if last_title else set()

    print("=" * 74)
    print(f"{sid}")
    print(f"  question    : {sample.question[:70]}")
    print(f"  answer      : {answer!r}")
    print(f"  gold titles : {gold_titles}")
    print(f"  last title  : {last_title!r}")
    print(f"\n  resolve_entity({answer!r})        -> {sorted(A)}")
    print(f"  entities_for_title({last_title!r}) -> {sorted(B)[:12]}"
          f"{' ...' if len(B) > 12 else ''}  (n={len(B)})")
    print(f"\n  A & B intersection : {sorted(A & B) or 'EMPTY  <-- this is the bug'}")
    print(f"  answer == a gold title (by name)? "
          f"{any(normalize(t) == normalize(answer) for t in gold_titles)}")

    # Is the answer node even present, and what is it connected to?
    for node_id in sorted(A):
        neighbours = graph.get_neighbors(node_id)
        overlap = neighbours & B
        print(f"  node {node_id!r}: {len(neighbours)} neighbours, "
              f"{len(overlap)} inside the title set")

    detector = BrokenHopDetector(pf, extractor,
                                 entity_map=prefetch_entities(extractor, [sample]))
    report = detector.check(sample)
    print(f"\n  failure_mode : {report.summary()['failure_mode']}")
    print(f"  terminals    : {report.summary()['chain_terminals']}")
    print()