"""
miscellenous/label_repair_facts.py

Measures the FABRICATION RATE of repair: of the facts targeted repair inserts
into the graph, what fraction are not actually entailed by the sentence they
claim to come from?

Why this number matters
-----------------------
kg/extractors/prompts.py REPAIR_SYSTEM_PROMPT tells the model:

    "A reasoning system ... has hit a dead end: it needs to get from one entity
     to another, but no edge connecting them exists ... find the connection
     that was missed."

That presupposes a connection exists. When it does not, the model is being
asked to find something that is not there — the standard setup for
confabulation. validation.py catches malformed facts (bad arity, bad sentence
index, placeholder relations) but a fluent, well-formed, UNSUPPORTED fact
passes every check and is written into the graph the agent then reasons over.

Nobody reports this number. GraphRAG papers report extraction quality on facts
that ARE present; none report fabrication rate on facts that are not.

Two phases
----------
    python -m miscellenous.label_repair_facts collect
        Replays repair on the repairable samples, capturing every inserted
        fact alongside its source sentence and the (src, dst) the repair was
        targeting. Writes data/repair_facts_to_label.jsonl.

    python -m miscellenous.label_repair_facts label
        Interactive, resumable. For each fact: is it entailed by the sentence?
        y / n / u / q. Writes data/repair_fact_labels.json as it goes.

    python -m miscellenous.label_repair_facts report
        Fabrication rate, broken down by segment vs terminal repair.

Note on replay: extract_targeted runs at REPAIR_TEMPERATURE (0.3), so this is
a fresh sample of repair outputs, not a replay of the exact facts from an
earlier run. That is fine for estimating a rate, and it captures the source
sentence properly, which the earlier runs did not store.
"""

import json
import os
import sys

COLLECT_PATH = "data/repair_facts_to_label.jsonl"
LABELS_PATH  = "data/repair_fact_labels.json"
API_URL      = "http://localhost:8001"
MODEL        = "qwen3-14b"


def verbalize(entities, relation) -> str:
    rel = relation.replace("_", " ")
    if len(entities) == 2:
        return f"{entities[0]}  —[{rel}]→  {entities[1]}"
    return f"{rel}({', '.join(entities)})"


# --------------------------------------------------------------------------- #
# PHASE 1 — collect
# --------------------------------------------------------------------------- #

def collect():
    from kg.data_loader import HotpotQALoader
    from kg.detection import BrokenHopDetector, prefetch_entities
    from kg.extractors import OpenAIBackend
    from kg.graph import PathFinder
    from kg.hypergraph_builder import HypergraphBuilder, add_repaired_fact_to_graph

    loader = HotpotQALoader(split="validation", chunk_size=5, overlap=1,
                            max_samples=200,
                            cache_path="data/data_loader_cache.jsonl")
    samples = loader.load()
    extractor = OpenAIBackend(model=MODEL, api_url=API_URL, pool_size=32)
    if not extractor.is_available():
        raise RuntimeError(f"not reachable at {API_URL}")

    entity_map = prefetch_entities(extractor, samples, max_workers=32)
    extract_cache: dict = {}
    print("Warming extract cache...")
    HypergraphBuilder(extractor=extractor, max_workers=32,
                      extract_cache=extract_cache).warmup(
        [c for s in samples for c in s.chunks])

    records = []
    repairable_modes = {"broken_mid_chain", "broken_terminal", "predicate_broken"}

    print("\nCollecting repair facts...")
    for i, sample in enumerate(samples, 1):
        graph = HypergraphBuilder(extractor=extractor, max_workers=4,
                                  extract_cache=extract_cache).build(sample.chunks)
        pf = PathFinder(graph)
        pf.index_samples([sample])
        report = BrokenHopDetector(pf, extractor, entity_map=entity_map).check(sample)
        if report.summary()["failure_mode"] not in repairable_modes:
            continue

        chunks_by_id = {c.chunk_id: c for c in sample.chunks}

        # rebuild the same gap list repair would target
        gaps = []
        for chain in report.reasoning_chains:
            if chain.terminal is None:
                continue
            for seg in chain.segments:
                if seg.is_broken:
                    src = (report.question_entities[0] if seg.from_node == "src"
                           else seg.from_node)
                    gaps.append(("segment", src, seg.to_node))
            if chain.terminal.is_broken_hop:
                titles = list(sample.gold_sentences.keys())
                if titles:
                    gaps.append(("terminal", titles[-1], sample.answer))

        for kind, src_name, dst_name in gaps:
            if not src_name or not dst_name:
                continue
            src_ents = (pf.resolve_entity(src_name)
                        or pf.entities_for_title(src_name))
            if not src_ents:
                continue
            chunk_ids = set()
            for eid in src_ents:
                node = graph.nodes.get(eid)
                if node:
                    chunk_ids.update(node.chunks)
            target_chunks = [chunks_by_id[c] for c in sorted(chunk_ids)
                             if c in chunks_by_id][:3]

            for chunk in target_chunks:
                try:
                    facts = extractor.extract_targeted(
                        chunk, src=src_name, dst=dst_name,
                        goal=f"connect {src_name} to {dst_name}",
                    )
                except Exception:
                    continue
                for fact in facts:
                    edge, _snaps = add_repaired_fact_to_graph(
                        graph, fact, chunk, pf)
                    if edge is None:
                        continue      # deduped or degenerate — never entered the graph
                    records.append({
                        "id":              f"{sample.sample_id}:{len(records)}",
                        "sample_id":       sample.sample_id,
                        "question":        sample.question,
                        "gold_answer":     sample.answer,
                        "repair_kind":     kind,
                        "target_src":      src_name,
                        "target_dst":      dst_name,
                        "fact_entities":   edge.entities,
                        "fact_relation":   edge.relation,
                        "source_sentence": chunk.sentences[fact["sentence_index"]],
                        "chunk_title":     chunk.title,
                        "chunk_id":        chunk.chunk_id,
                    })
        if i % 25 == 0:
            print(f"  {i}/{len(samples)}   {len(records)} facts so far")

    os.makedirs("data", exist_ok=True)
    with open(COLLECT_PATH, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    n_terminal = sum(1 for r in records if r["repair_kind"] == "terminal")
    print(f"\nWrote {len(records)} facts to {COLLECT_PATH}")
    print(f"  segment repairs : {len(records) - n_terminal}")
    print(f"  terminal repairs: {n_terminal}")
    print(f"\nNow run:  python -m miscellenous.label_repair_facts label")


# --------------------------------------------------------------------------- #
# PHASE 2 — label
# --------------------------------------------------------------------------- #

def label():
    if not os.path.exists(COLLECT_PATH):
        raise SystemExit(f"{COLLECT_PATH} not found — run `collect` first.")

    with open(COLLECT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            labels = json.load(f)
        print(f"Resuming — {len(labels)} already labelled\n")

    todo = [r for r in records if r["id"] not in labels]
    if not todo:
        print("All labelled. Run `report`.")
        return

    print("=" * 74)
    print("Is the FACT entailed by the SENTENCE?")
    print("  y = yes, the sentence supports it")
    print("  n = NO — fabricated or unsupported")
    print("  u = unclear / partially supported")
    print("  q = quit and save")
    print("=" * 74)

    for i, record in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}]  ({record['repair_kind']} repair)")
        print(f"  repair was asked to connect: "
              f"{record['target_src']!r} → {record['target_dst']!r}")
        print(f"\n  SENTENCE ({record['chunk_title']}):")
        print(f"    {record['source_sentence'].strip()}")
        print(f"\n  FACT:")
        print(f"    {verbalize(record['fact_entities'], record['fact_relation'])}")

        while True:
            choice = input("\n  entailed? [y/n/u/q] ").strip().lower()
            if choice in ("y", "n", "u", "q"):
                break
            print("  please enter y, n, u, or q")

        if choice == "q":
            break
        labels[record["id"]] = choice
        with open(LABELS_PATH, "w") as f:      # save every item — crash-safe
            json.dump(labels, f, indent=2)

    print(f"\nSaved {len(labels)} labels to {LABELS_PATH}")
    print(f"Run:  python -m miscellenous.label_repair_facts report")


# --------------------------------------------------------------------------- #
# PHASE 2a — LLM judge
# --------------------------------------------------------------------------- #

JUDGE_URL   = "http://localhost:8002"
JUDGE_MODEL = "llama-3.3-70b"
JUDGE_LABELS_PATH = "data/repair_fact_judge_labels.json"

JUDGE_SYSTEM_PROMPT = """You verify whether a fact was actually stated in a sentence.

You will be given a SENTENCE and a FACT extracted from it. Decide whether the
sentence ENTAILS the fact.

Critical rule: judge ONLY against the sentence. A fact can be true in the real
world and still NOT entailed, if the sentence does not state it. Those count as
NOT entailed. Do not use outside knowledge.

Examples:

SENTENCE: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace.
FACT: A Kiss for Corliss --[directed by]-> Richard Wallace
ANSWER: yes

SENTENCE: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace.
FACT: Richard Wallace --[born in]-> 1949
ANSWER: no
(1949 is the film's release year, not his birth year — the sentence does not say this)

SENTENCE: Shirley Temple Black was an American actress, singer, dancer and diplomat.
FACT: Shirley Temple Black --[served as]-> Chief of Protocol
ANSWER: no
(true in reality, but this sentence does not state it)

SENTENCE: It stars Shirley Temple in her final starring role.
FACT: Shirley Temple --[stars in]-> the film
ANSWER: yes

Answer with exactly one word: yes, no, or unclear."""


def judge():
    """LLM-judge every collected fact. Uses a DIFFERENT model family from the
    extractor: Qwen3-14B judging its own output is self-judging bias, and a
    reviewer will reject it. Serve a second model on JUDGE_URL, e.g.

        vllm serve /home/models/Llama-3.3-70B-Instruct \
            --served-model-name llama-3.3-70b --port 8002 \
            --tensor-parallel-size 2 --gpu-memory-utilization 0.90
    """
    from concurrent.futures import ThreadPoolExecutor
    from kg.extractors import OpenAIBackend

    if not os.path.exists(COLLECT_PATH):
        raise SystemExit(f"{COLLECT_PATH} not found — run `collect` first.")
    with open(COLLECT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    judge_backend = OpenAIBackend(model=JUDGE_MODEL, api_url=JUDGE_URL, pool_size=8)
    if not judge_backend.is_available():
        raise SystemExit(
            f"judge model {JUDGE_MODEL} not reachable at {JUDGE_URL}.\n"
            f"Do NOT fall back to the extractor model — self-judging invalidates "
            f"the measurement."
        )

    def judge_one(record):
        user = (
            f"SENTENCE: {record['source_sentence'].strip()}\n"
            f"FACT: {verbalize(record['fact_entities'], record['fact_relation'])}\n"
            f"ANSWER:"
        )
        raw = judge_backend._call_with_retry(
            JUDGE_SYSTEM_PROMPT, user, label=record["id"],
            schema=None, max_tokens=8,
        )
        text = (raw or "").strip().lower()
        if text.startswith("yes"):
            return "y"
        if text.startswith("no"):
            return "n"
        return "u"

    print(f"Judging {len(records)} facts with {JUDGE_MODEL}...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(judge_one, records))

    labels = {r["id"]: v for r, v in zip(records, verdicts)}
    with open(JUDGE_LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    counts = {v: sum(1 for x in verdicts if x == v) for v in ("y", "n", "u")}
    n = len(verdicts)
    print(f"\n  entailed   : {counts['y']:>4}  ({100*counts['y']/n:.1f}%)")
    print(f"  FABRICATED : {counts['n']:>4}  ({100*counts['n']/n:.1f}%)")
    print(f"  unclear    : {counts['u']:>4}  ({100*counts['u']/n:.1f}%)")
    print(f"\nWrote {JUDGE_LABELS_PATH}")
    print("\nNext: `validate` — hand-label ~30 to measure judge agreement.")
    print("A judge with no reported human agreement is not evidence.")


def validate(n_sample: int = 30):
    """Hand-label a random subset and report agreement with the LLM judge.

    This is the step that makes the judge citable. Report raw agreement AND
    Cohen's kappa: raw agreement alone is misleading when one class dominates
    (if 90% of facts are entailed, a judge that always says "yes" scores 90%).
    """
    import random

    with open(COLLECT_PATH) as f:
        records = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}
    if not os.path.exists(JUDGE_LABELS_PATH):
        raise SystemExit("run `judge` first")
    with open(JUDGE_LABELS_PATH) as f:
        judge_labels = json.load(f)

    human = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            human = json.load(f)

    random.seed(0)                      # same subset across reruns
    sample_ids = random.sample(sorted(records), min(n_sample, len(records)))
    todo = [rid for rid in sample_ids if rid not in human]

    if todo:
        print("=" * 74)
        print(f"Hand-labelling {len(todo)} facts to validate the judge.")
        print("Judge verdicts are HIDDEN so they cannot anchor you.")
        print("  y = sentence supports it | n = fabricated | u = unclear | q = quit")
        print("=" * 74)
        for i, rid in enumerate(todo, 1):
            record = records[rid]
            print(f"\n[{i}/{len(todo)}]  ({record['repair_kind']} repair)")
            print(f"  asked to connect: {record['target_src']!r} -> "
                  f"{record['target_dst']!r}")
            print(f"\n  SENTENCE ({record['chunk_title']}):")
            print(f"    {record['source_sentence'].strip()}")
            print(f"\n  FACT:")
            print(f"    {verbalize(record['fact_entities'], record['fact_relation'])}")
            while True:
                choice = input("\n  entailed? [y/n/u/q] ").strip().lower()
                if choice in ("y", "n", "u", "q"):
                    break
            if choice == "q":
                break
            human[rid] = choice
            with open(LABELS_PATH, "w") as f:
                json.dump(human, f, indent=2)

    both = [rid for rid in sample_ids if rid in human and rid in judge_labels]
    if not both:
        print("\nno overlapping labels yet")
        return

    agree = sum(1 for rid in both if human[rid] == judge_labels[rid])
    raw = agree / len(both)

    # Cohen's kappa — corrects for agreement expected by chance
    cats = ("y", "n", "u")
    expected = sum(
        (sum(1 for r in both if human[r] == c) / len(both))
        * (sum(1 for r in both if judge_labels[r] == c) / len(both))
        for c in cats
    )
    kappa = (raw - expected) / (1 - expected) if expected < 1 else 1.0

    print("\n" + "=" * 74)
    print(f"JUDGE VALIDATION  (n={len(both)})")
    print("=" * 74)
    print(f"  raw agreement : {100*raw:.1f}%")
    print(f"  Cohen's kappa : {kappa:.3f}")
    if kappa >= 0.8:
        print("  -> strong; judge labels are citable")
    elif kappa >= 0.6:
        print("  -> moderate; usable with the kappa reported alongside")
    else:
        print("  -> WEAK; do not rely on the judge, hand-label everything")

    disagreements = [rid for rid in both if human[rid] != judge_labels[rid]]
    if disagreements:
        print(f"\n  {len(disagreements)} disagreements (showing up to 5):")
        for rid in disagreements[:5]:
            record = records[rid]
            print(f"\n    human={human[rid]}  judge={judge_labels[rid]}")
            print(f"    sentence: {record['source_sentence'].strip()[:130]}")
            print(f"    fact:     "
                  f"{verbalize(record['fact_entities'], record['fact_relation'])}")


# --------------------------------------------------------------------------- #
# PHASE 3 — report
# --------------------------------------------------------------------------- #

def report():
    with open(COLLECT_PATH) as f:
        records = {json.loads(line)["id"]: json.loads(line)
                   for line in f if line.strip()}
    # prefer the full judge labels; fall back to hand labels
    source = JUDGE_LABELS_PATH if os.path.exists(JUDGE_LABELS_PATH) else LABELS_PATH
    with open(source) as f:
        labels = json.load(f)
    print(f"(labels from {source})\n")

    labelled = [(records[rid], lab) for rid, lab in labels.items() if rid in records]
    if not labelled:
        raise SystemExit("no labels found")

    def rate(subset):
        if not subset:
            return None
        n = len(subset)
        entailed = sum(1 for _, lab in subset if lab == "y")
        fabricated = sum(1 for _, lab in subset if lab == "n")
        unclear = sum(1 for _, lab in subset if lab == "u")
        return n, entailed, fabricated, unclear

    print("=" * 74)
    print("REPAIR FABRICATION RATE")
    print("=" * 74)

    groups = [
        ("all", labelled),
        ("segment repairs",  [x for x in labelled if x[0]["repair_kind"] == "segment"]),
        # Terminal repairs are the most leading prompt in the system: the
        # target IS the gold answer string, so the model is told exactly what
        # to find. If fabrication concentrates anywhere, expect it here.
        ("terminal repairs", [x for x in labelled if x[0]["repair_kind"] == "terminal"]),
    ]
    for name, subset in groups:
        result = rate(subset)
        if result is None:
            continue
        n, entailed, fabricated, unclear = result
        print(f"\n{name}  (n={n})")
        print(f"  entailed   : {entailed:>4}  ({100*entailed/n:.1f}%)")
        print(f"  FABRICATED : {fabricated:>4}  ({100*fabricated/n:.1f}%)")
        print(f"  unclear    : {unclear:>4}  ({100*unclear/n:.1f}%)")

    fabricated = [r for r, lab in labelled if lab == "n"]
    if fabricated:
        print("\n" + "=" * 74)
        print(f"FABRICATED EXAMPLES (showing up to 10 of {len(fabricated)})")
        print("=" * 74)
        for record in fabricated[:10]:
            print(f"\n  asked to connect {record['target_src']!r} → "
                  f"{record['target_dst']!r}")
            print(f"  sentence: {record['source_sentence'].strip()[:150]}")
            print(f"  fact:     "
                  f"{verbalize(record['fact_entities'], record['fact_relation'])}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    {"collect": collect, "label": label, "judge": judge,
     "validate": validate, "report": report}[mode]()