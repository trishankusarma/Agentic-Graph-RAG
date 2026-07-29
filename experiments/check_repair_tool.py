import json
from kg.data_loader import HotpotQALoader
from kg.extractors import OpenAIBackend
from kg.hypergraph_builder import HypergraphBuilder

loader = HotpotQALoader(split="validation", chunk_size=5, overlap=1,
                        cache_path="data/data_loader_cache.jsonl")
samples = loader.load()

# need enough chunks that max_workers is the binding constraint, not len(chunks)
big = max(samples[:200], key=lambda s: len(s.chunks))
print(f"sample {big.sample_id}: {len(big.chunks)} chunks\n")

ex = OpenAIBackend(model="qwen3-14b", api_url="http://localhost:8001", pool_size=64)

baseline = None
for workers in (4, 8, 12, 16, 20):
    if workers > len(big.chunks):
        print(f"workers={workers}: skipped, only {len(big.chunks)} chunks")
        continue
    ex.reset_stats()
    g = HypergraphBuilder(extractor=ex, max_workers=workers, extract_cache={}).build(big.chunks)
    if baseline is None:
        baseline = len(g.nodes)
    ratio = len(g.nodes) / baseline
    print(f"workers={workers:>2}  nodes={len(g.nodes):>4}  edges={len(g.edges):>4}  "
          f"arity={sum(len(set(e.entities)) for e in g.edges.values())/max(len(g.edges),1):.2f}  "
          f"vs baseline={ratio:.0%}  {'OK' if ratio > 0.8 else 'DEGRADED'}")