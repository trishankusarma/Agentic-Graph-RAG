"""
kg/embeddings/config.py

Tunable constants for the semantic entity-resolution fallback.
"""

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
"""
Small (~130MB), CPU-friendly bi-encoder, run locally via sentence-transformers
— deliberately decoupled from the vLLM server serving qwen3-14b. This keeps
embedding compute off the GPU entirely, so it never competes with extraction
for memory, and encoding a short entity name takes single-digit milliseconds
on CPU.

If you'd rather keep everything behind one server: swap backend.py for one
that calls vLLM's /v1/embeddings endpoint (vllm serve <model> --task embed).
index.py and resolver.py don't care which backend produced the vectors, only
that encode() returns L2-normalized rows — nothing else needs to change.
"""

SIMILARITY_THRESHOLD = 0.72
"""
Minimum cosine similarity for a semantic match to count at all.

This is the semantic equivalent of MIN_FUZZY_ENTITY_LEN in the lexical tiers
(kg/graph/resolver.py). Too low and this becomes the same "loose match found
everything" failure the lexical tiers were hardened against earlier in this
project — too high and it stops rescuing anything the lexical tiers already
miss. This is an UNTUNED starting point. Before trusting it across a full
run, check what it actually rescues:

    resolver.explain("some answer string")
"""

MAX_SEMANTIC_MATCHES = 3
"""
Hard cap on candidates the semantic tier can return.

Uncapped, this tier reproduces the substring-matching bug from earlier in the
graph work — a handful of loose matches feeding into best_path's
|from| x |to| shortest-path search — except now over the FULL node set rather
than one pre-filtered by shared tokens, which is a much larger blast radius.
Keep this small; raise it only if you've confirmed real matches are being cut
off, not just noise.
"""