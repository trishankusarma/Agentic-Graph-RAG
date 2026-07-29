"""
kg/graph/resolver.py

Free text -> graph node ids.

Two different questions, deliberately answered by two different tier stacks:

    resolve_entity("Robert Erskine Childers DSC")  -> which nodes IS this?
    entities_for_title("Ed Wood")                  -> which nodes came FROM
                                                      this article?

Almost every "broken hop" verdict in the audit traces back to this file. If a
chain reports broken and you believe the entity is present, check resolution
BEFORE checking connectivity — the graph is usually fine and the lookup missed.

Both methods are memoized. Caches are per-instance; graph_store builds one
PathFinder per sample, so they start cold each sample by design.

Semantic tier (new): both stacks now fall back to embedding similarity when
every lexical tier misses. This is the tier of LAST resort — lexical exact
match is unambiguous, token-subset is cheap and precise for real substrings,
semantic is the only one that can be "confidently wrong" (a fluent paraphrase
that isn't actually the same entity), so it only runs when nothing cheaper
already answered the question. See kg/embeddings/config.py for the threshold
and cap, and resolver.explain() to see what it's rescuing.
"""

import logging
from typing import Optional

from kg.embeddings import EmbeddingBackend, SemanticIndex, config as embed_config
from kg.text import normalize

logger = logging.getLogger(__name__)

MIN_FUZZY_ENTITY_LEN = 4
"""
Entities shorter than this are too generic to trust as a fuzzy match
("ed", "us", "the"). Raise it if junk pairings persist.
"""


class EntityResolver:
    """
    Owns entity/title lookup and the indices backing it.

    Args:
        hypergraph:        KnowledgeHypergraph — read-only here; the resolver
                            never mutates it, it only reads .nodes.
        use_semantic:      Toggle the embedding fallback tier off entirely
                            (e.g. to A/B whether it's actually helping, or to
                            skip the sentence-transformers dependency).
        embedding_backend:  Inject a specific EmbeddingBackend (e.g. a
                            different model). Defaults to a lazily-created
                            one using config.DEFAULT_MODEL — lazy so that
                            samples which never need the fallback never pay
                            for loading the embedding model at all.
    """

    def __init__(
        self,
        hypergraph,
        use_semantic: bool = False,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ):
        self.hypergraph = hypergraph
        self.use_semantic = use_semantic
        self._embedding_backend = embedding_backend

        self._chunk_title_index:  dict[str, str]      = {}
        self._title_entity_index: dict[str, set[str]] = {}
        self._title_cache:        dict[str, set[str]] = {}
        self._entity_cache:       dict[str, set[str]] = {}

        self._semantic_index: Optional[SemanticIndex] = None

        self.semantic_rescues = 0
        """
        Diagnostic counter: how many resolve calls were answered ONLY by the
        semantic tier (every lexical tier had already missed). Check this
        after a run — if it's near zero, the tier isn't earning its cost; if
        it's large, look at WHICH labels it rescued (explain()) before
        trusting the threshold blindly.
        """

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index_samples(self, samples) -> None:
        """
        Build chunk_id -> article_title and title -> entity_ids indices.

        Enables tier-1 matching in entities_for_title(). Call right after
        construction: tier 1 is the ONLY tier that uses real article
        provenance rather than string similarity, so resolution quality
        degrades noticeably without it.
        """
        self._chunk_title_index = {
            chunk.chunk_id: chunk.title
            for sample in samples
            for chunk in sample.chunks
        }

        title_index: dict[str, set[str]] = {}
        for node in self.hypergraph.nodes.values():
            for chunk_id in node.chunks:
                title = self._chunk_title_index.get(chunk_id)
                if title:
                    title_index.setdefault(normalize(title), set()).add(node.entity_id)
        self._title_entity_index = title_index

        # Tier 1 results change with the index — memoized values are stale.
        self._title_cache.clear()

        logger.info(
            f"Indexed {len(self._chunk_title_index)} chunk-title mappings, "
            f"{len(self._title_entity_index)} titles with entities"
        )

    # ------------------------------------------------------------------ #
    # Semantic tier — lazy, shared between resolve_entity and
    # entities_for_title (both search the SAME index: entity labels)
    # ------------------------------------------------------------------ #

    def _backend(self) -> EmbeddingBackend:
        if self._embedding_backend is None:
            self._embedding_backend = EmbeddingBackend()
        return self._embedding_backend

    def _get_semantic_index(self) -> SemanticIndex:
        """Built once per instance, on first lexical miss — not at construction."""
        if self._semantic_index is None:
            pairs = [(node.entity_id, node.label) for node in self.hypergraph.nodes.values()]
            self._semantic_index = SemanticIndex.build(self._backend(), pairs)
        return self._semantic_index

    def _semantic_candidates(self, query_text: str) -> set[str]:
        """
        Tier of last resort — see the module docstring for why this only
        runs after every lexical tier has already missed.
        """
        if not self.use_semantic or not query_text.strip():
            return set()

        index = self._get_semantic_index()
        if not index.ids:
            return set()

        query_vector = self._backend().encode([query_text])[0]
        hits = index.search(
            query_vector,
            top_k=embed_config.MAX_SEMANTIC_MATCHES,
            min_similarity=embed_config.SIMILARITY_THRESHOLD,
        )
        if hits:
            self.semantic_rescues += 1
        return {entity_id for entity_id, _score in hits}

    # ------------------------------------------------------------------ #
    # Entity resolution
    # ------------------------------------------------------------------ #

    def resolve_entity(self, label: str) -> set[str]:
        """
        Resolve a free-text label to graph nodes.

        Used for two things: deciding whether a dataset ANSWER is even in the
        graph before asking whether it is reachable (an answer that never
        resolves is not a broken hop — no edge insertion repairs it, see
        TerminalStatus.NOT_AN_ENTITY), and resolving a question entity to a
        chain's starting waypoint.

        Tier 1 — exact normalized match.
        Tier 2 — token-subset in EITHER direction, both sides long enough,
                 e.g. "Robert Erskine Childers DSC" <-> "robert erskine childers".
        Tier 3 — semantic similarity (last resort). Catches paraphrase/synonym
                 cases tiers 1-2 structurally cannot: "US" <-> "United States",
                 "the Falcons" <-> "Atlanta Falcons" — no shared token, no
                 substring relationship, but the same entity.

        Deliberately stricter than plain substring at tier 2: `"no" in "nolan"`
        would otherwise resolve a predicate answer to a real node and turn a
        PREDICATE into a spurious ENTITY_UNREACHED. Tier 3 has its own guard
        for the equivalent failure — see config.SIMILARITY_THRESHOLD's note
        on why too loose a threshold reproduces that exact problem.

        Caveat worth knowing: tier 2 is bidirectional, so a single-token query
        like "roman" matches every multi-token node containing that token
        ("roman empire", "roman catholic church"). Downstream that becomes
        |from| x |to| shortest-path searches in path_policy.best_path. If a
        segment is unexpectedly slow, print len(resolve_entity(x)) first.
        """
        norm = normalize(label)
        if not norm:
            return set()

        cached = self._entity_cache.get(norm)
        if cached is not None:
            return cached

        result = self._resolve_entity_uncached(norm, original=label)
        self._entity_cache[norm] = result
        return result

    def _resolve_entity_uncached(self, norm: str, original: str) -> set[str]:
        # tier 1 — exact
        if norm in self.hypergraph.nodes:
            return {norm}

        # tier 2 — token subset, guarded by length
        if len(norm) >= MIN_FUZZY_ENTITY_LEN:
            tokens = set(norm.split())
            if tokens:
                matches = {
                    node.entity_id
                    for node in self.hypergraph.nodes.values()
                    if len(node.entity_id) >= MIN_FUZZY_ENTITY_LEN
                    and (
                        set(node.entity_id.split()) <= tokens
                        or tokens <= set(node.entity_id.split())
                    )
                }
                if matches:
                    return matches

        # tier 3 — semantic, last resort
        return self._semantic_candidates(original)

    # ------------------------------------------------------------------ #
    # Title resolution
    # ------------------------------------------------------------------ #

    def entities_for_title(self, title: str) -> set[str]:
        """
        Return graph entity ids associated with a Wikipedia article title.

        Tiers 1 and 2 are UNIONED. Tiers 3 and 4 run in order, first non-empty
        wins — each is looser than the last, so unioning THOSE would let the
        loosest dominate and the stricter tiers would stop mattering.

        Tier 1 — chunk title metadata: entities actually extracted from that
                 article's text. Real provenance. Requires index_samples().
        Tier 2 — entity id equals the normalized title exactly.
        Tier 3 — entity's tokens are a subset of the title's tokens, entity at
                 least MIN_FUZZY_ENTITY_LEN chars.
        Tier 4 — semantic similarity (last resort), same index and guard as
                 resolve_entity's tier 3.

        Tiers 1 and 2 are UNIONED; tiers 3 and 4 remain first-non-empty-wins.
        The union is the fix for a real failure: tier 1 is provenance-based and
        can omit a node named after the article itself (when that node was
        created while extracting a different article), so returning tier 1
        alone made every waypoint check against such a title fail even though
        the node existed and was correctly connected. Tier 2 is an exact match
        contributing at most one entity, so unioning it cannot swamp tier 1.

        Tiers 3 and 4 stay exclusive because they are unbounded — tier 3's
        naive-substring ancestor matched "ed" against "Ed Wood" and returned
        dozens of entities, and best_path() then runs |from| x |to|
        shortest-path searches over that.

        Note tier 3 is one-directional (node tokens subset of title tokens),
        unlike resolve_entity's tier 2 which goes both ways. Intentional: a
        title is a fixed reference string, so entities NARROWER than the title
        belong to it ("jonathan stark" -> "Jonathan Stark (tennis)"), but an
        entity BROADER than the title does not. Tier 4 doesn't have this
        directionality constraint — semantic similarity isn't a substring
        relationship, so "broader/narrower" doesn't apply the same way.
        """
        title_norm = normalize(title)

        cached = self._title_cache.get(title_norm)
        if cached is not None:
            return cached

        result = self._resolve_title(title_norm, original=title)
        self._title_cache[title_norm] = result
        return result

    def _resolve_title(self, title_norm: str, original: str) -> set[str]:
        # Tiers 1 and 2 are UNIONED, deliberately breaking the first-non-empty
        # rule that governs the looser tiers below.
        #
        # Tier 1 is provenance-based: entities extracted FROM this article's
        # chunks. It can legitimately omit a node named after the article
        # itself, because that node may have been created while extracting a
        # DIFFERENT article that merely mentioned this one. When that happens,
        # returning tier 1 alone silently excludes the title's own entity, and
        # every waypoint check against this title fails even though the node
        # exists and is correctly connected.
        #
        # Measured: 8 of 18 samples where repair inserted an edge and the chain
        # stayed broken had the new edge connecting EXACTLY the two segment
        # endpoints (e.g. `maxeda` <-> `kohlberg kravis roberts`), with the
        # projected-edge count unchanged — meaning the pair was already
        # directly connected before repair ran. The gap was never a missing
        # edge; the waypoint set simply did not contain the node the edge
        # landed on.
        #
        # Unioning is safe here specifically because tier 2 is an exact match
        # and can add at most one entity, so it cannot swamp tier 1 the way an
        # unbounded fuzzy tier would.
        result: set[str] = set(self._title_entity_index.get(title_norm) or ())
        if title_norm in self.hypergraph.nodes:
            result.add(title_norm)
        if result:
            return result

        # tier 3 — token-subset containment, long entities only. Still
        # fallback-only: this one CAN return dozens of entities, which is what
        # the first-non-empty rule exists to contain.
        title_tokens = set(title_norm.split())
        if title_tokens:
            matches = {
                n.entity_id
                for n in self.hypergraph.nodes.values()
                if len(n.entity_id) >= MIN_FUZZY_ENTITY_LEN
                and set(n.entity_id.split()) <= title_tokens
            }
            if matches:
                return matches

        # tier 4 — semantic, last resort
        return self._semantic_candidates(original)

    # ------------------------------------------------------------------ #
    # Debugging
    # ------------------------------------------------------------------ #

    def explain(self, label: str) -> dict:
        """
        Why did (or didn't) this label resolve? For REPL use when a chain
        reports broken and you suspect lookup rather than connectivity, or
        when calibrating the semantic tier's threshold.

            resolver.explain("Marco Da Silva")
            # {'normalized': 'marco da silva', 'exact_hit': False,
            #  'as_entity': {'marco da silva (dancer)'}, 'as_title': set(),
            #  'title_indexed': False, 'too_short': False,
            #  'semantic_only': False}
        """
        norm = normalize(label)
        via_entity = self.resolve_entity(label)
        via_lexical_only = self._lexical_only(norm)
        return {
            "normalized":     norm,
            "exact_hit":      norm in self.hypergraph.nodes,
            "as_entity":      via_entity,
            "as_title":       self.entities_for_title(label),
            "title_indexed":  norm in self._title_entity_index,
            "too_short":      len(norm) < MIN_FUZZY_ENTITY_LEN,
            "semantic_only":  bool(via_entity) and not via_lexical_only,
        }

    def _lexical_only(self, norm: str) -> set[str]:
        """Re-run tiers 1-2 only, to detect whether resolve_entity's answer
        came from the semantic tier specifically (for explain())."""
        if norm in self.hypergraph.nodes:
            return {norm}
        if len(norm) < MIN_FUZZY_ENTITY_LEN:
            return set()
        tokens = set(norm.split())
        if not tokens:
            return set()
        return {
            node.entity_id
            for node in self.hypergraph.nodes.values()
            if len(node.entity_id) >= MIN_FUZZY_ENTITY_LEN
            and (
                set(node.entity_id.split()) <= tokens
                or tokens <= set(node.entity_id.split())
            )
        }