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
"""

import logging

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
        hypergraph: KnowledgeHypergraph — read-only here; the resolver never
                    mutates it, it only reads .nodes.
    """

    def __init__(self, hypergraph):
        self.hypergraph = hypergraph

        self._chunk_title_index:  dict[str, str]      = {}
        self._title_entity_index: dict[str, set[str]] = {}
        self._title_cache:        dict[str, set[str]] = {}
        self._entity_cache:       dict[str, set[str]] = {}

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

        Deliberately stricter than plain substring: `"no" in "nolan"` would
        otherwise resolve a predicate answer to a real node and turn a
        PREDICATE into a spurious ENTITY_UNREACHED.

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

        result = self._resolve_entity_uncached(norm)
        self._entity_cache[norm] = result
        return result

    def _resolve_entity_uncached(self, norm: str) -> set[str]:
        # tier 1 — exact
        if norm in self.hypergraph.nodes:
            return {norm}

        # tier 2 — token subset, guarded by length
        if len(norm) < MIN_FUZZY_ENTITY_LEN:
            return set()

        tokens = set(norm.split())
        if not tokens:
            return set()

        matches = set()
        for node in self.hypergraph.nodes.values():
            if len(node.entity_id) < MIN_FUZZY_ENTITY_LEN:
                continue
            node_tokens = set(node.entity_id.split())
            if node_tokens <= tokens or tokens <= node_tokens:
                matches.add(node.entity_id)
        return matches

    # ------------------------------------------------------------------ #
    # Title resolution
    # ------------------------------------------------------------------ #

    def entities_for_title(self, title: str) -> set[str]:
        """
        Return graph entity ids associated with a Wikipedia article title.

        Tiers run in order and the FIRST non-empty one wins — they are NOT
        unioned. Each is looser than the last, so unioning would let the
        loosest dominate and the stricter tiers would stop mattering.

        Tier 1 — chunk title metadata: entities actually extracted from that
                 article's text. Real provenance. Requires index_samples().
        Tier 2 — entity id equals the normalized title exactly.
        Tier 3 — entity's tokens are a subset of the title's tokens, entity at
                 least MIN_FUZZY_ENTITY_LEN chars.

        Tier 3 is narrower than plain containment on purpose: naive substring
        matched "ed" against "Ed Wood" and returned dozens of entities, and
        best_path() then runs |from| x |to| shortest-path searches over that.

        Note tier 3 is one-directional (node tokens subset of title tokens),
        unlike resolve_entity's tier 2 which goes both ways. Intentional: a
        title is a fixed reference string, so entities NARROWER than the title
        belong to it ("jonathan stark" -> "Jonathan Stark (tennis)"), but an
        entity BROADER than the title does not.
        """
        title_norm = normalize(title)

        cached = self._title_cache.get(title_norm)
        if cached is not None:
            return cached

        result = self._resolve_title(title_norm)
        self._title_cache[title_norm] = result
        return result

    def _resolve_title(self, title_norm: str) -> set[str]:
        # tier 1 — chunk title metadata
        hit = self._title_entity_index.get(title_norm)
        if hit:
            return set(hit)

        # tier 2 — exact entity id match
        if title_norm in self.hypergraph.nodes:
            return {title_norm}

        # tier 3 — token-subset containment, long entities only
        title_tokens = set(title_norm.split())
        if not title_tokens:
            return set()
        return {
            n.entity_id
            for n in self.hypergraph.nodes.values()
            if len(n.entity_id) >= MIN_FUZZY_ENTITY_LEN
            and set(n.entity_id.split()) <= title_tokens
        }

    # ------------------------------------------------------------------ #
    # Debugging
    # ------------------------------------------------------------------ #

    def explain(self, label: str) -> dict:
        """
        Why did (or didn't) this label resolve? For REPL use when a chain
        reports broken and you suspect lookup rather than connectivity.

            resolver.explain("Marco Da Silva")
            # {'normalized': 'marco da silva', 'exact_hit': False,
            #  'as_entity': {'marco da silva (dancer)'}, 'as_title': set(), ...}
        """
        norm = normalize(label)
        return {
            "normalized":     norm,
            "exact_hit":      norm in self.hypergraph.nodes,
            "as_entity":      self.resolve_entity(label),
            "as_title":       self.entities_for_title(label),
            "title_indexed":  norm in self._title_entity_index,
            "too_short":      len(norm) < MIN_FUZZY_ENTITY_LEN,
        }