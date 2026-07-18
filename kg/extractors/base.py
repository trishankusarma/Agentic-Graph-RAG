"""
kg/extractors/base.py

Two-layer extractor hierarchy:

    BaseExtractor (ABC)
        └── LLMExtractor
                ├── all shared logic: prompt building, retry, stripping, validation
                └── _call(system, user, schema, max_tokens) → str   ← only this is abstract

    OllamaBackend(LLMExtractor)   → POST /api/chat
    OpenAIBackend(LLMExtractor)   → POST /v1/chat/completions  (vLLM, OpenAI, etc.)

Adding a new backend = implement _call() and is_available().
"""
import json
import logging
import threading
import time
from abc import ABC, abstractmethod

from kg.data_loader import Chunk

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Extraction limits
# --------------------------------------------------------------------------- #

# Beyond this an "n-ary fact" is really the whole sentence dumped into a list.
# Clique expansion in PathFinder turns one arity-k hyperedge into k(k-1)/2
# binary edges, so a single arity-32 edge contributed 496 projected edges and
# pushed one sample's graph to density 0.54 and avg_degree 16. Paths through
# those edges represent no reasoning, and they silently inflate connectivity —
# a chain looks "connected" because two entities co-occurred in one sentence.
#
# 6 covers a genuine n-ary fact (subject, object, date, place, qualifier) with
# room to spare. Raising this without re-checking avg_degree is a mistake.
MAX_FACT_ARITY = 6


# --------------------------------------------------------------------------- #
# Output schemas — mirror _validate_facts() below. Keep the two in sync.
# --------------------------------------------------------------------------- #

FACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": MAX_FACT_ARITY,
            },
            "relation":       {"type": "string"},
            "sentence_index": {"type": "integer", "minimum": 0},
            "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["entities", "relation", "sentence_index", "confidence"],
        "additionalProperties": False,
    },
}

ENTITY_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}


# --------------------------------------------------------------------------- #
# Prompts
#
# Few-shot contamination — read this before editing the example block.
#
# `relation` is an unconstrained string under guided decoding, so grammar
# pressure pushes the model toward labels already present in context. ANY
# concrete label in the prompt becomes an attractor, and instructions not to
# copy it do not help. Three failed attempts, for the record:
#
#   v1: one film example (directed_in_year, nationality). Both labels then
#       appeared on hockey arenas, universities and book series.
#   v2: five examples across five domains, on the theory that variety would
#       stop any one label dominating. It did the opposite — all five became
#       attractors, taking 485/1244 edges (39%), with a sharp cliff to the next
#       most common label at 20. "seating_capacity" landed on a fight song.
#   v3: one example in an unrelated domain (corporate acquisition) plus an
#       explicit "NEVER reuse this label" rule. The rule was ignored:
#       acquired_company_in_year_for_amount took 117 edges (13.5%), applied to
#       a fight song, a fishing lake and a punk band.
#
# v4 (current): the example contains NO usable label at all — placeholders in
# angle brackets. A copied placeholder is not a plausible relation and is
# trivially visible in the label histogram, so contamination becomes
# self-reporting rather than silent.
#
# Verify with contamination_rate() below. Target < ~0.10. Also grep the label
# histogram for "<" — any literal placeholder means the model copied the
# scaffold instead of reading the sentence.
# --------------------------------------------------------------------------- #

EXTRACTION_SYSTEM_PROMPT = """You are a Knowledge Graph extraction engine.

You will receive a passage where every sentence is prefixed with a sentence index.

Extract every relational fact from the passage.
Treat every numbered sentence independently.
Never combine entities or information from different sentence indices.

For each fact return:
- "entities":       A list of two or more participating entities.
- "relation":       A concise snake_case relation label.
- "sentence_index": The integer index of the sentence this fact came from.
- "confidence":     A float 0-1 representing extraction confidence.

Rules for "relation":
- Derive the label from the main verb or predicate of the sentence you are
  extracting from. Build it out of words that actually appear in that sentence.
- Two sentences that express different things must not share a label.

Rules for "entities":
- Use proper nouns, named concepts, dates, or quantities.
- Keep surface forms exactly as they appear in the sentence.
- A fact may hold a subject, an object, and one or two qualifiers such as a
  date or a place. NEVER put more than 6 entities in a single fact. If a
  sentence contains more, split it into several separate facts.
- Do not repeat the same entity twice within one fact.

Other rules:
- Every fact must originate from exactly one sentence.
- Do not invent facts or merge information across sentences.
- Do not output generic facts such as "is a person" or "exists".
- Return ONLY a JSON array, no explanation, no markdown fences.

Output format (the angle-bracket values below are PLACEHOLDERS showing shape
only — never emit them literally, and never treat them as example labels):

[
  {
    "entities": ["<entity_from_sentence>", "<another_entity>", "<a_date_or_place>"],
    "relation": "<snake_case_verb_from_that_sentence>",
    "sentence_index": 0,
    "confidence": 0.95
  }
]

Every value you emit must come from the passage you are given."""

ENTITY_EXTRACTION_PROMPT = """You are a named entity extractor.

Given a question, extract the key named entities that the question is ABOUT.
These are the entities that would be the starting points for graph traversal.

Rules:
- For bridge questions (who/what/where did X do?): return the main subject entity
- For comparison questions (did X and Y share property Z?): return BOTH X and Y
- Return proper nouns only (people, places, orgs, works, dates)
- Never return a descriptive phrase that is not a proper noun. If the question
  says "the science fantasy young adult series", return the named work or
  person it refers to, or return nothing — not the description itself.
- Return ONLY a JSON array of entity strings, nothing else, no markdown fences.

Examples:
Q: "Who directed the film starring Shirley Temple as Corliss Archer?"
→ ["Shirley Temple"]

Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
→ ["Scott Derrickson", "Ed Wood"]

Q: "What year was the director of Inception born?"
→ ["Inception"]

Q: "What science fantasy young adult series has a companion book narrated by Tobias?"
→ ["Tobias"]"""


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def contamination_rate(graph) -> float:
    """
    Fraction of edges whose relation label shares no word with its own source
    sentence — a proxy for few-shot label copying.

    Duck-typed on purpose (takes anything with .edges of objects having
    .relation and .sentence) so this module stays free of a hypergraph import.

    Not a filter: legitimate labels do paraphrase, so some non-overlap is
    expected. It is a trend indicator. Target below ~0.10.

        from kg.extractors.base import contamination_rate
        print(f"{contamination_rate(graph):.1%}")
    """
    edges = list(getattr(graph, "edges", {}).values())
    if not edges:
        return 0.0

    off = 0
    for e in edges:
        label_tokens = {t for t in e.relation.split("_") if len(t) > 2}
        sentence     = e.sentence.lower()
        for ch in ",.;:()\"'":
            sentence = sentence.replace(ch, " ")
        sent_tokens = set(sentence.split())
        if label_tokens and not (label_tokens & sent_tokens):
            off += 1
    return off / len(edges)


class BaseExtractor(ABC):
    """
    Interface for n-ary relational fact extractors.

    Each extractor receives a Chunk and returns a list of validated fact dicts:
        [
            {
                "entities":       [str, ...],   # 2..MAX_FACT_ARITY surface forms
                "relation":       str,          # snake_case relation label
                "sentence_index": int,          # index into chunk.sentences
                "confidence":     float,        # 0.0 - 1.0
            },
            ...
        ]
    """

    @abstractmethod
    def extract(self, chunk: Chunk) -> list[dict]:
        """Extract relational facts from a single chunk."""
        ...

    @abstractmethod
    def extract_entities(self, question: str) -> list[str]:
        """Extract named entities from a question string."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable and the model is loaded."""
        ...


class LLMExtractor(BaseExtractor):
    """
    Shared logic for all LLM-based extractors.

    Subclasses implement only _call(system, user, schema, max_tokens) → str.

    Args:
        model:                  Model name as registered with the backend.
        extraction_max_tokens:  Output cap for fact extraction. Subclasses MUST
                                NOT redeclare this in their own signature — a
                                subclass default shadows this one and silently
                                reverts the cap.
        entity_max_tokens:      Output cap for question entity extraction.
        retry_limit:            Extra attempts after the first failure.
        retry_delay:            Seconds to sleep between attempts.
        use_structured_output:  Pass FACT_SCHEMA / ENTITY_SCHEMA down to _call.

    Parse failures are recorded in self.parse_failures (chunk_id → reason) so a
    sample whose chunks were dropped can be excluded from pooled statistics.
    Over-arity facts are counted separately in self.dropped_facts — that is a
    quality signal, not a transport failure, so it does not disqualify a sample.
    """

    def __init__(
        self,
        model:                  str,
        extraction_max_tokens:  int   = 2048,
        entity_max_tokens:      int   = 64,
        retry_limit:            int   = 2,
        retry_delay:            float = 1.0,
        use_structured_output:  bool  = True,
    ):
        self.model                 = model
        self.extraction_max_tokens = extraction_max_tokens
        self.entity_max_tokens     = entity_max_tokens
        self.retry_limit           = retry_limit
        self.retry_delay           = retry_delay
        self.use_structured_output = use_structured_output

        self.parse_failures: dict[str, str] = {}
        self.dropped_facts:  dict[str, int] = {}   # chunk_id → over-arity count
        self._failure_lock = threading.Lock()

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def extract(self, chunk: Chunk) -> list[dict]:
        """Extract relational facts from a chunk. Retries on transport failure."""
        prompt = self._build_extraction_prompt(chunk)
        raw    = self._call_with_retry(
            EXTRACTION_SYSTEM_PROMPT,
            prompt,
            label=chunk.chunk_id,
            schema=FACT_SCHEMA if self.use_structured_output else None,
            max_tokens=self.extraction_max_tokens,
        )
        if raw is None:
            self._record_failure(chunk.chunk_id, "no response")
            return []
        try:
            facts = json.loads(self._strip(raw))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Parse failure for chunk {chunk.chunk_id}: {e}")
            self._record_failure(chunk.chunk_id, str(e))
            return []
        if not isinstance(facts, list):
            self._record_failure(chunk.chunk_id, f"got {type(facts).__name__}")
            return []

        valid = self._validate_facts(facts, chunk)
        over  = sum(
            1 for f in facts
            if isinstance(f, dict)
            and isinstance(f.get("entities"), list)
            and len(f["entities"]) > MAX_FACT_ARITY
        )
        if over:
            self._record_dropped(chunk.chunk_id, over)
            logger.debug(
                f"{chunk.chunk_id}: dropped {over} facts over arity "
                f"{MAX_FACT_ARITY}"
            )
        return valid

    def extract_entities(self, question: str) -> list[str]:
        """Extract named entities from a question. Retries on transport failure."""
        raw = self._call_with_retry(
            ENTITY_EXTRACTION_PROMPT,
            f"Q: {question}",
            label=question[:40],
            schema=ENTITY_SCHEMA if self.use_structured_output else None,
            max_tokens=self.entity_max_tokens,
        )
        if raw is None:
            return []
        try:
            entities = json.loads(self._strip(raw))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Entity parse failure for '{question[:40]}': {e}")
            return []
        if not isinstance(entities, list):
            return []
        return [str(e).strip() for e in entities if str(e).strip()]

    def _record_failure(self, chunk_id: str, reason: str) -> None:
        with self._failure_lock:
            self.parse_failures[chunk_id] = reason

    def _record_dropped(self, chunk_id: str, count: int) -> None:
        with self._failure_lock:
            self.dropped_facts[chunk_id] = self.dropped_facts.get(chunk_id, 0) + count

    def failure_count_for(self, chunk_ids) -> int:
        """How many of the given chunk ids failed to produce usable output."""
        with self._failure_lock:
            return sum(1 for cid in chunk_ids if cid in self.parse_failures)

    def dropped_count_for(self, chunk_ids) -> int:
        """How many over-arity facts were discarded across the given chunks."""
        with self._failure_lock:
            return sum(self.dropped_facts.get(cid, 0) for cid in chunk_ids)

    # ----------------------------------------------------------------- #
    # Backend hook
    # ----------------------------------------------------------------- #

    @abstractmethod
    def _call(
        self,
        system:     str,
        user:       str,
        schema:     dict | None = None,
        max_tokens: int | None  = None,
    ) -> str:
        """
        Make one call to the LLM backend.

        Args:
            system:     System prompt.
            user:       User message.
            schema:     JSON Schema for constrained decoding, or None. Backends
                        that cannot constrain output should ignore it.
            max_tokens: Output cap for this call.

        Returns:
            Raw response string (may contain <think> blocks or markdown fences).
        """
        ...

    def _call_with_retry(
        self,
        system:     str,
        user:       str,
        label:      str         = "",
        schema:     dict | None = None,
        max_tokens: int | None  = None,
    ) -> str | None:
        """Call _call() with retry on any exception. Returns None on total failure."""
        for attempt in range(self.retry_limit + 1):
            try:
                return self._call(system, user, schema=schema, max_tokens=max_tokens)
            except Exception as e:
                if attempt < self.retry_limit:
                    logger.warning(
                        f"Call failed for '{label}' "
                        f"(attempt {attempt + 1}): {e} — retrying"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Giving up on '{label}': {e}")
                    return None
        return None

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #

    @staticmethod
    def _build_extraction_prompt(chunk: Chunk) -> str:
        numbered = "\n".join(
            f"[{i}] {sent}" for i, sent in enumerate(chunk.sentences)
        )
        return f"Extract all relational facts from the following passage:\n\n{numbered}"

    @staticmethod
    def _strip(raw: str) -> str:
        """Strip <think> blocks and markdown fences."""
        raw = raw.strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return raw

    @staticmethod
    def _validate_facts(facts: list, chunk: Chunk) -> list[dict]:
        """
        Filter out malformed facts.

        Still required under constrained decoding: the grammar guarantees shape,
        not semantics. sentence_index is only bounded against the real chunk
        length here, and MAX_FACT_ARITY is enforced here as well as in the
        schema so it still holds when use_structured_output is False.
        """
        valid = []
        for f in facts:
            if (
                isinstance(f, dict)
                and isinstance(f.get("entities"), list)
                and 2 <= len(f["entities"]) <= MAX_FACT_ARITY
                and all(isinstance(e, str) and e.strip() for e in f["entities"])
                and isinstance(f.get("relation"), str)
                and f["relation"].strip()
                and isinstance(f.get("sentence_index"), int)
                and not isinstance(f.get("sentence_index"), bool)
                and 0 <= f["sentence_index"] < len(chunk.sentences)
                and isinstance(f.get("confidence"), (int, float))
                and not isinstance(f.get("confidence"), bool)
                and 0.0 <= float(f["confidence"]) <= 1.0
            ):
                valid.append(f)
        return valid

    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()