"""
kg/extractors/validation.py

Fact validation and quality diagnostics.
"""

from collections import Counter
from enum import Enum
from typing import Any

from .config import MAX_FACT_ARITY, MIN_FACT_ARITY, PLACEHOLDER_PREFIX


class RejectReason(str, Enum):
    """Why a fact was discarded. Counted, not silently dropped."""

    NOT_A_DICT           = "not_a_dict"
    BAD_ENTITY_LIST      = "bad_entity_list"
    TOO_FEW_ENTITIES     = "too_few_entities"
    TOO_MANY_ENTITIES    = "too_many_entities"
    BLANK_ENTITY         = "blank_entity"
    BAD_RELATION         = "bad_relation"
    PLACEHOLDER_RELATION = "placeholder_relation"
    BAD_SENTENCE_INDEX   = "bad_sentence_index"
    INDEX_OUT_OF_RANGE   = "index_out_of_range"
    BAD_CONFIDENCE       = "bad_confidence"


def _is_real_int(v: Any) -> bool:
    """bool is a subclass of int in Python — `True` must not pass as an index."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_fact(fact: Any, num_sentences: int) -> RejectReason | None:
    """
    Validate one fact. Returns the reason it failed, or None if it passed.

    Split out from validate_facts so a single suspicious fact can be checked in
    isolation from a REPL.
    """
    if not isinstance(fact, dict):
        return RejectReason.NOT_A_DICT

    entities = fact.get("entities")
    if not isinstance(entities, list):
        return RejectReason.BAD_ENTITY_LIST
    if len(entities) < MIN_FACT_ARITY:
        return RejectReason.TOO_FEW_ENTITIES
    if len(entities) > MAX_FACT_ARITY:
        return RejectReason.TOO_MANY_ENTITIES
    if not all(isinstance(e, str) and e.strip() for e in entities):
        return RejectReason.BLANK_ENTITY

    relation = fact.get("relation")
    if not isinstance(relation, str) or not relation.strip():
        return RejectReason.BAD_RELATION
    if relation.strip().startswith(PLACEHOLDER_PREFIX):
        # The model copied the prompt scaffolding instead of reading the
        # sentence. See prompts.py for why the example uses placeholders.
        return RejectReason.PLACEHOLDER_RELATION

    index = fact.get("sentence_index")
    if not _is_real_int(index):
        return RejectReason.BAD_SENTENCE_INDEX
    if not 0 <= index < num_sentences:
        return RejectReason.INDEX_OUT_OF_RANGE

    confidence = fact.get("confidence")
    if not _is_real_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
        return RejectReason.BAD_CONFIDENCE

    return None


def validate_facts(
    facts: Any,
    sentences: list[str],
) -> tuple[list[dict], Counter]:
    """
    Filter a raw fact list.

    Returns (valid_facts, reject_counts). The counter is returned rather than
    logged so callers can aggregate it across a corpus — a spike in one reason
    tells you exactly which prompt rule stopped working.
    """
    rejects: Counter = Counter()

    if not isinstance(facts, list):
        rejects[RejectReason.NOT_A_DICT] += 1
        return [], rejects

    valid = []
    for fact in facts:
        reason = check_fact(fact, len(sentences))
        if reason is None:
            valid.append(fact)
        else:
            rejects[reason] += 1
    return valid, rejects


# --------------------------------------------------------------------------- #
# Corpus diagnostics
# --------------------------------------------------------------------------- #

def contamination_rate(graph) -> float:
    """
    Fraction of edges whose relation label shares no word with its own source
    sentence — a proxy for few-shot label copying.

    Duck-typed (anything with .edges of objects having .relation and .sentence)
    so this module stays free of a hypergraph import.

    Not a filter: legitimate labels do paraphrase, so some non-overlap is
    expected. It is a trend indicator. Target below ~0.10.
    """
    edges = list(getattr(graph, "edges", {}).values())
    if not edges:
        return 0.0

    off = 0
    for e in edges:
        label_tokens = {t for t in e.relation.split("_") if len(t) > 2}
        sentence = e.sentence.lower()
        for ch in ",.;:()\"'":
            sentence = sentence.replace(ch, " ")
        if label_tokens and not (label_tokens & set(sentence.split())):
            off += 1
    return off / len(edges)


def arity_histogram(graph) -> Counter:
    """
    Distribution of distinct entities per hyperedge.

    A healthy distribution decays monotonically. A bump at MAX_FACT_ARITY means
    facts are being clamped by the schema rather than split by the model — see
    the note on MAX_FACT_ARITY in config.py.
    """
    return Counter(
        len(set(e.entities)) for e in getattr(graph, "edges", {}).values()
    )


def relation_histogram(graph, top: int = 15) -> list[tuple[str, int]]:
    """Most common relation labels — the prompt-contamination check."""
    counts = Counter(e.relation for e in getattr(graph, "edges", {}).values())
    return counts.most_common(top)