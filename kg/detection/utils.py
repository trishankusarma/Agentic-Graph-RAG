"""
kg/detection/utils.py

Small helpers shared across the detection package.
"""

import re

from kg.data_loader import HotpotSample

PREDICATE_ANSWERS = {"yes", "no"}
"""Answers that are computed predicates, not entities."""


def normalize(label: str) -> str:
    return label.lower().strip()


def answer_in_text(sample: HotpotSample) -> bool:
    """
    Whether the answer string appears verbatim in any retrieved sentence.

    Separates "never retrieved" from "retrieved but not extracted as an
    entity" — the second is an extraction-recall failure, not a missing edge,
    and wants a different repair action.

    Word-boundary matched, and predicates are excluded outright: plain
    substring matching made "no" match inside "not"/"known"/"Nolan", so every
    comparison sample reported as text-grounded regardless of content.
    """
    needle = sample.answer.lower().strip()
    if not needle or needle in PREDICATE_ANSWERS:
        return False
    pattern = r"\b" + re.escape(needle) + r"\b"
    return any(
        re.search(pattern, sentence.lower())
        for chunk in sample.chunks
        for sentence in chunk.sentences
    )