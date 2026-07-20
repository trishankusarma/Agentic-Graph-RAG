"""
kg/extractors/config.py

Every tunable extraction constant lives here, so changing a limit is a
one-line edit in one file rather than a hunt through prompts, schemas and
validation code that all have to agree.
"""

# --------------------------------------------------------------------------- #
# Fact shape
# --------------------------------------------------------------------------- #

MIN_FACT_ARITY = 2
"""A fact needs at least two entities to be a relation at all."""

MAX_FACT_ARITY = 6
"""
Beyond this an "n-ary fact" is really a whole sentence dumped into a list.
"""

MAX_FACTS_PER_SENTENCE = 8
"""
Soft budget stated in the prompt (not enforceable in the schema).

Truncation failures were never long facts — they were ~40 facts from a single
5-sentence chunk, the model enumerating every possible entity pairing. Raising
max_tokens treats the symptom; this treats the cause.
"""

PLACEHOLDER_PREFIX = "<"
"""
Relation labels starting with this are copied prompt scaffolding, not real
labels. See prompts.py for why the example uses angle-bracket placeholders.
"""


# --------------------------------------------------------------------------- #
# Token budgets
# --------------------------------------------------------------------------- #

EXTRACTION_MAX_TOKENS = 2048
"""
Output cap for fact extraction.
"""

ENTITY_MAX_TOKENS = 64
"""Output cap for question entity extraction — a handful of short strings."""


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

RETRY_LIMIT = 2
"""Extra attempts after the first failure, for transport errors only."""

RETRY_DELAY = 1.0
"""Seconds between retry attempts."""

REQUEST_TIMEOUT = 120.0
"""Per-request timeout in seconds."""

HEALTH_CHECK_TIMEOUT = 10.0
"""Timeout for the /v1/models availability probe."""