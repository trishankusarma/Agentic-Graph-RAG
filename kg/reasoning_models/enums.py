"""
kg/reasoning_models/enums.py

Vocabulary for classifying reasoning-chain outcomes. This is semantics, not
detection logic — kg/detection/ imports from here, this file never imports
from kg/detection/.
"""

from enum import Enum


class TerminalStatus(str, Enum):
    """How the dataset answer relates to the graph."""

    ENTITY_REACHED   = "entity_reached"     # answer is a node, path found
    ENTITY_UNREACHED = "entity_unreached"   # answer is a node, no path — REAL break
    NOT_AN_ENTITY    = "not_an_entity"      # answer never extracted as a node
    PREDICATE        = "predicate"          # yes/no — no terminal hop exists


class FailureMode(str, Enum):
    """
    Single-label classification of one sample, for histogramming.

    Repairable by a graph action:
        BROKEN_MID_CHAIN, BROKEN_TERMINAL, PREDICATE_BROKEN
    Not a graph failure:
        CONNECTED, HEALED, ANSWER_NOT_ENTITY, PREDICATE_OK
    Upstream failure (the question parser, not the graph):
        SRC_UNRESOLVED, NO_QUESTION_ENTITIES
    """

    CONNECTED            = "connected"
    HEALED               = "healed"                 # connected via skip-ahead
    BROKEN_MID_CHAIN     = "broken_mid_chain"       # a gold title is unreachable
    BROKEN_TERMINAL      = "broken_terminal"        # answer is a node, no path
    ANSWER_NOT_ENTITY    = "answer_not_entity"      # chain fine, answer isn't a node
    PREDICATE_OK         = "predicate_ok"           # comparison, both reached
    PREDICATE_BROKEN     = "predicate_broken"       # comparison, one unreachable
    SRC_UNRESOLVED       = "src_unresolved"         # question entity not in graph
    NO_QUESTION_ENTITIES = "no_question_entities"   # extractor returned nothing


REPAIRABLE_MODES = frozenset({
    FailureMode.BROKEN_MID_CHAIN,
    FailureMode.BROKEN_TERMINAL,
    FailureMode.PREDICATE_BROKEN,
})
"""
Modes a graph repair action could plausibly move.
"""


SEVERITY_ORDER = (
    FailureMode.NO_QUESTION_ENTITIES,
    FailureMode.SRC_UNRESOLVED,
    FailureMode.BROKEN_MID_CHAIN,
    FailureMode.PREDICATE_BROKEN,
    FailureMode.BROKEN_TERMINAL,
    FailureMode.ANSWER_NOT_ENTITY,
    FailureMode.HEALED,
    FailureMode.PREDICATE_OK,
    FailureMode.CONNECTED,
)
"""
Worst-first. A sample's mode is the worst across its chains: a comparison
sample where one article is unreachable is broken even if the other is fine.
"""


def worst_mode(modes) -> FailureMode:
    """
    Most severe mode present in `modes`. Empty iterable → BROKEN_MID_CHAIN
    """
    present = set(modes)
    for mode in SEVERITY_ORDER:
        if mode in present:
            return mode
    return FailureMode.BROKEN_MID_CHAIN