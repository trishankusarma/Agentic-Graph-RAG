from enum import Enum

class TerminalStatus(str, Enum):
    """How the answer relates to the graph."""

    ENTITY_REACHED   = "entity_reached"     # answer is a node, path found
    ENTITY_UNREACHED = "entity_unreached"   # answer is a node, no path — REAL broken hop
    NOT_AN_ENTITY    = "not_an_entity"      # answer never extracted as a node
    PREDICATE        = "predicate"          # yes/no comparison — no terminal hop exists


class FailureMode(str, Enum):
    """
    Single-label classification of one sample, for histogramming.

    Repairable by a graph action:  BROKEN_MID_CHAIN, BROKEN_TERMINAL,
                                   PREDICATE_BROKEN
    Not a graph failure:           ANSWER_NOT_ENTITY, PREDICATE_OK
    Upstream failure:              NO_QUESTION_ENTITIES
    """

    CONNECTED            = "connected"
    HEALED               = "healed"                 # connected via skip-ahead
    BROKEN_MID_CHAIN     = "broken_mid_chain"       # a gold title is unreachable
    BROKEN_TERMINAL      = "broken_terminal"        # answer is a node, no path to it
    ANSWER_NOT_ENTITY    = "answer_not_entity"      # chain fine, answer isn't a node
    PREDICATE_OK         = "predicate_ok"           # comparison, both articles reached
    PREDICATE_BROKEN     = "predicate_broken"       # comparison, an article unreachable
    NO_QUESTION_ENTITIES = "no_question_entities"   # entity extraction returned nothing