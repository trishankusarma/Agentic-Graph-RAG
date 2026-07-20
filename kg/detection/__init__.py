"""
kg/detection

    detector.py       BrokenHopDetector — orchestration only
    prefetch.py       prefetch_entities(), QuestionEntityCache
    pairing.py        Pairing — entity -> gold title (comparison questions)
    terminal.py       TerminalResolver — gold[-1] -> answer
    chain_builder.py  ChainBuilder — src -> gold[0] -> ... -> gold[n]
    path_policy.py    best_path() — shared by chain_builder and terminal
    utils.py          normalize(), answer_in_text(), PREDICATE_ANSWERS
"""

from .chain_builder import ChainBuilder
from .detector import BrokenHopDetector
from .pairing import Pairing
from .prefetch import QuestionEntityCache, prefetch_entities
from .terminal import TerminalResolver

__all__ = [
    "BrokenHopDetector",
    "ChainBuilder",
    "Pairing",
    "QuestionEntityCache",
    "TerminalResolver",
    "prefetch_entities",
]