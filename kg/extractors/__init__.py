"""
kg/extractors

Layout:
    config.py            limits, token budgets, timeouts
    schemas.py           JSON Schemas for constrained decoding
    prompts.py           prompt text + what has already been tried
    validation.py        fact filtering + corpus diagnostics (pure functions)
    base.py              BaseExtractor ABC + LLMExtractor shared logic
    openai_extractor.py  vLLM / OpenAI-compatible backend (HTTP only)
"""

from .base import BaseExtractor, LLMExtractor
from .qwen_extractor import OpenAIBackend
from .validation import (
    RejectReason,
    arity_histogram,
    check_fact,
    contamination_rate,
    relation_histogram,
    validate_facts,
)

__all__ = [
    "BaseExtractor",
    "LLMExtractor",
    "OpenAIBackend",
    "RejectReason",
    "arity_histogram",
    "check_fact",
    "contamination_rate",
    "relation_histogram",
    "validate_facts",
]