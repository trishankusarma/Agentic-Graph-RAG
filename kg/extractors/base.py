"""
kg/extractors/base.py

The extractor interface and the logic shared by every LLM backend.

    BaseExtractor (ABC)          — what the rest of kg/ depends on
        └── LLMExtractor         — prompt building, retry, parsing, stats
                └── _call(...)   — the ONLY abstract piece a backend implements

Then check stats() for where facts are being lost:

    extractor.stats()
    # {'calls': 305, 'parse_failures': 4, 'facts_valid': 2411,
    #  'facts_rejected': {'placeholder_relation': 248, ...}}
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Optional

from kg.data_loader import Chunk

from . import config
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_repair_prompt,
)
from .schemas import ENTITY_SCHEMA, FACT_SCHEMA
from .validation import validate_facts

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Interface for n-ary relational fact extractors.

    extract() returns validated fact dicts:
        [
            {
                "entities":       [str, ...],   # 2..MAX_FACT_ARITY surface forms
                "relation":       str,          # snake_case label
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

    A backend subclass implements exactly one method: _call(). Everything about
    prompts, retry, parsing, validation and bookkeeping is handled here, so two
    backends cannot drift apart in how they treat a response.

    Args:
        model:                 Model name as registered with the backend.
        use_structured_output: Pass schemas down to _call for constrained
                               decoding. Set False for backends without it, or
                               to A/B whether grammar masking hurts quality.
        debug_dir:             If set, unparseable raw responses are written
                               here, one file per failing chunk.
        extraction_max_tokens / entity_max_tokens / retry_limit / retry_delay:
                               default to the values in config.py. Backends
                               MUST NOT redeclare these in their own signature —
                               a subclass default shadows the config one.
    """

    def __init__(
        self,
        model:                  str,
        use_structured_output:  bool          = True,
        debug_dir:              Optional[str] = None,
        extraction_max_tokens:  int           = config.EXTRACTION_MAX_TOKENS,
        entity_max_tokens:      int           = config.ENTITY_MAX_TOKENS,
        retry_limit:            int           = config.RETRY_LIMIT,
        retry_delay:            float         = config.RETRY_DELAY,
        repair_temperature:     float         = config.REPAIR_TEMPERATURE,
    ):
        self.model                 = model
        self.use_structured_output = use_structured_output
        self.extraction_max_tokens = extraction_max_tokens
        self.entity_max_tokens     = entity_max_tokens
        self.retry_limit           = retry_limit
        self.retry_delay           = retry_delay
        self.repair_temperature    = repair_temperature

        self.debug_dir: Optional[Path] = None
        if debug_dir:
            self.debug_dir = Path(debug_dir)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # bookkeeping — all mutated under _lock, all readable via stats()
        self._lock = threading.Lock()
        self.parse_failures: dict[str, str] = {}   # chunk_id -> reason
        self.reject_counts:  Counter        = Counter()
        self._calls              = 0
        self._facts_valid        = 0
        self._repair_calls       = 0
        self._repair_facts_found = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self, chunk: Chunk) -> list[dict]:
        """Extract relational facts from a chunk. Returns [] on any failure."""
        raw = self._call_with_retry(
            EXTRACTION_SYSTEM_PROMPT,
            build_extraction_prompt(chunk.sentences),
            label=chunk.chunk_id,
            schema=FACT_SCHEMA if self.use_structured_output else None,
            max_tokens=self.extraction_max_tokens,
        )
        if raw is None:
            self._record_parse_failure(chunk.chunk_id, "no response", raw="")
            return []

        parsed = self._parse_json(raw, chunk.chunk_id)
        if parsed is None:
            return []

        valid, rejects = validate_facts(parsed, chunk.sentences)
        with self._lock:
            self.reject_counts.update(rejects)
            self._facts_valid += len(valid)
        if rejects:
            logger.debug(f"{chunk.chunk_id}: rejected {dict(rejects)}")
        return valid

    def extract_targeted(
        self,
        chunk: Chunk,
        src: str,
        dst: str,
        goal: str = "",
    ) -> list[dict]:
        """
        Re-extract a chunk, conditioned on a connection the caller is looking for.

        Same output contract as extract() — validated fact dicts, same shape, so
        hypergraph_builder.add_fact_to_graph consumes them identically. The
        difference is entirely in the prompt (see prompts.REPAIR_SYSTEM_PROMPT for
        why a separate prompt is necessary rather than reusing extract()).

        Returns [] when the passage does not support a connection, which is a
        legitimate and expected outcome — the caller should treat "no facts" as
        "this gap is not repairable from this chunk", not as an error.
        """
        raw = self._call_with_retry(
            REPAIR_SYSTEM_PROMPT,
            build_repair_prompt(chunk.sentences, src=src, dst=dst, goal=goal),
            label=f"repair:{chunk.chunk_id}:{src[:20]}->{dst[:20]}",
            schema=FACT_SCHEMA if self.use_structured_output else None,
            max_tokens=self.extraction_max_tokens,
            temperature=self.repair_temperature,
        )
        if raw is None:
            return []

        parsed = self._parse_json(raw, label=f"repair:{chunk.chunk_id}", record=False)
        if parsed is None:
            return []

        valid, rejects = validate_facts(parsed, chunk.sentences)
        with self._lock:
            self.reject_counts.update(rejects)
            self._repair_calls += 1
            self._repair_facts_found += len(valid)
        return valid

    def extract_entities(self, question: str) -> list[str]:
        """Extract named entities from a question. Returns [] on any failure."""
        raw = self._call_with_retry(
            ENTITY_EXTRACTION_PROMPT,
            f"Q: {question}",
            label=question[:40],
            schema=ENTITY_SCHEMA if self.use_structured_output else None,
            max_tokens=self.entity_max_tokens,
        )
        if raw is None:
            return []

        parsed = self._parse_json(raw, label=question[:40], record=False)
        if not isinstance(parsed, list):
            return []
        return [str(e).strip() for e in parsed if str(e).strip()]

    # ------------------------------------------------------------------ #
    # Backend hook — the one thing a subclass implements
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _call(
        self,
        system:      str,
        user:        str,
        schema:      Optional[dict]  = None,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        One call to the LLM backend.

        Args:
            system:     System prompt.
            user:       User message.
            schema:     JSON Schema for constrained decoding, or None. Backends
                        without constrained decoding should ignore it.
            max_tokens:  Output cap for this call.
            temperature: Sampling temperature, or None for the backend default
                         (0.0). Only repair extraction passes a nonzero value —
                         see extract_targeted().

        Returns:
            Raw response string; may contain <think> blocks or markdown fences,
            which _strip() handles.

        Raises:
            Anything. _call_with_retry catches and retries.
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared machinery
    # ------------------------------------------------------------------ #

    def _call_with_retry(
        self,
        system:     str,
        user:       str,
        label:      str            = "",
        schema:      Optional[dict]  = None,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Call _call() with retry on transport errors. None on total failure."""
        with self._lock:
            self._calls += 1

        for attempt in range(self.retry_limit + 1):
            try:
                return self._call(
                    system, user, schema=schema,
                    max_tokens=max_tokens, temperature=temperature,
                )
            except Exception as e:
                if attempt < self.retry_limit:
                    logger.warning(
                        f"Call failed for '{label}' "
                        f"(attempt {attempt + 1}/{self.retry_limit + 1}): {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Giving up on '{label}': {e}")
                    return None
        return None

    def _parse_json(self, raw: str, label: str, record: bool = True):
        """Strip wrappers and parse. Records + dumps the raw text on failure."""
        try:
            return json.loads(self._strip(raw))
        except (json.JSONDecodeError, ValueError) as e:
            if record:
                logger.error(f"Parse failure for {label}: {e}")
                self._record_parse_failure(label, str(e), raw)
            else:
                logger.warning(f"Parse failure for '{label}': {e}")
            return None

    @staticmethod
    def _strip(raw: str) -> str:
        """Remove <think> blocks and markdown fences."""
        raw = raw.strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return raw

    def _record_parse_failure(self, key: str, reason: str, raw: str) -> None:
        with self._lock:
            self.parse_failures[key] = reason
        if self.debug_dir is not None and raw:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
            try:
                (self.debug_dir / f"{safe}.txt").write_text(raw, encoding="utf-8")
            except OSError as e:
                logger.warning(f"Could not write debug dump for {key}: {e}")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def failure_count_for(self, chunk_ids) -> int:
        """How many of the given chunk ids failed to produce usable output."""
        with self._lock:
            return sum(1 for cid in chunk_ids if cid in self.parse_failures)

    def stats(self) -> dict:
        """
        Where facts were lost. Check this before blaming the graph.

        facts_rejected breaks down by RejectReason: a spike in one reason
        points at exactly which prompt rule stopped working.
        """
        with self._lock:
            return {
                "model":          self.model,
                "calls":          self._calls,
                "parse_failures": len(self.parse_failures),
                "facts_valid":    self._facts_valid,
                "repair_calls":   self._repair_calls,
                "repair_facts":   self._repair_facts_found,
                "facts_rejected": {k.value: v for k, v in self.reject_counts.items()},
            }

    def reset_stats(self) -> None:
        """Clear counters between experiments; leaves debug dumps on disk."""
        with self._lock:
            self.parse_failures.clear()
            self.reject_counts.clear()
            self._calls = 0
            self._facts_valid = 0
            self._repair_calls = 0
            self._repair_facts_found = 0