"""
kg/extractors/base.py
 
Two-layer extractor hierarchy:
 
    BaseExtractor (ABC)
        └── LLMExtractor
                ├── all shared logic: prompt building, retry, stripping, validation
                └── _call(prompt) → str   ← only this is abstract
 
    OllamaBackend(LLMExtractor)   → POST /api/chat
    OpenAIBackend(LLMExtractor)   → POST /v1/chat/completions  (vLLM, OpenAI, etc.)
 
Adding a new backend = implement _call() and is_available().
"""
import json
import logging
import time
from abc import ABC, abstractmethod
 
from kg.data_loader import Chunk
 
logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a Knowledge Graph extraction engine.
 
You will receive a passage where every sentence is prefixed with a sentence index.
 
Example:
 
[0] Christopher Nolan directed The Dark Knight in 2008.
[1] He is a British-American filmmaker.
 
Extract every relational fact from the passage.
Treat every numbered sentence independently.
Never combine entities or information from different sentence indices.
 
For each fact return:
- "entities":       A list of two or more participating entities.
- "relation":       A concise snake_case relation label.
- "sentence_index": The integer index of the sentence this fact came from.
- "confidence":     A float 0-1 representing extraction confidence.
 
Rules:
- Every fact must originate from exactly one sentence.
- Entities should be proper nouns or named concepts.
- Keep entity surface forms exactly as they appear in the sentence.
- Do not invent facts or merge information across sentences.
- Do not output generic facts such as "is a person" or "exists".
- Return ONLY a JSON array, no explanation, no markdown fences.
 
Example output:
[
  {
    "entities": ["Christopher Nolan", "The Dark Knight", "2008"],
    "relation": "directed_in_year",
    "sentence_index": 0,
    "confidence": 0.97
  },
  {
    "entities": ["Christopher Nolan", "British-American"],
    "relation": "nationality",
    "sentence_index": 1,
    "confidence": 0.95
  }
]"""

ENTITY_EXTRACTION_PROMPT = """You are a named entity extractor.
 
Given a question, extract the key named entities that the question is ABOUT.
These are the entities that would be the starting points for graph traversal.
 
Rules:
- For bridge questions (who/what/where did X do?): return the main subject entity
- For comparison questions (did X and Y share property Z?): return BOTH X and Y
- Return proper nouns only (people, places, orgs, works, dates)
- Return ONLY a JSON array of entity strings, nothing else, no markdown fences.
 
Examples:
Q: "Who directed the film starring Shirley Temple as Corliss Archer?"
→ ["Shirley Temple"]
 
Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
→ ["Scott Derrickson", "Ed Wood"]
 
Q: "What year was the director of Inception born?"
→ ["Inception"]"""

class BaseExtractor(ABC):
    """
    Interface for n-ary relational fact extractors.

    Each extractor receives a Chunk and returns a list of validated
    fact dicts:
        [
            {
                "entities":       [str, ...],   # 2+ normalized entity strings
                "relation":       str,           # snake_case relation label
                "sentence_index": int,           # index into chunk.sentences
                "confidence":     float,         # 0.0 - 1.0
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
 
    Subclasses implement only _call(system, user) → str,
    which handles the backend-specific HTTP format.
    """
    def __init__(
        self,
        model:       str,
        max_tokens:  int   = 4096,
        retry_limit: int   = 2,
        retry_delay: float = 1.0,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
    
    def extract(self, chunk: Chunk) -> list[dict]:
        """Extract relational facts from a chunk. Retries on parse failure."""
        prompt = self._build_extraction_prompt(chunk)
        raw    = self._call_with_retry(EXTRACTION_SYSTEM_PROMPT, prompt, chunk.chunk_id)
        if raw is None:
            return []
        try:
            facts = json.loads(self._strip(raw))
            return self._validate_facts(facts, chunk)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Final parse failure for chunk {chunk.chunk_id}: {e}")
            return []
        
    def extract_entities(self, question: str) -> list[str]:
        """Extract named entities from a question. Retries on parse failure."""
        raw = self._call_with_retry(
            ENTITY_EXTRACTION_PROMPT,
            f"Q: {question}",
            label=question[:40],
        )
        if raw is None:
            return []
        try:
            entities = json.loads(self._strip(raw))
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Entity parse failure for '{question[:40]}': {e}")
            return []
    
    @abstractmethod
    def _call(self, system: str, user: str) -> str:
        """
        Make one HTTP call to the LLM backend.
        Returns the raw response string (may contain <think> blocks, fences).
        """
        ...
    
    def _call_with_retry(
        self,
        system: str,
        user:   str,
        label:  str = "",
    ) -> str | None:
        """Call _call() with retry on any exception. Returns None on total failure."""
        for attempt in range(self.retry_limit + 1):
            try:
                return self._call(system, user)
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
        """Filter out malformed facts."""
        valid = []
        for f in facts:
            if (
                isinstance(f, dict)
                and isinstance(f.get("entities"), list)
                and len(f["entities"]) >= 2
                and isinstance(f.get("relation"), str)
                and f["relation"].strip()
                and isinstance(f.get("sentence_index"), int)
                and 0 <= f["sentence_index"] < len(chunk.sentences)
                and isinstance(f.get("confidence"), (int, float))
                and 0.0 <= float(f["confidence"]) <= 1.0
            ):
                valid.append(f)
        return valid
 
    @staticmethod
    def _normalize(label: str) -> str:
        return label.lower().strip()