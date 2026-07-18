"""
kg/extractors/ollama_extractor.py

LLMExtractor backend for a local Ollama server — POST /api/chat.

NOTE: written to match the current LLMExtractor contract, since the original
file wasn't on hand. Reconcile against your version before replacing it —
the parts that matter are the four-argument _call() signature and the
super().__init__() keyword names, both of which changed.

Ollama has no constrained-decoding equivalent to vLLM's guided_json, so the
`schema` argument is accepted and ignored. Construct with
use_structured_output=False to skip building schemas that can't be used;
correctness still rests on the prompt plus _validate_facts().

Serving:
    ollama serve
    ollama pull qwen3:14b
"""

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from kg.extractors.base import LLMExtractor

logger = logging.getLogger(__name__)


class OllamaBackend(LLMExtractor):
    """
    Ollama backend — POST /api/chat.

    Args:
        model:      Model tag as pulled (e.g. "qwen3:14b", "deepseek-r1:32b").
        api_url:    Base URL of the Ollama server.
        pool_size:  HTTP connection pool size. Match HypergraphBuilder's
                    max_workers.
        num_ctx:    Context window. Ollama silently truncates to 2048 by
                    default, which will cut the ~450-token system prompt plus a
                    5-sentence chunk short and produce mangled JSON.
        keep_alive: How long to keep the model resident. Without this, Ollama
                    unloads between calls and every request pays a reload.
        timeout:    Per-request timeout in seconds.

    Concurrency caveat: Ollama does not do continuous batching the way vLLM
    does. Set OLLAMA_NUM_PARALLEL (default 1 on older builds) or concurrent
    requests just queue server-side and the thread pool buys nothing.
    """

    def __init__(
        self,
        model:                  str   = "qwen3:14b",
        api_url:                str   = "http://localhost:11434",
        pool_size:              int   = 8,
        num_ctx:                int   = 4096,
        keep_alive:             str   = "10m",
        timeout:                float = 180.0,
        extraction_max_tokens:  int   = 768,
        entity_max_tokens:      int   = 64,
        retry_limit:            int   = 2,
        retry_delay:            float = 1.0,
        use_structured_output:  bool  = False,
    ):
        super().__init__(
            model=model,
            extraction_max_tokens=extraction_max_tokens,
            entity_max_tokens=entity_max_tokens,
            retry_limit=retry_limit,
            retry_delay=retry_delay,
            use_structured_output=use_structured_output,
        )
        self.api_url    = api_url.rstrip("/")
        self.num_ctx    = num_ctx
        self.keep_alive = keep_alive
        self.timeout    = timeout
        self._available: Optional[bool] = None

        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,      # retries live in _call_with_retry
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def is_available(self) -> bool:
        """Memoized — HypergraphBuilder.build() calls this once per instance."""
        if self._available is not None:
            return self._available

        try:
            resp = self.session.get(f"{self.api_url}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = [m["name"] for m in resp.json().get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Cannot reach Ollama at {self.api_url}: {e}")
            self._available = False
            return False
        except (ValueError, KeyError) as e:
            logger.error(f"Malformed /api/tags response: {e}")
            self._available = False
            return False

        # Ollama reports "qwen3:14b"; accept a bare "qwen3" too.
        if not any(t == self.model or t.split(":")[0] == self.model for t in tags):
            logger.warning(
                f"Model '{self.model}' not pulled. Available: {tags}"
            )
            self._available = False
            return False

        logger.info(f"Ollama ready — model: {self.model}")
        self._available = True
        return True

    def _call(
        self,
        system:     str,
        user:       str,
        schema:     Optional[dict] = None,
        max_tokens: Optional[int]  = None,
    ) -> str:
        # schema intentionally unused — no guided decoding on this backend.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":     False,
            "keep_alive": self.keep_alive,
            "format":     "json",   # Ollama's loose JSON mode: shape only
            "options": {
                "temperature": 0.0,
                "num_ctx":     self.num_ctx,
                "num_predict": max_tokens or self.extraction_max_tokens,
            },
        }
        resp = self.session.post(
            f"{self.api_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("done_reason") == "length":
            logger.warning(
                f"Output truncated at {payload['options']['num_predict']} tokens "
                "— JSON will not parse. Raise extraction_max_tokens."
            )
        return data["message"]["content"].strip()

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False