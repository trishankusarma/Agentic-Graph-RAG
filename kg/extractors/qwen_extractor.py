"""
kg/extractors/openai_extractor.py

LLMExtractor backend for any OpenAI-compatible API.
Works with vLLM, LM Studio, OpenAI, together.ai, etc.

vLLM serving example (local model path):
    vllm serve /home/models/Qwen3-14B-Instruct \
        --served-model-name qwen3-14b \
        --port 8000 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 4096 \
        --enable-prefix-caching \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192

    --enable-prefix-caching matters: the system prompt is byte identical on
    every call, so it gets prefilled once instead of N times.
    --max-model-len 4096 (not 8192) roughly doubles how many sequences fit in
    KV cache, which directly doubles batch throughput.
"""

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from kg.extractors.base import LLMExtractor

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMExtractor):
    """
    OpenAI-compatible backend — POST /v1/chat/completions.

    Args:
        model:              Model name as registered with the server.
        api_url:            Base URL (e.g. "http://localhost:8000" for local vLLM).
        api_key:            API key ("EMPTY" for local vLLM, real key for OpenAI).
        pool_size:          HTTP connection pool size. Set to the
                            ThreadPoolExecutor max_workers in HypergraphBuilder —
                            if the pool is smaller, threads block waiting for a
                            connection and you lose the concurrency.
        enable_thinking:    Qwen3 chat-template flag. Keep False for extraction.
        structured_output:  "guided_json" (vLLM <= ~0.9), "response_format"
                            (vLLM >= ~0.10 and the OpenAI API proper), or "off".
                            Confirmed working on this deployment: guided_json.
        timeout:            Per-request timeout in seconds.

    Token caps are inherited from LLMExtractor and NOT redeclared here. An
    earlier version repeated `extraction_max_tokens: int = 768` in this
    signature, which shadowed the base default — raising the base to 2048 then
    silently had no effect and truncation kept discarding chunks. If a cap
    needs changing, change it in base.py or pass it explicitly at the call site.
    """

    def __init__(
        self,
        model:                  str   = "qwen3-14b",
        api_url:                str   = "http://localhost:8000",
        api_key:                str   = "EMPTY",
        pool_size:              int   = 32,
        enable_thinking:        bool  = False,
        structured_output:      str   = "guided_json",
        timeout:                float = 120.0,
        **extractor_kwargs,
    ):
        # extractor_kwargs forwards extraction_max_tokens, entity_max_tokens,
        # retry_limit, retry_delay, use_structured_output — deliberately not
        # restated above, so base defaults always win unless explicitly passed.
        super().__init__(model=model, **extractor_kwargs)

        if structured_output not in ("guided_json", "response_format", "off"):
            raise ValueError(
                f"structured_output must be 'guided_json', 'response_format' "
                f"or 'off' — got {structured_output!r}"
            )

        self.api_url           = api_url.rstrip("/")
        self.api_key           = api_key
        self.enable_thinking   = enable_thinking
        self.structured_output = structured_output
        self.timeout           = timeout
        self._available: Optional[bool] = None

        # Without a Session every call pays a fresh TCP (and TLS) handshake.
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,          # retries are handled in _call_with_retry
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })

    # ----------------------------------------------------------------- #

    def is_available(self) -> bool:
        """Memoized — build() calls this once per builder instance."""
        if self._available is not None:
            return self._available

        try:
            resp = self.session.get(f"{self.api_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
        except requests.RequestException as e:
            # Broad on purpose: raise_for_status raises HTTPError, a hung
            # server raises Timeout, DNS raises ConnectionError.
            logger.error(f"Cannot reach API server at {self.api_url}: {e}")
            self._available = False
            return False
        except (ValueError, KeyError) as e:
            logger.error(f"Malformed /v1/models response from {self.api_url}: {e}")
            self._available = False
            return False

        if self.model not in models:
            logger.warning(
                f"Model '{self.model}' not found at {self.api_url}. "
                f"Available: {models}"
            )
            self._available = False
            return False

        logger.info(f"OpenAI-compatible server ready — model: {self.model}")
        self._available = True
        return True

    def _call(
        self,
        system:     str,
        user:       str,
        schema:     Optional[dict] = None,
        max_tokens: Optional[int]  = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":  max_tokens or self.extraction_max_tokens,
            "temperature": 0.0,
            # Qwen3 chat-template flag. Ignored harmlessly by other models,
            # but it silently no-ops on some vLLM builds — confirm the response
            # carries no <think> block before trusting it.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        if schema is not None and self.structured_output != "off":
            payload.update(self._structured_output_payload(schema))

        resp = self.session.post(
            f"{self.api_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            logger.warning(
                f"Output truncated at {payload['max_tokens']} tokens — "
                "JSON will not parse. Raise extraction_max_tokens."
            )

        msg = choice["message"]
        # Some builds route thinking traces to reasoning_content and leave
        # content null; .strip() on None would crash mid-run.
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return content.strip()

    def _structured_output_payload(self, schema: dict) -> dict:
        """Constrained-decoding fields for the configured API flavour."""
        if self.structured_output == "guided_json":
            # guided_decoding_backend deliberately omitted — this server
            # accepts guided_json without it, and it is the field most likely
            # to break across vLLM versions.
            return {"guided_json": schema}
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name":   "extraction",
                    "schema": schema,
                    "strict": True,
                },
            },
        }

    def close(self) -> None:
        """Release pooled connections. Safe to call more than once."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False