"""
kg/extractors/qwen_extractor.py

Backend for any OpenAI-compatible API (vLLM, LM Studio, OpenAI, together.ai).

This file is HTTP only. Prompts, schemas, limits and validation live in their
own modules — if extraction quality is wrong, this is not the file to edit.

vLLM serving:
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
    every call, so it is prefilled once instead of N times.
    --max-model-len 4096 (not 8192) roughly doubles how many sequences fit in
    KV cache, which directly doubles batch throughput.
"""

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from . import config
from .base import LLMExtractor

logger = logging.getLogger(__name__)

STRUCTURED_OUTPUT_MODES = ("guided_json", "response_format", "off")


class OpenAIBackend(LLMExtractor):
    """
    POST /v1/chat/completions.

    Args:
        model:             Model name as registered with the server.
        api_url:           Base URL, e.g. "http://localhost:8000".
        api_key:           "EMPTY" for local vLLM, a real key for OpenAI.
        pool_size:         HTTP connection pool size. Set it to the
                           ThreadPoolExecutor max_workers in HypergraphBuilder;
                           if the pool is smaller, threads block waiting for a
                           connection and the extra workers buy nothing.
        enable_thinking:   Qwen3 chat-template flag. Keep False for extraction.
        structured_output: Which constrained-decoding API this server speaks.
                           "guided_json"     vLLM <= ~0.9  (confirmed on ours)
                           "response_format" vLLM >= ~0.10 and OpenAI proper
                           "off"             no constraint; prompt-only JSON
                           These are mutually incompatible and a strict server
                           400s on the wrong one. Check with:
                               python -c "import vllm; print(vllm.__version__)"
        timeout:           Per-request timeout in seconds.
        **extractor_kwargs: forwarded to LLMExtractor (use_structured_output,
                           debug_dir, token caps, retry settings). Deliberately
                           NOT restated here so config.py defaults always win.
    """

    def __init__(
        self,
        model:             str   = "qwen3-14b",
        api_url:           str   = "http://localhost:8000",
        api_key:           str   = "EMPTY",
        pool_size:         int   = 32,
        enable_thinking:   bool  = False,
        structured_output: str   = "guided_json",
        timeout:           float = config.REQUEST_TIMEOUT,
        **extractor_kwargs,
    ):
        super().__init__(model=model, **extractor_kwargs)

        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"structured_output must be one of {STRUCTURED_OUTPUT_MODES} — "
                f"got {structured_output!r}"
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
            max_retries=0,          # retries belong to _call_with_retry
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })

    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Check the server is up and the model is registered. Memoized."""
        if self._available is not None:
            return self._available

        self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.api_url}/v1/models", timeout=config.HEALTH_CHECK_TIMEOUT
            )
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
        except requests.RequestException as e:
            # Broad on purpose: raise_for_status raises HTTPError, a hung
            # server raises Timeout, bad DNS raises ConnectionError.
            logger.error(f"Cannot reach API server at {self.api_url}: {e}")
            return False
        except (ValueError, KeyError) as e:
            logger.error(f"Malformed /v1/models response from {self.api_url}: {e}")
            return False

        if self.model not in models:
            logger.warning(
                f"Model '{self.model}' not found at {self.api_url}. "
                f"Available: {models}"
            )
            return False

        logger.info(f"OpenAI-compatible server ready — model: {self.model}")
        return True

    # ------------------------------------------------------------------ #

    def _call(
        self,
        system:     str,
        user:       str,
        schema:     Optional[dict] = None,
        max_tokens: Optional[int]  = None,
    ) -> str:
        payload = self._build_payload(system, user, schema, max_tokens)

        resp = self.session.post(
            f"{self.api_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._read_content(resp.json(), payload["max_tokens"])

    def _build_payload(
        self,
        system:     str,
        user:       str,
        schema:     Optional[dict],
        max_tokens: Optional[int],
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":  max_tokens or self.extraction_max_tokens,
            "temperature": 0.0,
            # Qwen3 chat-template flag. Harmless for other models, but it
            # silently no-ops on some vLLM builds — confirm no <think> block
            # in the response before trusting it.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if schema is not None and self.structured_output != "off":
            payload.update(self._structured_output_fields(schema))
        return payload

    def _structured_output_fields(self, schema: dict) -> dict:
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

    @staticmethod
    def _read_content(data: dict, max_tokens: int) -> str:
        choice = data["choices"][0]

        if choice.get("finish_reason") == "length":
            logger.warning(
                f"Output truncated at {max_tokens} tokens — JSON will not "
                "parse. Usually means too MANY facts, not long ones: check "
                "MAX_FACTS_PER_SENTENCE in config.py before raising the cap."
            )

        msg = choice["message"]
        # Some builds route thinking traces to reasoning_content and leave
        # content null; .strip() on None would crash mid-run.
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release pooled connections. Safe to call more than once."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False