"""
kg/extractors/openai_extractor.py
 
LLMExtractor backend for any OpenAI-compatible API.
Works with vLLM, LM Studio, OpenAI, together.ai, etc.
 
vLLM serving example (local model path):
    vllm serve /path/to/Qwen2.5-32B-Instruct \
        --served-model-name qwen2.5-32b \
        --port 8000 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192
"""

import logging
import requests
from kg.extractors.base import LLMExtractor
 
logger = logging.getLogger(__name__)

# Hyper-parameters
MODEL_NAME = "qwen2.5-32b"
OLLAMA_URL = "http://localhost:8000"
API_KEY = "EMPTY"
MAX_TOKENS = 4096
RETRY_LIMIT = 2
RETRY_DELAY = 1.0

class OpenAIBackend(LLMExtractor):
    """
    OpenAI-compatible backend — POST /v1/chat/completions.
 
    Args:
        model:      Model name as registered in the server
        api_url:    Base URL (e.g. "http://localhost:8000" for local vLLM)
        api_key:    API key ("EMPTY" for local vLLM, real key for OpenAI)
        max_tokens: max tokens for response
        retry_limit/retry_delay: inherited from LLMExtractor
    """
    def __init__(
        self,
        model:       str   = MODEL_NAME,
        api_url:     str   = OLLAMA_URL,
        api_key:     str   = API_KEY,
        max_tokens:  int   = MAX_TOKENS,
        retry_limit: int   = RETRY_LIMIT,
        retry_delay: float = RETRY_DELAY,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
            retry_delay=retry_delay,
        )
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self.api_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            if self.model not in models:
                logger.warning(
                    f"Model '{self.model}' not found at {self.api_url}. "
                    f"Available: {models}"
                )
                return False
            logger.info(f"OpenAI-compatible server ready — model: {self.model}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot reach API server at {self.api_url}. "
                "Is vLLM running?"
            )
            return False
 
    def _call(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":  self.max_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()