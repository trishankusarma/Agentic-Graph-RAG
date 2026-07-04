"""
kg/extractors/ollama_extractor.py
 
LLMExtractor backend for locally running Ollama models.
Speaks the Ollama /api/chat format.
"""
import logging
import requests
from kg.extractors.base import LLMExtractor
 
logger = logging.getLogger(__name__)

# Hyper-parameters
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_URL = "http://localhost:11434"
MAX_TOKENS = 4096
RETRY_LIMIT = 2
RETRY_DELAY = 1.0

class OllamaBackend(LLMExtractor):
    """
    Ollama backend — POST /api/chat.
 
    Args:
        model:      Ollama model name (e.g. "deepseek-r1:32b")
        ollama_url: Ollama base URL (default: http://localhost:11434)
        max_tokens: max tokens for response
        retry_limit/retry_delay: inherited from LLMExtractor
    """
    def __init__(
        self,
        model:       str   = MODEL_NAME,
        ollama_url:  str   = OLLAMA_URL,
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
        self.ollama_url = ollama_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
                return False
            logger.info(f"Ollama ready — model: {self.model}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot reach Ollama at {self.ollama_url}. "
                "Is it running? → ollama serve"
            )
            return False

    def _call(self, system: str, user: str) -> str:
        payload = {
            "model":   self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.0,
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()