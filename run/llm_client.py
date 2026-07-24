"""
LLM API client abstraction.

Supports OpenAI-compatible and Anthropic backends.  Add new providers by
subclassing ``BaseLLMClient`` and registering in ``get_llm_client``.

Usage::

    from run.config import LLMConfig
    from run.llm_client import get_llm_client

    client = get_llm_client(LLMConfig(provider="openai", model="gpt-4o", api_key="..."))
    response = client.generate("What is 2+2?")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from run.config import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send *prompt* to the LLM and return the text response."""
        ...


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """Calls the OpenAI Chat Completions API (also works with vLLM, Ollama, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None  # lazy init

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url,
            )
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug("LLM request: model=%s, prompt_len=%d", self.config.model, len(prompt))

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.stop:
            kwargs["stop"] = self.config.stop
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        logger.debug("LLM response: len=%d", len(content))
        return content


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """Calls the Anthropic Messages API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.stop:
            kwargs["stop_sequences"] = self.config.stop
        if system_prompt:
            kwargs["system"] = system_prompt

        logger.debug("Anthropic request: model=%s, prompt_len=%d", self.config.model, len(prompt))

        response = self.client.messages.create(**kwargs)

        # Anthropic returns a list of content blocks; grab the first text block
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
        logger.debug("Anthropic response: len=%d", len(content))
        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Create an LLM client from a configuration object."""
    provider = config.provider.lower()
    if provider in ("openai", "openai-compatible"):
        return OpenAIClient(config)
    elif provider == "anthropic":
        return AnthropicClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openai, anthropic")
