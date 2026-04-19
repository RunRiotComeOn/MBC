"""Unified LLM client for generator + judge roles.

Dispatches to Anthropic (claude-*), OpenAI (gpt-*, o1-*, o3-*, o4-*), or
DeepSeek (deepseek-*) by model name prefix.
Returns a uniform dict: {"text": str, "api_call_id": str, "model": str, "usage": {...}}.

Network is optional at import time; errors surface when .complete() is called
without the corresponding SDK / API key.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from ..utils.io import load_env_file


@dataclass
class LLMResponse:
    text: str
    api_call_id: str
    model: str
    usage: dict[str, Any]
    raw: Any = None


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._backend = self._detect_backend(model)
        self._client = None  # lazy

    @staticmethod
    def _detect_backend(model: str) -> str:
        m = model.lower()
        if m.startswith("claude"):
            return "anthropic"
        if m.startswith("deepseek"):
            return "deepseek"
        if m.startswith(("gpt", "o1", "o3", "o4")):
            return "openai"
        raise ValueError(f"Unknown LLM model family for {model!r}")

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        load_env_file()
        if self._backend == "anthropic":
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic()
        else:
            from openai import OpenAI  # type: ignore
            if self._backend == "deepseek":
                self._client = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY"),
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                )
            else:
                self._client = OpenAI()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> LLMResponse:
        self._ensure_client()
        t = self.temperature if temperature is None else temperature
        mt = self.max_tokens if max_tokens is None else max_tokens
        if self._backend == "anthropic":
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=mt,
                temperature=t,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            return LLMResponse(
                text=text,
                api_call_id=getattr(msg, "id", str(uuid.uuid4())),
                model=self.model,
                usage={
                    "input_tokens": getattr(msg.usage, "input_tokens", None),
                    "output_tokens": getattr(msg.usage, "output_tokens", None),
                },
                raw=msg,
            )
        # openai
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": t,
            "max_tokens": mt,
            "messages": [],
        }
        if system:
            kwargs["messages"].append({"role": "system", "content": system})
        kwargs["messages"].append({"role": "user", "content": prompt})
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            api_call_id=getattr(resp, "id", str(uuid.uuid4())),
            model=self.model,
            usage={
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
            raw=resp,
        )
