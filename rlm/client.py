"""Thin OpenRouter client (OpenAI-compatible chat completions)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class CallStats:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, pt: int, ct: int) -> None:
        self.n_calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct


class LLMClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing — put it in .env")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self.model = model or os.environ.get("RLM_ROOT_MODEL", "anthropic/claude-haiku-4.5")
        self.stats = CallStats()

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.stats.add(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
        return resp.choices[0].message.content or ""
