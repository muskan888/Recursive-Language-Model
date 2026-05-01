"""Baseline: dump the whole prompt into a single LLM call.

If the prompt would clearly overflow, head/tail truncate (this is honest;
the paper notes base LMs degrade or fail outright on long inputs).
"""
from __future__ import annotations

from dataclasses import dataclass

from .client import LLMClient


@dataclass
class BaselineResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool


# Rough char budget — model windows vary on OpenRouter; this is conservative
# so we approximate the paper's "truncate to fit" baseline behavior.
DEFAULT_MAX_CHARS = 180_000  # ~45k tokens


def run_baseline(
    user_query: str,
    context: str,
    client: LLMClient,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> BaselineResult:
    truncated = False
    if len(context) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        context = context[:head] + "\n...[truncated]...\n" + context[-tail:]
        truncated = True

    msg = (
        f"Answer the following query using the provided context.\n\n"
        f"QUERY:\n{user_query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Answer concisely. If asked to count or list, give exactly that and nothing else."
    )
    pt0, ct0 = client.stats.prompt_tokens, client.stats.completion_tokens
    answer = client.chat(
        [{"role": "user", "content": msg}],
        max_tokens=512,
        temperature=0.0,
    )
    return BaselineResult(
        answer=answer.strip(),
        prompt_tokens=client.stats.prompt_tokens - pt0,
        completion_tokens=client.stats.completion_tokens - ct0,
        truncated=truncated,
    )
