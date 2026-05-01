"""Algorithm 1: Recursive Language Model loop.

Root LLM only sees metadata about the prompt (length, prefix). It emits
```repl blocks that we exec() against a persistent Python REPL whose globals
include `context` (the full input) and `llm_query` (a callable that fires a
sub-LLM, allowing programmatic recursion).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .client import LLMClient
from .prompts import RLM_NO_SUBCALL_PROMPT, RLM_SYSTEM_PROMPT
from .repl import Repl

REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)```", re.DOTALL)
FINAL_RE = re.compile(r"FINAL\((.*?)\)\s*$", re.DOTALL)
FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


@dataclass
class RLMResult:
    answer: str
    n_iterations: int
    n_subcalls: int
    n_root_calls: int
    prompt_tokens: int
    completion_tokens: int
    trace: list[str] = field(default_factory=list)


def _metadata(prompt: str, prefix_chars: int = 400) -> str:
    return (
        f"Length: {len(prompt)} characters.\n"
        f"Prefix (first {prefix_chars} chars):\n{prompt[:prefix_chars]}\n"
        f"Suffix (last 200 chars):\n{prompt[-200:]}"
    )


def _parse_final(text: str, repl: Repl) -> Optional[str]:
    # Look outside any ```repl blocks
    stripped = REPL_BLOCK_RE.sub("", text)
    m = FINAL_VAR_RE.search(stripped)
    if m:
        var = m.group(1)
        if var in repl.globals:
            return str(repl.globals[var])
        return f"(FINAL_VAR referenced unknown variable: {var})"
    m = FINAL_RE.search(stripped.strip())
    if m:
        return m.group(1).strip()
    # also accept FINAL(...) anywhere
    m = re.search(r"FINAL\(((?:[^()]|\([^()]*\))*)\)", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def run_rlm(
    user_query: str,
    context: str,
    root_client: LLMClient,
    sub_client: Optional[LLMClient] = None,
    max_iterations: int = 10,
    sub_char_budget: int = 200_000,
    context_type: str = "string",
    verbose: bool = False,
) -> RLMResult:
    """Run the RLM loop. If sub_client is None, runs in no-sub-call mode."""

    subcall_count = [0]

    def llm_query(prompt: str) -> str:
        if sub_client is None:
            raise RuntimeError("Sub-calls disabled in this RLM run.")
        if len(prompt) > sub_char_budget * 2:
            prompt = prompt[: sub_char_budget * 2]
        subcall_count[0] += 1
        return sub_client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )

    repl_globals = {"context": context}
    if sub_client is not None:
        repl_globals["llm_query"] = llm_query
    repl = Repl(repl_globals)

    def _fill(template: str) -> str:
        return (
            template.replace("{context_type}", context_type)
            .replace("{context_total_length}", str(len(context)))
            .replace("{sub_char_budget}", str(sub_char_budget))
        )

    sys_prompt = _fill(RLM_SYSTEM_PROMPT if sub_client is not None else RLM_NO_SUBCALL_PROMPT)

    user_msg = (
        f"USER QUERY:\n{user_query}\n\n"
        f"CONTEXT METADATA (the full text is in the `context` variable):\n{_metadata(context)}"
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    trace: list[str] = []

    for i in range(max_iterations):
        reply = root_client.chat(messages, max_tokens=2048, temperature=0.0)
        trace.append(f"--- iter {i} root ---\n{reply}")
        if verbose:
            print(f"\n=== ROOT iter {i} ===\n{reply}\n")

        # Did the model declare FINAL?
        ans = _parse_final(reply, repl)
        # Run any code blocks first (paper allows code + FINAL in same turn? we run code then check)
        code_blocks = REPL_BLOCK_RE.findall(reply)
        repl_outputs: list[str] = []
        for block in code_blocks:
            out = repl.exec(block)
            repl_outputs.append(out)
            if verbose:
                print(f"--- repl out ---\n{out}\n")

        if ans is not None and not code_blocks:
            return RLMResult(
                answer=ans,
                n_iterations=i + 1,
                n_subcalls=subcall_count[0],
                n_root_calls=root_client.stats.n_calls,
                prompt_tokens=root_client.stats.prompt_tokens,
                completion_tokens=root_client.stats.completion_tokens,
                trace=trace,
            )
        if ans is not None and code_blocks:
            # Speculative FINAL alongside code — ignore it; the model
            # hasn't seen the code's output yet. Run the code and continue.
            joined = "\n\n".join(f"REPL output {j+1}:\n{o}" for j, o in enumerate(repl_outputs))
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        joined
                        + "\n\nNote: your message contained both code blocks and a FINAL(...). "
                        "FINAL was ignored because it was likely speculative. Review the REPL "
                        "output above, then either run more code or emit FINAL(...) / FINAL_VAR(name) on its own."
                    ),
                }
            )
            trace.append("--- speculative-final-ignored ---")
            continue

        if not code_blocks:
            # Nudge the model to either run code or finalize.
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": "You did not emit a ```repl block and did not provide FINAL(...). Either run code or output FINAL(...).",
                }
            )
            trace.append("--- nudge ---")
            continue

        joined = "\n\n".join(f"REPL output {j+1}:\n{o}" for j, o in enumerate(repl_outputs))
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": joined + "\n\nContinue, or emit FINAL(...) / FINAL_VAR(name)."})

    # Iteration cap reached.
    return RLMResult(
        answer="(no final answer — iteration cap reached)",
        n_iterations=max_iterations,
        n_subcalls=subcall_count[0],
        n_root_calls=root_client.stats.n_calls,
        prompt_tokens=root_client.stats.prompt_tokens,
        completion_tokens=root_client.stats.completion_tokens,
        trace=trace,
    )
