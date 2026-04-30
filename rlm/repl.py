"""Persistent Python REPL with stdout capture + truncation.

The REPL holds a single globals dict that persists across turns, so
intermediate variables (including the user's input as `context`) survive
between LLM iterations.
"""
from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from typing import Any


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = limit // 2
    tail = limit - head - 80
    return f"{s[:head]}\n...[truncated {len(s) - head - tail} chars]...\n{s[-tail:]}"


class Repl:
    def __init__(self, initial_globals: dict[str, Any], stdout_limit: int = 4000):
        self.globals: dict[str, Any] = dict(initial_globals)
        self.globals.setdefault("__builtins__", __builtins__)
        self.stdout_limit = stdout_limit

    def exec(self, code: str) -> str:
        """Execute code, return truncated stdout (or traceback on error)."""
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.globals)
            out = buf.getvalue()
        except Exception:
            out = buf.getvalue() + "\n--- TRACEBACK ---\n" + traceback.format_exc()
        if not out.strip():
            out = "(no output)"
        return _truncate(out, self.stdout_limit)
