"""LLM clients. The copilot depends only on `complete(system, user) -> str`."""
from __future__ import annotations

import os
from typing import Callable, Protocol

import httpx


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Claude via the Messages API. On Databricks the same interface can wrap a Model Serving endpoint."""

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model or os.environ.get("PAYLAKE_LLM_MODEL", "claude-sonnet-5-5")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 800, "temperature": 0, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


class StubLLM:
    """Deterministic stand-in for tests: maps a question to a canned response."""

    model = "stub"

    def __init__(self, responder: Callable[[str], str]):
        self.responder = responder

    def complete(self, system: str, user: str) -> str:
        return self.responder(user)
