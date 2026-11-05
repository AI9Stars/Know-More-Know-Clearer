"""Thin wrapper around the vLLM OpenAI-compatible endpoint.

Provides the three things every stage needs and would otherwise reimplement:
completion/chat calls, retry-with-backoff, and tolerant JSON extraction from a
model reply that may be wrapped in prose or code fences.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .config import EndpointConfig

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JSONParseError(ValueError):
    """Raised when a reply cannot be coerced into a JSON object."""


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Tries, in order: the whole string, the contents of a `````json`` fence, and
    the widest ``{...}`` span. Raises :class:`JSONParseError` if all fail.
    """
    candidates: List[str] = [text.strip()]

    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JSONParseError(f"no JSON object found in reply: {text[:400]!r}")


class LLMClient:
    """A retrying client for one vLLM endpoint."""

    def __init__(self, endpoint: EndpointConfig):
        if not endpoint.model:
            raise ValueError(
                "endpoint.model is empty; set it in your config or export VLLM_MODEL"
            )
        self.endpoint = endpoint
        self._client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=endpoint.timeout,
        )

    # -- raw calls ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        logprobs: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Text completion. Returns ``{"text", "avg_surprisal"}``.

        ``avg_surprisal`` is the mean negative log-probability over generated
        tokens, or ``nan`` when ``logprobs`` is False or the server returns none.
        """
        resp = self._retry(
            lambda: self._client.completions.create(
                model=self.endpoint.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                seed=seed,
            )
        )
        choice = resp.choices[0]
        return {
            "text": (choice.text or "").strip(),
            "avg_surprisal": _mean_surprisal(getattr(choice, "logprobs", None)),
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        resp = self._retry(
            lambda: self._client.chat.completions.create(
                model=self.endpoint.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
        return (resp.choices[0].message.content or "").strip()

    # -- JSON-returning helpers -------------------------------------------

    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Completion whose reply is parsed as JSON, retrying on parse failure."""
        return self._json_with_retry(lambda: self.complete(prompt, **kwargs)["text"])

    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat whose reply is parsed as JSON, retrying on parse failure."""
        return self._json_with_retry(lambda: self.chat(messages, **kwargs))

    def _json_with_retry(self, call) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(1, self.endpoint.max_retries + 1):
            text = call()
            try:
                return extract_json(text)
            except JSONParseError as exc:
                last = exc
                log.debug("JSON parse failed (attempt %d/%d)", attempt, self.endpoint.max_retries)
                time.sleep(1.5 * attempt)
        raise JSONParseError(str(last))

    # -- transport retry ---------------------------------------------------

    def _retry(self, call):
        last: Optional[Exception] = None
        for attempt in range(1, self.endpoint.max_retries + 1):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last = exc
                log.warning(
                    "request failed (attempt %d/%d): %s",
                    attempt,
                    self.endpoint.max_retries,
                    exc,
                )
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"endpoint call failed after {self.endpoint.max_retries} attempts: {last}")


def _mean_surprisal(logprobs: Any) -> float:
    """Mean ``-logprob`` over returned tokens, tolerating both response shapes."""
    values: List[float] = []
    if logprobs is None:
        return float("nan")

    raw = None
    if hasattr(logprobs, "token_logprobs"):
        raw = logprobs.token_logprobs
    elif isinstance(logprobs, dict):
        raw = logprobs.get("token_logprobs")

    if raw:
        values = [float(x) for x in raw if x is not None]

    return float(-np.mean(values)) if values else float("nan")
