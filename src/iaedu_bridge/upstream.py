"""IAedu chatbot API client.

Verified against the live API: multipart POST with x-api-key; the response is
blank-line separated JSON objects of type start, token, message, error, done. Only
"token" carries deltas ("message" repeats the full text). Errors such as
"Rate limit reached (429)" arrive as an `error` event inside an HTTP 200.
"""

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from .config import Settings

log = logging.getLogger("iaedu_bridge")


class UpstreamError(Exception):
    """IAedu failed. `status` is the HTTP status the bridge answers with."""

    def __init__(self, message: str, status: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def flatten_messages(messages: list[dict]) -> str:
    """The IAedu API takes one message: fold the OpenAI history into it."""
    if len(messages) == 1:
        return _text(messages[0])
    return "\n\n".join(f"{m.get('role', 'user')}: {_text(m)}" for m in messages)


def _text(message: dict) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return content if isinstance(content, str) else str(content)


def clean(text: str) -> str:
    """Lone surrogates (valid in JSON, not in UTF-8) would crash multipart encoding."""
    return text.encode("utf-8", "replace").decode("utf-8")


def build_request(settings: Settings, text: str, thread_id: str) -> dict:
    return {
        "headers": {"x-api-key": settings.api_key},
        "fields": {
            "channel_id": settings.channel_id,
            "thread_id": thread_id,
            "user_info": "{}",
            "message": clean(text),
        },
    }


def classify_error(content: str) -> UpstreamError:
    low = content.lower()
    if "429" in low or "rate limit" in low:
        return UpstreamError(f"IAedu error: {content}", 429, retryable=True)
    if any(w in low for w in ("timeout", "timed out", "unavailable", "overloaded", "500", "502", "503")):
        return UpstreamError(f"IAedu error: {content}", 502, retryable=True)
    return UpstreamError(f"IAedu error: {content}", 502)


def parse_event(line: str) -> tuple[str, str]:
    """Return (kind, text) of one streamed line; ("", "") for blank or garbled lines."""
    line = line.strip()
    if not line:
        return "", ""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return "", ""  # truncated or garbled line: skip it, never leak raw protocol text
    if not isinstance(obj, dict):
        return "", ""
    kind, content = obj.get("type"), obj.get("content")
    if kind == "token":
        return "token", content if isinstance(content, str) else ""
    if kind == "error":
        raise classify_error(str(content)[:500])
    return ("done", "") if kind == "done" else ("other", "")


def parse_chunk(line: str) -> str:
    """Text delta of one streamed line ('' if none); raises on an error event."""
    kind, text = parse_event(line)
    return text if kind == "token" else ""


class Upstream:
    """Rate-limit aware client: concurrency cap, retry with backoff before any output."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings, self.client, self.sleep = settings, client, sleep
        self.gate = asyncio.Semaphore(max(1, settings.max_concurrency))

    async def _wait(self, attempt: int) -> None:
        delay = self.settings.backoff * (2 ** (attempt - 1))
        await self.sleep(delay + random.uniform(0, delay / 4))

    async def stream(self, text: str, thread_id: str) -> AsyncIterator[str]:
        """Yield deltas as they arrive; retry only while nothing was yielded."""
        attempt = 0
        while True:
            emitted = False
            try:
                async with self.gate:
                    async for delta in self._once(text, thread_id):
                        emitted = True
                        yield delta
                return
            except UpstreamError as exc:
                if emitted or not exc.retryable or attempt >= self.settings.retries:
                    raise
                why = str(exc)
            attempt += 1
            log.warning("retry %d/%d after: %s", attempt, self.settings.retries, why)
            await self._wait(attempt)

    async def collect(self, text: str, thread_id: str) -> str:
        """Whole reply; retries even after a partial one since nothing was sent yet."""
        attempt = 0
        while True:
            try:
                async with self.gate:
                    return "".join([d async for d in self._once(text, thread_id)])
            except UpstreamError as exc:
                if not exc.retryable or attempt >= self.settings.retries:
                    raise
                why = str(exc)
            attempt += 1
            log.warning("retry %d/%d after: %s", attempt, self.settings.retries, why)
            await self._wait(attempt)

    async def _once(self, text: str, thread_id: str) -> AsyncIterator[str]:
        req = build_request(self.settings, text, thread_id)
        files = {k: (None, v) for k, v in req["fields"].items()}
        tokens, done = 0, False
        try:
            async with self.client.stream(
                "POST", self.settings.endpoint, headers=req["headers"], files=files
            ) as resp:
                code = resp.status_code
                if code in (401, 403):
                    raise UpstreamError(f"IAedu rejected the API key ({code})", 502)
                if code == 429 or code >= 500:
                    raise UpstreamError(f"IAedu HTTP {code}", 429 if code == 429 else 502, True)
                if code >= 400:
                    raise UpstreamError(f"IAedu HTTP {code}", 502)
                async for line in resp.aiter_lines():
                    kind, delta = parse_event(line)
                    if kind == "done":
                        done = True
                    elif delta:
                        tokens += 1
                        yield delta
        except httpx.TimeoutException as exc:
            raise UpstreamError(f"IAedu timeout: {exc!r}", 504, True) from exc
        except httpx.TransportError as exc:
            raise UpstreamError(f"IAedu connection error: {exc!r}", 502, True) from exc
        if not tokens:
            raise UpstreamError("IAedu returned an empty reply", 502, True)
        if not done:
            raise UpstreamError("IAedu stream ended before 'done'", 502, True)
