"""In-process fake of the IAedu stream API with fault injection (no network).

Behaviours verified against the live API: multipart POST, JSON events separated by
blank lines (start, token, message, error, done), errors inside an HTTP 200, thread
history kept on the IAedu side. Use `FakeIaedu().transport()` as httpx transport.
"""

import asyncio
import json
import re

import httpx

FUNC_RE = re.compile(r'^\{"name": "([^"]+)"', re.M)
FAULTS = (
    "rate_limit",
    "http429",
    "http500",
    "http401",
    "empty",
    "error_event",
    "garbage",
    "truncated",
    "midstream_error",
    "timeout",
    "too_long",
)


class _BrokenStream(httpx.AsyncByteStream):
    def __init__(self, head: bytes) -> None:
        self.head = head

    async def __aiter__(self):
        yield self.head
        raise httpx.ReadError("connection reset")


def parse_form(request: httpx.Request) -> dict[str, str]:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    out: dict[str, str] = {}
    for part in request.content.split(b"--" + boundary):
        if b'name="' not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head).group(1).decode()  # type: ignore[union-attr]
        out[name] = (body[:-2] if body.endswith(b"\r\n") else body).decode()
    return out


def events(reply: str, run: str = "r1") -> str:
    ev = [{"run_id": run, "type": "start", "content": "Processing"}]
    ev += [{"run_id": run, "type": "token", "content": reply[i : i + 6]} for i in range(0, len(reply), 6)]
    ev += [
        {"run_id": run, "type": "message", "content": {"type": "ai", "content": reply}},
        {"run_id": run, "type": "done", "content": run},
    ]
    return "\n\n".join(json.dumps(e) for e in ev) + "\n\n"


class FakeIaedu:
    def __init__(self, latency: float = 0.0, max_chars: int | None = None) -> None:
        self.threads: dict[str, list[str]] = {}
        self.requests: list[dict[str, str]] = []
        self.faults: list[str] = []  # consumed one per request, first in first out
        self.latency, self.max_chars = latency, max_chars
        self.style = "plain"  # how the fake wraps tool-call JSON, see `_tool_reply`
        self.active = self.peak = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    async def handle(self, request: httpx.Request) -> httpx.Response:
        form = parse_form(request)
        self.requests.append(form)
        if request.headers.get("x-api-key") != "k":
            return httpx.Response(401, text="unauthorized")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
            return self._respond(form)
        finally:
            self.active -= 1

    def _respond(self, form: dict[str, str]) -> httpx.Response:
        fault = self.faults.pop(0) if self.faults else None
        msg = form["message"]
        if self.max_chars and len(msg) > self.max_chars:
            fault = "too_long"
        hist = self.threads.setdefault(form["thread_id"], [])
        if fault == "http429":
            return httpx.Response(429, text="slow down")
        if fault == "http401":
            return httpx.Response(401, text="unauthorized")
        if fault == "http500":
            return httpx.Response(500, text="boom")
        if fault == "timeout":
            raise httpx.ReadTimeout("timed out")
        err = {
            "rate_limit": "Rate limit reached (429)",
            "error_event": "Something exploded in the agent",
            "too_long": "Message too long",
        }
        if fault in err:
            return self._events_only("error", err[fault])
        if fault == "empty":
            return httpx.Response(200, text=self._raw(["start", "done"]))
        hist.append(msg)
        reply = self._reply(hist)
        body = events(reply)
        if fault == "garbage":
            body = "not json at all\n\n{broken\n\n[1,2]\n\n" + body
        if fault == "truncated":
            return httpx.Response(200, text=body[: len(body) // 2])
        if fault == "midstream_error":
            return httpx.Response(200, stream=_BrokenStream(body[: len(body) // 2].encode()))
        return httpx.Response(200, text=body)

    @staticmethod
    def _raw(kinds: list[str]) -> str:
        return "\n\n".join(json.dumps({"type": k, "content": "x"}) for k in kinds) + "\n\n"

    @staticmethod
    def _events_only(kind: str, content: str) -> httpx.Response:
        evs = [
            {"type": "start", "content": "Processing"},
            {"type": kind, "content": content},
            {"type": "done", "content": "x"},
        ]
        return httpx.Response(200, text="\n\n".join(json.dumps(e) for e in evs) + "\n\n")

    # ---- deterministic "model" -------------------------------------------------
    def _reply(self, hist: list[str]) -> str:
        last, first = hist[-1], hist[0]
        names = FUNC_RE.findall(first)
        if names and "Executor functions" in first:
            return self._tool_reply(last, names)
        m = re.search(r"codeword is (\S+?)[.\s]", " ".join(hist) + " ")
        if "codeword?" in last:
            return f"The codeword is {m.group(1)}" if m else "unknown"
        return f"ECHO[{len(hist)}] {last[:40]}"

    def _tool_reply(self, last: str, names: list[str]) -> str:
        if "Executor result (" in last:
            return json.dumps({"content": "done: " + last.split("):", 1)[-1].strip()[:40]})
        call = {"tool_calls": [{"name": names[0], "arguments": {"command": "echo {hi}"}}]}
        raw = json.dumps(call)
        return {
            "plain": raw,
            "fenced": f"```json\n{raw}\n```",
            "prose": f"Sure, here you go: {raw} Hope that helps!",
            "string_args": json.dumps({"tool_calls": [{"name": names[0], "arguments": '{"command":"ls"}'}]}),
            "multi": json.dumps({"tool_calls": [{"name": n, "arguments": {}} for n in names]}),
            "unknown_tool": json.dumps({"tool_calls": [{"name": "nope", "arguments": {}}]}),
            "invalid": '{"tool_calls": [{"name": ',
            "prose_only": "I cannot do that.",
            "content": json.dumps({"content": "plain answer"}),
        }[self.style]
