"""FastAPI app exposing /v1/models and /v1/chat/completions (OpenAI format)."""

import asyncio
import hmac
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .threads import ThreadStore
from .tokens import usage as count_usage
from .tools import active_tools, parse_reply
from .upstream import Upstream, UpstreamError

log = logging.getLogger("iaedu_bridge")


def _err(message: str, status: int, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def validate(body: object) -> list[dict]:
    """Return the messages of a chat request or raise HTTPException(400)."""
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise HTTPException(400, "each message needs a string role")
    if all(m["role"] == "assistant" for m in messages):
        raise HTTPException(400, "at least one non-assistant message is required")
    return messages


def create_app(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FastAPI:
    app = FastAPI(title="iaedu-bridge")
    http = client or httpx.AsyncClient(timeout=settings.timeout)
    up = Upstream(settings, http, sleep)
    threads = ThreadStore()

    def auth(authorization: str = Header(default="")) -> None:
        want = f"Bearer {settings.bridge_key}"
        if settings.bridge_key and not hmac.compare_digest(authorization, want):
            raise HTTPException(401, "invalid API key")

    @app.exception_handler(UpstreamError)
    async def upstream_error(_, exc: UpstreamError):
        log.error("upstream error status=%d: %s", exc.status, exc)
        return _err(str(exc), exc.status, "upstream_error")

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": settings.model, "object": "model", "owned_by": "iaedu"}],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat(body: dict):
        messages = validate(body)
        tools = active_tools(body)
        plan = threads.plan(messages, str(body.get("user") or ""), tools)
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        t0 = time.monotonic()
        log.info(
            "chat thread=%s sent=%d chars history=%d msgs tools=%d stream=%s new_thread=%s",
            plan.thread_id[:8],
            len(plan.text),
            len(messages),
            len(tools),
            body.get("stream") is True,
            plan.start == 0,
        )

        def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                obj["choices"] = []
                obj["usage"] = usage
            return f"data: {json.dumps(obj)}\n\n"

        async def collect() -> str:
            text = await up.collect(plan.text, plan.thread_id)
            threads.commit(plan)
            return text

        if body.get("stream") is True:
            # with tools the reply must be parsed as JSON before anything is sent;
            # collecting first also lets upstream errors become a proper HTTP error
            reply, agen, first = "", None, ""
            if tools:
                reply = await collect()
            else:  # wait for the first token so an early failure is a real HTTP error
                agen = up.stream(plan.text, plan.thread_id)
                first = await anext(agen)

            async def events():
                yield chunk({"role": "assistant"})
                finish, out = "stop", ""
                try:
                    if tools:
                        kind, val = parse_reply(reply, tools)
                        if kind == "tool_calls":
                            calls = [{"index": i, **c} for i, c in enumerate(val)]
                            yield chunk({"tool_calls": calls})
                            finish, out = "tool_calls", json.dumps(val)
                        else:
                            yield chunk({"content": val})
                            out = val
                    else:
                        assert agen is not None
                        parts = [first]
                        yield chunk({"content": first})
                        async for text in agen:
                            parts.append(text)
                            yield chunk({"content": text})
                        threads.commit(plan)
                        out = "".join(parts)
                except UpstreamError as exc:  # headers are sent: report in-band
                    err = {"error": {"message": str(exc), "type": "upstream_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                log.info(
                    "chat done thread=%s secs=%.1f reply=%d chars",
                    plan.thread_id[:8],
                    time.monotonic() - t0,
                    len(out),
                )
                yield chunk({}, finish)
                opts = body.get("stream_options")
                if isinstance(opts, dict) and opts.get("include_usage"):
                    yield chunk({}, usage=count_usage(messages, out))
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        text = await collect()
        log.info(
            "chat done thread=%s secs=%.1f reply=%d chars",
            plan.thread_id[:8],
            time.monotonic() - t0,
            len(text),
        )
        message: dict = {"role": "assistant", "content": text}
        finish = "stop"
        if tools:
            kind, val = parse_reply(text, tools)
            if kind == "tool_calls":
                message = {"role": "assistant", "content": None, "tool_calls": val}
                finish, text = "tool_calls", json.dumps(val)
            else:
                message["content"] = text = val
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": settings.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": count_usage(messages, text),
            }
        )

    return app
