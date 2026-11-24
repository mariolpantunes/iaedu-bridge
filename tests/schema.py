"""Validators for OpenAI chat-completion responses and SSE streams."""

import json


def check_completion(obj: dict) -> None:
    assert obj["object"] == "chat.completion", obj
    assert isinstance(obj["id"], str) and obj["id"].startswith("chatcmpl-")
    assert isinstance(obj["created"], int) and isinstance(obj["model"], str)
    (choice,) = obj["choices"]
    assert choice["index"] == 0
    msg = choice["message"]
    assert msg["role"] == "assistant"
    if choice["finish_reason"] == "tool_calls":
        assert msg["content"] is None and msg["tool_calls"]
        for c in msg["tool_calls"]:
            assert c["type"] == "function" and c["id"].startswith("call_")
            assert isinstance(c["function"]["name"], str)
            json.loads(c["function"]["arguments"])  # must be a JSON string
    else:
        assert choice["finish_reason"] == "stop"
        assert isinstance(msg["content"], str)
    u = obj["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["prompt_tokens"] > 0


def parse_sse(text: str) -> list[dict]:
    frames = [f for f in text.split("\n\n") if f.strip()]
    assert frames and frames[-1] == "data: [DONE]", frames[-2:]
    return [json.loads(f[6:]) for f in frames[:-1] if f.startswith("data: ")]


def check_stream(text: str) -> dict:
    """Validate an SSE body; return {"content", "tool_calls", "finish", "usage"}."""
    chunks = parse_sse(text)
    if chunks and "error" in chunks[-1]:
        return {"error": chunks[-1]["error"]["message"]}
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    content, calls, finishes, usage = "", {}, [], None
    for c in chunks:
        assert c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"]
        if "usage" in c:
            usage = c["usage"]
            continue
        ch = c["choices"][0]
        d = ch["delta"]
        content += d.get("content") or ""
        for tc in d.get("tool_calls") or []:
            calls[tc["index"]] = tc
        if ch["finish_reason"]:
            finishes.append(ch["finish_reason"])
    assert len(finishes) == 1 and finishes[0] in ("stop", "tool_calls"), finishes
    for tc in calls.values():
        json.loads(tc["function"]["arguments"])
    return {"content": content, "tool_calls": list(calls.values()), "finish": finishes[0], "usage": usage}
