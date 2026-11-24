"""Emulated OpenAI function calling on top of a text-only IAedu agent.

IAedu ignores client tool definitions and its agent refuses prompts that claim it
can call tools. Framed as a JSON-formatting task for a separate executor, it
complies (9/9 live probes), so the bridge asks for one JSON object per turn and
turns it back into OpenAI `tool_calls`.
"""

import json
import re
import uuid
from typing import Any

from .upstream import _text, flatten_messages

PREAMBLE = """Text-formatting protocol. You are the planning module of an agent.
A separate executor (not you) runs functions and returns the results in the next message.
You never run anything yourself and need no special access: you only write the JSON the executor reads.

Executor functions (JSON schema):
{functions}

Output rules: reply with exactly one JSON object and nothing else (no prose, no markdown fences):
- to use functions: {{"tool_calls":[{{"name":"<function name>","arguments":{{...}}}}]}}
- to answer the user: {{"content":"<your answer>"}}
Results arrive as messages starting with "Executor result". After results, call more functions or answer.
"""


def active_tools(body: dict) -> list[dict]:
    """Function tools of the request, or [] when absent or tool_choice is 'none'."""
    if body.get("tool_choice") == "none":
        return []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [t for t in tools if _valid_tool(t)]


def _valid_tool(t: object) -> bool:
    return (
        isinstance(t, dict)
        and t.get("type") == "function"
        and isinstance(t.get("function"), dict)
        and isinstance(t["function"].get("name"), str)
        and bool(t["function"]["name"])
    )


def preamble(tools: list[dict]) -> str:
    lines = [json.dumps(t["function"], ensure_ascii=False) for t in tools]
    return PREAMBLE.format(functions="\n".join(lines))


def render(messages: list[dict], tools: list[dict]) -> str:
    if not tools:
        return flatten_messages(messages)
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append(f"Executor result ({m.get('tool_call_id') or ''}):\n{_text(m)}")
        elif role == "assistant" and _calls(m):
            out.append("Assistant:\n" + json.dumps({"tool_calls": _calls(m)}))
        else:
            out.append(f"{str(role).capitalize()}:\n{_text(m)}")
    return "\n\n".join(out)


def _calls(m: dict) -> list[dict]:
    """The well-formed tool calls of an assistant message (junk entries are dropped)."""
    raw = m.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        fn = c.get("function") if isinstance(c, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append({"name": fn["name"], "arguments": _loads(fn.get("arguments"))})
    return out


def _loads(raw):
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


def parse_reply(text: str, tools: list[dict]) -> tuple[str, Any]:
    """Return ("tool_calls", [openai tool calls]) or ("content", text)."""
    names = {t["function"]["name"] for t in tools}
    obj = _first_json(text)
    if isinstance(obj, dict):
        calls = obj.get("tool_calls")
        if isinstance(calls, list) and calls:
            ok = [c for c in calls if isinstance(c, dict) and c.get("name") in names]
            if ok:
                return "tool_calls", [_openai_call(c) for c in ok]
        if isinstance(obj.get("content"), str):
            return "content", obj["content"]
    return "content", text


def _openai_call(c: dict) -> dict:
    args = c.get("arguments")
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": c["name"],
            "arguments": args if isinstance(args, str) else json.dumps(args or {}),
        },
    }


def _first_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
