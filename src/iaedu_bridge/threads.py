"""Map OpenAI chat histories to IAedu threads so IAedu keeps the history.

A conversation key is sha256(user, system prompt, first user message); the thread id
is sha256(key, start time) so a restarted or edited conversation gets a fresh thread.
The bridge remembers how many non-assistant messages IAedu has already seen and a
digest of them: when the next request extends that prefix, only the new messages
are sent; otherwise a new thread starts and the whole history is sent.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

from .tools import preamble, render
from .upstream import _text, flatten_messages

MAX_THREADS = 1000


def _digest(messages: list[dict]) -> str:
    blob = json.dumps([(m.get("role"), m.get("content")) for m in messages], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:24]


@dataclass
class Plan:
    thread_id: str
    text: str
    key: str
    seen: int
    digest: str
    start: int
    tools: str = ""


class ThreadStore:
    def __init__(self, clock=time.time_ns) -> None:
        self._clock = clock
        self._state: OrderedDict[str, dict] = OrderedDict()

    def plan(self, messages: list[dict], user: str = "", tools: list[dict] | None = None) -> Plan:
        inputs = [m for m in messages if m.get("role") != "assistant"]
        system = "".join(_text(m) for m in inputs if m.get("role") == "system")
        first = next((m for m in inputs if m.get("role") == "user"), inputs[0])
        first_text = flatten_messages([first])
        key = _sha(user, system, first_text)
        tools = tools or []
        tdig = _sha(json.dumps(tools, sort_keys=True)) if tools else ""
        head = preamble(tools) + "\n\n" if tools else ""
        st = self._state.get(key)
        if st and len(inputs) > st["seen"] and _digest(inputs[: st["seen"]]) == st["digest"]:
            text = render(inputs[st["seen"] :], tools)
            if tools and st.get("tools") != tdig:
                text = head + text
            return Plan(st["thread_id"], text, key, len(inputs), _digest(inputs), st["seen"], tdig)
        thread_id = _sha(key, str(self._clock()))
        return Plan(thread_id, head + render(messages, tools), key, len(inputs), _digest(inputs), 0, tdig)

    def commit(self, plan: Plan) -> None:
        """Call after IAedu answered: only then is the thread known to hold the turn."""
        self._state[plan.key] = {
            "thread_id": plan.thread_id,
            "seen": plan.seen,
            "digest": plan.digest,
            "tools": plan.tools,
        }
        self._state.move_to_end(plan.key)
        while len(self._state) > MAX_THREADS:
            self._state.popitem(last=False)
