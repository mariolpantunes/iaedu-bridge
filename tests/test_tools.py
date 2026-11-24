import json
import unittest

import httpx
from fastapi.testclient import TestClient

from iaedu_bridge.app import create_app
from iaedu_bridge.config import Settings
from iaedu_bridge.tools import parse_reply, render

S = Settings(endpoint="http://up/x", api_key="k", channel_id="c", backoff=0)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]
CALL = '{"tool_calls":[{"name":"run_shell","arguments":{"command":"lscpu"}}]}'


def sse(reply: str) -> str:
    evs = [{"type": "token", "content": reply[i : i + 7]} for i in range(0, len(reply), 7)]
    return "\n\n".join(json.dumps(e) for e in evs + [{"type": "done", "content": "x"}])


class ToolTest(unittest.TestCase):
    def setUp(self):
        self.sent: list[str] = []
        self.reply = CALL

        def upstream(request: httpx.Request) -> httpx.Response:
            self.sent.append(request.content.decode())
            return httpx.Response(200, text=sse(self.reply))

        http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.client = TestClient(create_app(S, http))

    def post(self, messages, **extra):
        return self.client.post("/v1/chat/completions", json={"messages": messages, "tools": TOOLS, **extra})

    def test_tool_call_non_stream(self):
        msg = self.post([{"role": "user", "content": "cpu?"}]).json()["choices"][0]
        self.assertEqual(msg["finish_reason"], "tool_calls")
        call = msg["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "run_shell")
        self.assertEqual(json.loads(call["arguments"]), {"command": "lscpu"})
        self.assertIn("Executor functions", self.sent[0])

    def test_tool_call_stream(self):
        r = self.post([{"role": "user", "content": "cpu?"}], stream=True)
        events = [json.loads(x[6:]) for x in r.text.split("\n\n") if x.startswith("data: {")]
        calls = [e["choices"][0]["delta"] for e in events if "tool_calls" in e["choices"][0]["delta"]]
        self.assertEqual(calls[0]["tool_calls"][0]["function"]["name"], "run_shell")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_result_turn_sends_only_new_messages(self):
        u = {"role": "user", "content": "cpu?"}
        first = self.post([u]).json()["choices"][0]["message"]
        self.reply = '{"content":"It is a Ryzen."}'
        tool = {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "Ryzen 7"}
        r = self.post([u, first, tool]).json()["choices"][0]
        self.assertEqual(r["message"]["content"], "It is a Ryzen.")
        self.assertEqual(r["finish_reason"], "stop")
        self.assertIn("Executor result", self.sent[1])
        self.assertIn("Ryzen 7", self.sent[1])
        self.assertNotIn("Executor functions", self.sent[1])  # preamble only once per thread

    def test_plain_text_reply_passes_through(self):
        self.reply = "Just prose."
        m = self.post([{"role": "user", "content": "hi"}]).json()["choices"][0]["message"]
        self.assertEqual(m["content"], "Just prose.")

    def test_tool_choice_none_disables_emulation(self):
        self.post([{"role": "user", "content": "hi"}], tool_choice="none")
        self.assertNotIn("Executor functions", self.sent[0])

    def test_history_replay_renders_calls_and_survives_junk(self):

        hist = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "run_shell", "arguments": '{"command":"ls"}'},
                    },
                    {"id": "c2", "type": "function"},
                    "junk",
                    None,
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "files"},
        ]
        text = render(hist, TOOLS)
        self.assertIn('"name": "run_shell"', text)
        self.assertIn("Executor result (c1):\nfiles", text)

    def test_parse_reply(self):
        self.assertEqual(parse_reply("```json\n" + CALL + "\n```", TOOLS)[0], "tool_calls")
        self.assertEqual(parse_reply('{"tool_calls":[{"name":"nope","arguments":{}}]}', TOOLS)[0], "content")
        self.assertEqual(parse_reply('{"content":"x"}', TOOLS), ("content", "x"))


if __name__ == "__main__":
    unittest.main()
