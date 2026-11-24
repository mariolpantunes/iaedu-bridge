import json
import unittest

import httpx
from fastapi.testclient import TestClient

from iaedu_bridge.app import create_app
from iaedu_bridge.config import Settings
from iaedu_bridge.upstream import parse_chunk

S = Settings(endpoint="http://up/x", api_key="k", channel_id="c", bridge_key="b", backoff=0)
H = {"Authorization": "Bearer b"}
BODY = {"messages": [{"role": "user", "content": "hi"}]}
UPSTREAM = "\n\n".join(
    json.dumps(o)
    for o in [
        {"type": "start", "content": "Processing"},
        {"type": "token", "content": "Hel"},
        {"type": "token", "content": "lo"},
        {"type": "message", "content": {"type": "ai", "content": "Hello"}},
        {"type": "done", "content": "id"},
    ]
)


SENT: list[bytes] = []


def fake_upstream(request: httpx.Request) -> httpx.Response:
    SENT.append(request.content)
    assert request.headers["x-api-key"] == "k"
    assert b'name="channel_id"' in request.content
    assert b'name="user_info"' in request.content
    return httpx.Response(200, text=UPSTREAM)


class BridgeTest(unittest.TestCase):
    def setUp(self):
        SENT.clear()
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))
        self.client = TestClient(create_app(S, http))

    def test_models(self):
        r = self.client.get("/v1/models", headers=H)
        self.assertEqual(r.json()["data"][0]["id"], "iaedu-5.5")

    def test_auth(self):
        self.assertEqual(self.client.get("/v1/models").status_code, 401)

    def test_empty_messages(self):
        r = self.client.post("/v1/chat/completions", headers=H, json={"messages": []})
        self.assertEqual(r.status_code, 400)

    def test_completion_ignores_message_event(self):
        r = self.client.post("/v1/chat/completions", headers=H, json=BODY)
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "Hello")

    def test_stream(self):
        r = self.client.post("/v1/chat/completions", headers=H, json={**BODY, "stream": True})
        self.assertEqual(r.text.count('"content"'), 2)
        self.assertTrue(r.text.rstrip().endswith("[DONE]"))

    def test_usage_non_stream(self):
        u = self.client.post("/v1/chat/completions", headers=H, json=BODY).json()["usage"]
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertGreater(u["completion_tokens"], 0)
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def test_usage_stream_option(self):
        body = {**BODY, "stream": True, "stream_options": {"include_usage": True}}
        r = self.client.post("/v1/chat/completions", headers=H, json=body)
        last = [x for x in r.text.split("\n\n") if x.startswith("data: {")][-1]
        self.assertGreater(json.loads(last[6:])["usage"]["total_tokens"], 0)

    def test_parse_chunk(self):
        self.assertEqual(parse_chunk(""), "")
        self.assertEqual(parse_chunk('{"type":"start","content":"x"}'), "")


def field(raw: bytes, name: str) -> str:
    part = raw.decode().split(f'name="{name}"\r\n\r\n', 1)[1]
    return part.split("\r\n--", 1)[0]


class ThreadTest(unittest.TestCase):
    def setUp(self):
        SENT.clear()
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))
        self.client = TestClient(create_app(S, http))

    def ask(self, messages, **extra):
        r = self.client.post("/v1/chat/completions", headers=H, json={"messages": messages, **extra})
        self.assertEqual(r.status_code, 200)

    def test_continuation_reuses_thread_and_sends_only_new_message(self):
        u1 = {"role": "user", "content": "first question"}
        a1 = {"role": "assistant", "content": "Hello"}
        u2 = {"role": "user", "content": "second question"}
        self.ask([u1])
        self.ask([u1, a1, u2])
        self.assertEqual(field(SENT[0], "thread_id"), field(SENT[1], "thread_id"))
        self.assertEqual(field(SENT[1], "message"), "second question")

    def test_edited_history_starts_new_thread_with_full_history(self):
        u1 = {"role": "user", "content": "first question"}
        a1 = {"role": "assistant", "content": "Hello"}
        self.ask([u1, a1, {"role": "user", "content": "q2"}])
        self.ask([u1, a1, {"role": "user", "content": "q2 edited"}, a1, {"role": "user", "content": "q3"}])
        self.assertNotEqual(field(SENT[0], "thread_id"), field(SENT[1], "thread_id"))
        self.assertIn("q2 edited", field(SENT[1], "message"))

    def test_other_user_gets_other_thread(self):
        m = [{"role": "user", "content": "same"}]
        self.ask(m, user="a")
        self.ask(m, user="b")
        self.assertNotEqual(field(SENT[0], "thread_id"), field(SENT[1], "thread_id"))

    def test_failed_request_is_not_committed(self):
        def boom(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="x")

        http = httpx.AsyncClient(transport=httpx.MockTransport(boom))

        async def nosleep(_):
            pass

        c = TestClient(create_app(S, http, sleep=nosleep), raise_server_exceptions=False)
        m = [{"role": "user", "content": "q"}]
        self.assertEqual(c.post("/v1/chat/completions", headers=H, json={"messages": m}).status_code, 502)
        c2 = TestClient(create_app(S, httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))))
        c2.post("/v1/chat/completions", headers=H, json={"messages": m})
        self.assertEqual(field(SENT[0], "message"), "q")


if __name__ == "__main__":
    unittest.main()
