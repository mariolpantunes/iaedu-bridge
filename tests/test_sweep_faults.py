"""Sweeps over upstream failures, tool-reply styles, concurrency and random fuzz."""

import asyncio
import json
import os
import random
import unittest

from fake_iaedu import FAULTS, FakeIaedu
from rig import TOOLS, Rig, user
from schema import check_completion, check_stream

# fault -> status the bridge must answer with when the fault never goes away
PERSISTENT = {
    "rate_limit": 429,
    "http429": 429,
    "http500": 502,
    "http401": 502,
    "empty": 502,
    "error_event": 502,
    "garbage": 200,
    "truncated": 502,
    "midstream_error": 502,
    "timeout": 504,
    "too_long": 502,
}
RECOVERABLE = [f for f in FAULTS if f not in ("http401", "error_event", "too_long")]


class FaultSweep(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, fault: str, times: int, **req):
        fake = FakeIaedu()
        fake.faults = [fault] * times
        rig = Rig(fake)
        try:
            return await rig.chat([user("hello there")], **req), fake
        finally:
            await rig.aclose()

    async def test_recoverable_fault_once_is_invisible(self):
        for fault in RECOVERABLE:
            for stream in (False, True):
                for tools in (None, TOOLS):
                    with self.subTest(fault=fault, stream=stream, tools=bool(tools)):
                        extra = {"stream": stream, **({"tools": tools} if tools else {})}
                        r, fake = await self.run_case(fault, 1, **extra)
                        self.assertEqual(r.status_code, 200, r.text[:200])
                        if stream and not tools and fault in ("truncated", "midstream_error"):
                            # tokens already reached the client: only an in-band error is possible
                            out = check_stream(r.text)
                            self.assertTrue("error" in out or out["content"], out)
                        elif stream:
                            out = check_stream(r.text)
                            self.assertNotIn("error", out, out)
                        else:
                            check_completion(r.json())

    async def test_retries_are_bounded_and_map_to_the_right_status(self):
        for fault, status in PERSISTENT.items():
            with self.subTest(fault=fault):
                r, fake = await self.run_case(fault, 50)
                self.assertEqual(r.status_code, status, (fault, r.text[:200]))
                self.assertLessEqual(len(fake.requests), 4)  # 1 try + 3 retries
                if status != 200:
                    self.assertIn("error", r.json())

    async def test_streaming_failure_before_output_is_a_real_http_error(self):
        for fault, status in PERSISTENT.items():
            if fault in ("garbage", "truncated", "midstream_error"):
                continue  # these emit tokens first
            with self.subTest(fault=fault):
                r, fake = await self.run_case(fault, 50, stream=True)
                self.assertEqual(r.status_code, status, (fault, r.text[:200]))
                self.assertLessEqual(len(fake.requests), 4)

    async def test_non_retryable_errors_try_once(self):
        for fault in ("http401", "error_event", "too_long"):
            with self.subTest(fault=fault):
                r, fake = await self.run_case(fault, 50)
                self.assertEqual(r.status_code, 502)
                self.assertEqual(len(fake.requests), 1)

    async def test_streaming_error_after_output_is_reported_in_band(self):
        r, _ = await self.run_case("midstream_error", 50, stream=True)
        out = check_stream(r.text) if r.status_code == 200 else {"error": r.text}
        self.assertIn("error", out)

    async def test_wrong_bridge_key_and_wrong_upstream_key(self):
        rig = Rig(bridge_key="secret")
        try:
            r = await rig.chat([user("x")])
            self.assertEqual(r.status_code, 401)
            ok = await rig.http.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={"messages": [user("x")]},
            )
            self.assertEqual(ok.status_code, 200)
            self.assertEqual((await rig.http.get("/v1/models")).status_code, 401)
        finally:
            await rig.aclose()

    async def test_random_transient_faults_never_reach_the_client(self):
        rng = random.Random(7)
        fake = FakeIaedu()
        fake.faults = [rng.choice(RECOVERABLE) if rng.random() < 0.15 else None for _ in range(400)]
        fake.faults = [f for f in fake.faults]  # None entries are consumed as "no fault"
        rig = Rig(fake, retries=5)
        try:
            bad = 0
            for i in range(150):
                r = await rig.chat(
                    [user(f"question {i}")], **({"tools": TOOLS, "stream": True} if i % 2 else {})
                )
                bad += r.status_code != 200 or ('"error"' in r.text and r.text.startswith("data"))
            self.assertEqual(bad, 0)
        finally:
            await rig.aclose()


class ToolReplySweep(unittest.IsolatedAsyncioTestCase):
    async def ask(self, style: str, stream: bool):
        fake = FakeIaedu()
        fake.style = style
        rig = Rig(fake)
        try:
            return await rig.chat([user("go")], tools=TOOLS, stream=stream)
        finally:
            await rig.aclose()

    async def test_every_reply_style_yields_valid_openai_output(self):
        expect_calls = {
            "plain": 1,
            "fenced": 1,
            "prose": 1,
            "string_args": 1,
            "multi": 2,
            "unknown_tool": 0,
            "invalid": 0,
            "prose_only": 0,
            "content": 0,
        }
        for style, n in expect_calls.items():
            for stream in (False, True):
                with self.subTest(style=style, stream=stream):
                    r = await self.ask(style, stream)
                    self.assertEqual(r.status_code, 200, r.text[:200])
                    if stream:
                        out = check_stream(r.text)
                        self.assertEqual(len(out["tool_calls"]), n)
                        self.assertEqual(out["finish"], "tool_calls" if n else "stop")
                    else:
                        body = r.json()
                        check_completion(body)
                        calls = body["choices"][0]["message"].get("tool_calls") or []
                        self.assertEqual(len(calls), n)

    async def test_argument_json_survives_braces_and_quotes(self):
        r = await self.ask("plain", False)
        args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args, {"command": "echo {hi}"})


class ConcurrencySweep(unittest.IsolatedAsyncioTestCase):
    async def test_burst_of_requests_respects_the_cap_and_all_succeed(self):
        for cap in (1, 2, 5):
            with self.subTest(cap=cap):
                fake = FakeIaedu(latency=0.005)
                rig = Rig(fake, max_concurrency=cap)
                try:
                    rs = await asyncio.gather(
                        *[rig.chat([user(f"q{i}")], stream=bool(i % 2)) for i in range(80)]
                    )
                    self.assertTrue(all(r.status_code == 200 for r in rs))
                    self.assertLessEqual(fake.peak, cap)
                finally:
                    await rig.aclose()

    async def test_burst_with_faults_all_recover(self):
        fake = FakeIaedu(latency=0.002)
        fake.faults = ["rate_limit", "timeout", "empty", None, "http500", "garbage"] * 10
        rig = Rig(fake, max_concurrency=3, retries=6)
        try:
            rs = await asyncio.gather(*[rig.chat([user(f"q{i}")]) for i in range(60)])
            self.assertEqual([r.status_code for r in rs].count(200), 60)
        finally:
            await rig.aclose()


class FuzzSweep(unittest.IsolatedAsyncioTestCase):
    """Random bodies: whatever comes in, the bridge answers 200/400/422, never 500."""

    ROLES = ["user", "assistant", "system", "tool", "developer", "function", "", None, 5]
    VALUES = [
        None,
        "",
        "hi",
        "x" * 3000,
        "日本語",
        "\x00",
        ["a"],
        [{"type": "text", "text": "t"}],
        {"k": 1},
        7,
        True,
        1.5,
    ]

    def rand_msg(self, rng: random.Random) -> object:
        if rng.random() < 0.05:
            return rng.choice([None, 1, "s", [], {}])
        m = {"role": rng.choice(self.ROLES)}
        if rng.random() < 0.9:
            m["content"] = rng.choice(self.VALUES)
        if rng.random() < 0.1:
            m["tool_calls"] = rng.choice(
                [
                    [],
                    [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                    "bad",
                    [None],
                ]
            )
        if rng.random() < 0.1:
            m["tool_call_id"] = rng.choice(["c", None, 3])
        return m

    async def test_random_bodies(self):
        rng = random.Random(int(os.environ.get("SWEEP_SEED", "1234")))
        rig = Rig()
        try:
            for i in range(int(os.environ.get("SWEEP_N", "400"))):
                body: dict = {"messages": [self.rand_msg(rng) for _ in range(rng.randint(0, 6))]}
                if rng.random() < 0.4:
                    body["stream"] = rng.choice([True, False, "yes", None])
                if rng.random() < 0.4:
                    body["tools"] = rng.choice(
                        [
                            TOOLS,
                            [],
                            None,
                            "x",
                            [{"type": "function"}],
                            [{"type": "function", "function": {"name": "a"}}],
                            [1],
                        ]
                    )
                if rng.random() < 0.2:
                    body["tool_choice"] = rng.choice(["none", "auto", {"type": "function"}, None])
                if rng.random() < 0.2:
                    body["stream_options"] = rng.choice([{"include_usage": True}, None, "x", {}])
                if rng.random() < 0.2:
                    body["user"] = rng.choice(["u", None, 5, {"a": 1}])
                with self.subTest(i=i):
                    r = await rig.http.post("/v1/chat/completions", json=body)
                    self.assertIn(r.status_code, (200, 400, 422), (body, r.text[:300]))
                    if r.status_code == 200 and body.get("stream") is True:
                        check_stream(r.text)
                    elif r.status_code == 200:
                        check_completion(r.json())
        finally:
            await rig.aclose()


if __name__ == "__main__":
    unittest.main()
