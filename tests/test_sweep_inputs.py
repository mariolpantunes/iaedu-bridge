"""Sweeps over request sizes, text content, message shapes and long conversations."""

import unittest

from rig import TOOLS, Rig, user
from schema import check_completion, check_stream

SIZES = [1, 10, 1_000, 10_000, 100_000, 400_000]
TEXTS = {
    "emoji": "hello 👋🏽 世界 🇵🇹",
    "rtl": "مرحبا بالعالم שלום",
    "accents": "ação coração à é ü ñ ß",
    "newlines": "line1\nline2\r\nline3\n\n\n",
    "control": "a\x00b\x01c\x1bd\x7fe",
    "quotes": "He said \"hi\" and 'bye' \\ back\\slash",
    "json_like": '{"tool_calls":[{"name":"x"}]} {"content":"y"}',
    "boundary_like": "--boundary\r\nContent-Disposition: form-data",
    "markdown": "# T\n```python\nprint('x')\n```\n| a | b |",
    "long_word": "x" * 5000,
    "spaces": "   ",
    "tab_only": "\t",
}


class InputSweep(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rig = Rig()

    async def asyncTearDown(self):
        await self.rig.aclose()

    async def test_size_sweep(self):
        for n in SIZES:
            for stream in (False, True):
                with self.subTest(size=n, stream=stream):
                    r = await self.rig.chat([user("a" * n)], stream=stream)
                    self.assertEqual(r.status_code, 200, r.text[:200])
                    if stream:
                        self.assertTrue(check_stream(r.text)["content"].startswith("ECHO"))
                    else:
                        check_completion(r.json())
                    self.assertEqual(len(self.rig.fake.requests[-1]["message"]), n)

    async def test_text_sweep(self):
        for name, text in TEXTS.items():
            for stream in (False, True):
                with self.subTest(text=name, stream=stream):
                    r = await self.rig.chat([user(text)], stream=stream)
                    self.assertEqual(r.status_code, 200, r.text[:200])
                    self.assertEqual(self.rig.fake.requests[-1]["message"], text)

    async def test_lone_surrogates_do_not_crash(self):
        raw = b'{"messages":[{"role":"user","content":"bad \\ud800 surrogate \\udfff end"}]}'
        r = await self.rig.http.post(
            "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
        )
        self.assertEqual(r.status_code, 200, r.text[:200])
        check_completion(r.json())

    async def test_message_shapes(self):
        shapes = {
            "none_content": [{"role": "user", "content": None}],
            "missing_content": [{"role": "user"}],
            "empty": [user("")],
            "parts": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
            "weird_parts": [{"role": "user", "content": [1, None, "s", {"type": "text"}]}],
            "dict_content": [{"role": "user", "content": {"a": 1}}],
            "int_content": [{"role": "user", "content": 42}],
            "system_only": [{"role": "system", "content": "be nice"}],
            "system_user": [{"role": "system", "content": "be nice"}, user("hi")],
            "developer_role": [{"role": "developer", "content": "x"}, user("hi")],
            "tool_without_tools": [
                user("q"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "res"},
            ],
            "extra_keys": [{"role": "user", "content": "hi", "name": "bob", "x": [1]}],
            "assistant_first": [{"role": "assistant", "content": "hello"}, user("hi")],
        }
        for name, msgs in shapes.items():
            for stream in (False, True):
                with self.subTest(shape=name, stream=stream):
                    r = await self.rig.chat(msgs, stream=stream)
                    self.assertEqual(r.status_code, 200, r.text[:300])
                    check_stream(r.text) if stream else check_completion(r.json())

    async def test_invalid_requests_are_400_not_500(self):
        bad = [
            {},
            {"messages": []},
            {"messages": "x"},
            {"messages": [1]},
            {"messages": [{"content": "x"}]},
            {"messages": [{"role": 5}]},
            {"messages": [{"role": "assistant", "content": "x"}]},
        ]
        for body in bad:
            with self.subTest(body=body):
                r = await self.rig.http.post("/v1/chat/completions", json=body)
                self.assertIn(r.status_code, (400, 422), r.text)
        r = await self.rig.http.post(
            "/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"}
        )
        self.assertIn(r.status_code, (400, 422))
        r = await self.rig.http.post("/v1/chat/completions", json=[1, 2])
        self.assertIn(r.status_code, (400, 422))

    async def test_extra_openai_parameters_are_ignored(self):
        r = await self.rig.chat(
            [user("hi")],
            temperature=0.2,
            top_p=1,
            max_tokens=5,
            n=1,
            stop=["x"],
            presence_penalty=0.1,
            seed=3,
            response_format={"type": "text"},
            logit_bias={},
            user="u1",
        )
        self.assertEqual(r.status_code, 200)


class ConversationSweep(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rig = Rig()

    async def asyncTearDown(self):
        await self.rig.aclose()

    async def converse(self, turns: int, **extra):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            user("The codeword is KIWI-9. Remember it."),
        ]
        for i in range(turns):
            r = await self.rig.chat(msgs, **extra)
            self.assertEqual(r.status_code, 200, r.text[:200])
            reply = r.json()["choices"][0]["message"]["content"]
            msgs += [{"role": "assistant", "content": reply}, user(f"turn {i}: padding " + "z" * 200)]
        msgs.append(user("What is the codeword?"))
        r = await self.rig.chat(msgs, **extra)
        return r.json()["choices"][0]["message"]["content"]

    async def test_long_conversation_keeps_one_thread_and_small_requests(self):
        answer = await self.converse(40)
        self.assertIn("KIWI-9", answer)  # recalled from IAedu's thread, not from the request
        sent = self.rig.fake.requests
        self.assertEqual(len({f["thread_id"] for f in sent}), 1)
        self.assertLess(max(len(f["message"]) for f in sent[1:]), 400)  # only new messages

    async def test_long_agent_loop_with_tools(self):
        msgs = [{"role": "system", "content": "agent"}, user("do the task")]
        for i in range(20):
            r = await self.rig.chat(msgs, tools=TOOLS)
            self.assertEqual(r.status_code, 200, r.text[:200])
            choice = r.json()["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")  # fake always asks for a tool first
            msg = choice["message"]
            call = msg["tool_calls"][0]
            msgs += [msg, {"role": "tool", "tool_call_id": call["id"], "content": f"output {i}"}]
            r = await self.rig.chat(msgs, tools=TOOLS)  # result in: model answers
            self.assertEqual(r.json()["choices"][0]["finish_reason"], "stop")
            msgs += [
                {"role": "assistant", "content": r.json()["choices"][0]["message"]["content"]},
                user(f"next {i}"),
            ]
        sent = self.rig.fake.requests
        self.assertEqual(len({f["thread_id"] for f in sent}), 1)
        self.assertEqual(sum("Executor functions" in f["message"] for f in sent), 1)  # preamble once

    async def test_concurrent_conversations_do_not_mix_threads(self):
        import asyncio

        async def one(i: int):
            rig_msgs = [user(f"The codeword is WORD-{i}. ok")]
            r = await self.rig.chat(rig_msgs)
            reply = r.json()["choices"][0]["message"]["content"]
            rig_msgs += [{"role": "assistant", "content": reply}, user("What is the codeword?")]
            r = await self.rig.chat(rig_msgs)
            return r.json()["choices"][0]["message"]["content"]

        answers = await asyncio.gather(*[one(i) for i in range(30)])
        for i, a in enumerate(answers):
            self.assertIn(f"WORD-{i}", a)

    async def test_regenerate_and_edit_start_new_thread_with_full_history(self):
        u1, a1 = user("first"), {"role": "assistant", "content": "ECHO"}
        await self.rig.chat([u1, a1, user("second")])
        await self.rig.chat([u1, a1, user("second EDITED"), a1, user("third")])
        t = [f["thread_id"] for f in self.rig.fake.requests]
        self.assertNotEqual(t[0], t[1])
        self.assertIn("second EDITED", self.rig.fake.requests[1]["message"])
        self.assertIn("first", self.rig.fake.requests[1]["message"])


if __name__ == "__main__":
    unittest.main()
