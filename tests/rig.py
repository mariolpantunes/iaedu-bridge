"""Test rig: the bridge app wired to a FakeIaedu through in-process transports."""

import httpx
from fake_iaedu import FakeIaedu

from iaedu_bridge.app import create_app
from iaedu_bridge.config import Settings

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}},
    },
]


async def nosleep(_: float) -> None:
    return None


class Rig:
    def __init__(self, fake: FakeIaedu | None = None, **settings) -> None:
        self.fake = fake or FakeIaedu()
        cfg = {"retries": 3, "backoff": 0.0, "max_concurrency": 2, **settings}
        self.settings = Settings(endpoint="http://iaedu/stream", api_key="k", channel_id="c", **cfg)
        self.app = create_app(self.settings, self.fake.client(), sleep=nosleep)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://bridge", timeout=60
        )

    async def chat(self, messages, **extra) -> httpx.Response:
        return await self.http.post("/v1/chat/completions", json={"messages": messages, **extra})

    async def aclose(self) -> None:
        await self.http.aclose()


def user(text: str) -> dict:
    return {"role": "user", "content": text}
