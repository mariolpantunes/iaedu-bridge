# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28.1"]
# ///
"""Opt-in live sweep: drives a RUNNING bridge (real IAedu behind it) with a request budget.

Usage:  scripts/live_sweep.py [--budget 25] [--pace 4] [--url http://127.0.0.1:8000]
Env:    BRIDGE_API_KEY (bearer for the bridge, optional)
Report: $TMPDIR/iaedu-live-report.json plus a summary on stdout.

Stops early after 3 rate-limit answers in a row so it never hammers a limited quota.
Each case counts the HTTP calls it makes against the budget.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


class Budget(Exception):
    pass


class Sweep:
    def __init__(self, url: str, budget: int, pace: float) -> None:
        key = os.environ.get("BRIDGE_API_KEY", "")
        self.http = httpx.Client(
            base_url=url, timeout=240, headers={"Authorization": f"Bearer {key}"} if key else {}
        )
        self.budget, self.pace, self.calls, self.rl_streak = budget, pace, 0, 0
        self.results: list[dict] = []

    def post(self, label: str, messages: list[dict], **extra) -> dict:
        if self.calls >= self.budget:
            raise Budget
        if self.rl_streak >= 3:
            raise Budget("3 rate limits in a row: stopping")
        self.calls += 1
        t0 = time.time()
        try:
            r = self.http.post("/v1/chat/completions", json={"messages": messages, **extra})
            status, text = r.status_code, r.text
        except httpx.HTTPError as exc:
            status, text = -1, repr(exc)
        self.rl_streak = self.rl_streak + 1 if status == 429 else 0
        out: dict = {"case": label, "status": status, "seconds": round(time.time() - t0, 2)}
        if status == 200 and not extra.get("stream"):
            msg = json.loads(text)["choices"][0]
            out["finish"], out["message"] = msg["finish_reason"], msg["message"]
        elif status == 200:
            out["stream_ok"] = text.rstrip().endswith("data: [DONE]") and '"error"' not in text
            out["text"] = "".join(
                json.loads(f[6:])["choices"][0]["delta"].get("content") or ""
                for f in text.split("\n\n")
                if f.startswith("data: {") and '"choices": [{' in f
            )
        else:
            out["error"] = text[:300]
        self.results.append(
            {k: v for k, v in out.items() if k not in ("message", "text")}
            | {"preview": str(out.get("message") or out.get("text") or "")[:120]}
        )
        time.sleep(self.pace)
        return out

    def cases(self) -> None:
        u = lambda t: {"role": "user", "content": t}  # noqa: E731
        self.post("plain", [u("Reply with exactly: ok")])
        self.post("stream", [u("Count from 1 to 5, comma separated.")], stream=True)
        self.post("stream_usage", [u("Say hi")], stream=True, stream_options={"include_usage": True})
        self.post("unicode", [u("Repeat exactly: ação 世界 👋🏽 مرحبا")])
        self.post("large_input", [u("Ignore the filler and reply 'ok'. " + "lorem ipsum " * 2500)])
        self.post(
            "system_prompt",
            [{"role": "system", "content": "Always answer in exactly one word."}, u("Capital of Portugal?")],
        )
        # memory through the IAedu thread: turn 2 sends only the new message
        msgs = [u("Remember the codeword PINEAPPLE-42. Reply: stored.")]
        a = self.post("memory_1", msgs)
        if a["status"] == 200:
            msgs += [{"role": "assistant", "content": a["message"]["content"]}, u("What was the codeword?")]
            b = self.post("memory_2", msgs)
            if b["status"] == 200 and "PINEAPPLE-42" not in (b["message"]["content"] or ""):
                self.results[-1]["error"] = "codeword not recalled"
        # emulated tool call round trip
        t = [u("Use the tool to run `echo bridge-ok` and tell me its output.")]
        c = self.post("tool_call", t, tools=TOOLS)
        if c["status"] == 200 and c.get("finish") == "tool_calls":
            call = c["message"]["tool_calls"][0]
            t += [c["message"], {"role": "tool", "tool_call_id": call["id"], "content": "bridge-ok"}]
            d = self.post("tool_result", t, tools=TOOLS)
            if d["status"] == 200 and "bridge-ok" not in (d["message"]["content"] or ""):
                self.results[-1]["error"] = "tool output not used"
        elif c["status"] == 200:
            self.results[-1]["error"] = "no tool call produced"
        self.post("tool_stream", [u("Use the tool to run `date`.")], tools=TOOLS, stream=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--pace", type=float, default=4.0, help="seconds between requests")
    args = ap.parse_args()
    sw = Sweep(args.url, args.budget, args.pace)
    try:
        sw.cases()
    except Budget as exc:
        print("stopped:", exc or "budget reached")
    bad = [r for r in sw.results if r["status"] != 200 or r.get("error") or r.get("stream_ok") is False]
    report = Path(tempfile.gettempdir()) / "iaedu-live-report.json"
    report.write_text(json.dumps(sw.results, indent=1))
    for r in sw.results:
        flag = "FAIL" if r in bad else "ok  "
        print(f"{flag} {r['case']:14} {r['status']} {r['seconds']:>6}s  {r.get('error') or r['preview']}")
    print(f"{len(sw.results) - len(bad)}/{len(sw.results)} ok, {sw.calls} calls, report {report}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
