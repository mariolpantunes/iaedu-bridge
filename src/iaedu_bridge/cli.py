"""Command line: serve, configure, check, goose-provider, goose."""

import argparse
import getpass
import json
import logging
import os
import secrets
import sys
from importlib import metadata
from pathlib import Path

from . import goose as goose_mod
from .config import ConfigError, Settings, config_file, load_values, read_env_file

MASK = "********"


def _version() -> str:
    try:
        return metadata.version("iaedu-bridge")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iaedu-bridge", description="OpenAI-compatible bridge for the IAedu chatbot API"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the bridge (default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("configure", help="store endpoint, API key and channel id for this user")
    c.add_argument("--endpoint")
    c.add_argument("--channel-id")
    c.add_argument("--model")
    ck = sub.add_parser("check", help="show the configuration in use")
    ck.add_argument("--live", action="store_true", help="also send one real request to IAedu")
    gp = sub.add_parser("goose-provider", help="print or install the Goose custom provider file")
    gp.add_argument("--install", action="store_true")
    gp.add_argument("--dir", type=Path, help="target folder (default: the Goose custom_providers folder)")
    gp.add_argument("--port", type=int, default=8000)
    g = sub.add_parser("goose", help="run goose on the IAedu provider (starts the bridge)")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("goose_args", nargs=argparse.REMAINDER, help="arguments for goose (after --)")
    return p


def write_config(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# written by `iaedu-bridge configure`; keep private\n" + "".join(
        f"{k}={v}\n" for k, v in values.items() if v
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    if os.name != "nt":
        path.chmod(0o600)


def cmd_configure(args: argparse.Namespace) -> int:
    path = config_file()
    cur = read_env_file(path)

    def ask(prompt: str, key: str, given: str | None = None, secret: bool = False) -> str:
        if given:
            return given
        old = cur.get(key, "")
        shown = f" [{MASK}]" if (old and secret) else (f" [{old}]" if old else "")
        read = getpass.getpass if secret else input
        return read(f"{prompt}{shown}: ").strip() or old

    cur["IAEDU_ENDPOINT"] = ask("API endpoint", "IAEDU_ENDPOINT", args.endpoint)
    cur["IAEDU_API_KEY"] = os.environ.get("IAEDU_API_KEY") or ask(
        "API key (input hidden)", "IAEDU_API_KEY", secret=True
    )
    cur["IAEDU_CHANNEL_ID"] = ask("Channel ID", "IAEDU_CHANNEL_ID", args.channel_id)
    if args.model:
        cur["IAEDU_MODEL"] = args.model
    cur.setdefault("BRIDGE_API_KEY", secrets.token_urlsafe(24))  # local bearer for clients of the bridge
    missing = [k for k in ("IAEDU_ENDPOINT", "IAEDU_API_KEY", "IAEDU_CHANNEL_ID") if not cur.get(k)]
    if missing:
        print("not saved, missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    write_config(path, cur)
    print(f"saved {path}")
    print("next: iaedu-bridge check   |   iaedu-bridge serve   |   iaedu-bridge goose-provider --install")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    values, used = load_values()
    print("config files read:", ", ".join(used) or "none", f"(user file: {config_file()})")
    settings = Settings.from_values(values)
    print(
        f"endpoint   {settings.endpoint}\nchannel    {settings.channel_id}\napi key    {MASK}\n"
        f"bridge key {'set' if settings.bridge_key else 'not set (open on localhost)'}\n"
        f"model      {settings.model}  retries={settings.retries} concurrency={settings.max_concurrency}"
    )
    if not args.live:
        return 0
    import asyncio

    import httpx

    from .upstream import Upstream, UpstreamError

    async def ping() -> str:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            return await Upstream(settings, client).collect("Reply with: ok", "check-" + secrets.token_hex(4))

    try:
        print("live reply:", asyncio.run(ping())[:80])
    except UpstreamError as exc:
        print(f"live check failed ({exc.status}): {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "configure":
            return cmd_configure(args)
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "goose-provider":
            if args.install:
                print(f"installed {goose_mod.install_provider(args.dir, args.port)}")
            else:
                print(json.dumps(goose_mod.provider_json(port=args.port), indent=2))
            return 0
        if args.cmd == "goose":
            rest = args.goose_args[1:] if args.goose_args[:1] == ["--"] else args.goose_args
            return goose_mod.run_goose(Settings.from_env(), rest, args.port)
        import uvicorn

        from .app import create_app

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        host, port = getattr(args, "host", "127.0.0.1"), getattr(args, "port", 8000)
        uvicorn.run(create_app(Settings.from_env()), host=host, port=port, log_level="info")
        return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
