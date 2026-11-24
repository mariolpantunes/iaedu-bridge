"""Goose integration: custom provider file and a launcher that runs the bridge in-process."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings

PROVIDER = "custom_iaedu"
DEFAULT_MODELS = ("iaedu-5.5", "iaedu-4o")


def provider_json(models: tuple[str, ...] = DEFAULT_MODELS, port: int = 8000) -> dict:
    return {
        "name": PROVIDER,
        "engine": "openai",
        "display_name": "iaedu",
        "description": "IAedu agents through the local iaedu-bridge (OpenAI-compatible)",
        "api_key_env": "IAEDU_BRIDGE_API_KEY",
        "base_url": f"http://127.0.0.1:{port}",
        "base_path": "v1/chat/completions",
        "models": [
            {
                "name": m,
                "context_limit": 128000,
                "input_token_cost": None,
                "output_token_cost": None,
                "currency": None,
                "supports_cache_control": None,
                "reasoning": False,
                "supports_vision": False,
            }
            for m in models
        ],
        "headers": None,
        "timeout_seconds": 600,
        "supports_streaming": True,
        "requires_auth": True,
        "catalog_provider_id": None,
        "env_vars": None,
        "dynamic_models": False,
        "skip_canonical_filtering": True,
        "setup_steps": [],
        "fast_model": models[0],
        "preserves_thinking": False,
    }


def goose_config_dir(env: dict[str, str] | None = None, platform: str | None = None) -> Path:
    """Where Goose reads its configuration (Windows: %APPDATA%\\Block\\goose\\config)."""
    env = dict(os.environ) if env is None else env
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        base = env.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Block" / "goose" / "config"
    return Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "goose"


def install_provider(directory: Path | None = None, port: int = 8000) -> Path:
    target = (directory or goose_config_dir() / "custom_providers") / f"{PROVIDER}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(provider_json(port=port), indent=2) + "\n", encoding="utf-8")
    return target


def bridge_alive(port: int, key: str) -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run_goose(settings: Settings, args: list[str], port: int = 8000) -> int:
    """Run `goose <args>` on the custom_iaedu provider, starting the bridge if needed."""
    import uvicorn

    from .app import create_app

    goose = shutil.which("goose")
    if goose is None:
        print("goose not found on PATH; install it first (https://block.github.io/goose)", file=sys.stderr)
        return 127
    server, thread = None, None
    if not bridge_alive(port, settings.bridge_key):
        cfg = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.1)
        if not server.started:
            print(f"could not start the bridge on port {port} (is it in use?)", file=sys.stderr)
            return 1
    env = dict(os.environ)
    env.update(
        GOOSE_PROVIDER=PROVIDER,
        GOOSE_MODEL=settings.model,
        IAEDU_BRIDGE_API_KEY=settings.bridge_key or "unused",
    )
    try:
        return subprocess.call([goose, *args], env=env)
    except KeyboardInterrupt:
        return 130
    finally:
        if server is not None and thread is not None:
            server.should_exit = True
            thread.join(timeout=5)
