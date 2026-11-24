"""Settings: environment variables, then ./.env, then the per-user config file.

The per-user file lives in the platform config folder (Windows: %APPDATA%\\iaedu-bridge,
elsewhere: $XDG_CONFIG_HOME or ~/.config, then iaedu-bridge) and is written by
`iaedu-bridge configure`. Values from the real environment always win.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP = "iaedu-bridge"
REQUIRED = ("IAEDU_ENDPOINT", "IAEDU_API_KEY", "IAEDU_CHANNEL_ID")
OPTIONAL = (
    "BRIDGE_API_KEY",
    "IAEDU_MODEL",
    "IAEDU_RETRIES",
    "IAEDU_BACKOFF",
    "IAEDU_MAX_CONCURRENCY",
    "IAEDU_TIMEOUT",
)


class ConfigError(Exception):
    """Missing or invalid configuration (message is meant for the user)."""


def config_dir(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        base = env.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP


def config_file(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    return config_dir(env, platform) / "config.env"


def parse_env_text(text: str) -> dict[str, str]:
    """Tiny .env parser: KEY=VALUE lines, # comments, optional quotes, CRLF safe."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key.strip():
            out[key.strip()] = value
    return out


def read_env_file(path: Path) -> dict[str, str]:
    try:
        return parse_env_text(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_values(
    env: Mapping[str, str] | None = None, cwd: Path | None = None
) -> tuple[dict[str, str], list[str]]:
    """Merge the sources (lowest priority first); also return the files that were read."""
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    merged: dict[str, str] = {}
    used: list[str] = []
    for path in (config_file(env), cwd / ".env"):
        values = read_env_file(path)
        if values:
            merged.update(values)
            used.append(str(path))
    merged.update({k: v for k, v in env.items() if k in REQUIRED + OPTIONAL and v != ""})
    return merged, used


@dataclass(frozen=True)
class Settings:
    endpoint: str
    api_key: str
    channel_id: str
    bridge_key: str = ""
    model: str = "iaedu-5.5"
    retries: int = 3
    backoff: float = 2.0
    max_concurrency: int = 2
    timeout: float = 180.0

    @classmethod
    def from_values(cls, v: Mapping[str, str]) -> "Settings":
        missing = [k for k in REQUIRED if not v.get(k)]
        if missing:
            raise ConfigError(
                "missing " + ", ".join(missing) + ". Run `iaedu-bridge configure` "
                f"(writes {config_file()}) or set them in the environment or a .env file."
            )
        try:
            return cls(
                endpoint=v["IAEDU_ENDPOINT"].strip(),
                api_key=v["IAEDU_API_KEY"].strip(),
                channel_id=v["IAEDU_CHANNEL_ID"].strip(),
                bridge_key=v.get("BRIDGE_API_KEY", ""),
                model=v.get("IAEDU_MODEL") or cls.model,
                retries=int(v.get("IAEDU_RETRIES") or cls.retries),
                backoff=float(v.get("IAEDU_BACKOFF") or cls.backoff),
                max_concurrency=int(v.get("IAEDU_MAX_CONCURRENCY") or cls.max_concurrency),
                timeout=float(v.get("IAEDU_TIMEOUT") or cls.timeout),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid number in the configuration: {exc}") from exc

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_values(load_values()[0])
