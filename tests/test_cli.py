import contextlib
import io
import json
import os
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from iaedu_bridge import cli, config
from iaedu_bridge import goose as goose_mod

FULL = {"IAEDU_ENDPOINT": "http://x/stream", "IAEDU_API_KEY": "sk-secret-123", "IAEDU_CHANNEL_ID": "chan"}


class EnvParseTest(unittest.TestCase):
    def test_variants(self):
        text = "\ufeff# comment\r\nA=1\r\nexport B = two words \r\nC='q # not comment'\nD=\"x\"\nE=v # trailing\n\nbad line\n=novalue\n"
        got = config.parse_env_text(text)
        self.assertEqual(got, {"A": "1", "B": "two words", "C": "q # not comment", "D": "x", "E": "v"})


class PathTest(unittest.TestCase):
    def test_windows_and_posix_dirs(self):
        win = config.config_dir({"APPDATA": r"C:\Users\me\AppData\Roaming"}, "win32")
        self.assertEqual(win.name, "iaedu-bridge")
        self.assertIn("Roaming", str(win))
        self.assertEqual(config.config_dir({"XDG_CONFIG_HOME": "/x"}, "linux"), Path("/x/iaedu-bridge"))
        self.assertEqual(config.config_dir({}, "darwin").parts[-2:], (".config", "iaedu-bridge"))

    def test_goose_dir(self):
        win = goose_mod.goose_config_dir({"APPDATA": "C:/A"}, "win32")
        self.assertEqual(win.parts[-3:], ("Block", "goose", "config"))
        self.assertEqual(goose_mod.goose_config_dir({"XDG_CONFIG_HOME": "/x"}, "linux"), Path("/x/goose"))


class LoadTest(unittest.TestCase):
    def test_priority_env_over_dotenv_over_user_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, cwd = Path(tmp) / "cfg", Path(tmp) / "proj"
            (home / "iaedu-bridge").mkdir(parents=True)
            cwd.mkdir()
            (home / "iaedu-bridge" / "config.env").write_text(
                "IAEDU_ENDPOINT=user\nIAEDU_CHANNEL_ID=user\nIAEDU_API_KEY=user\n"
            )
            (cwd / ".env").write_text("IAEDU_ENDPOINT=dotenv\nIAEDU_CHANNEL_ID=dotenv\n")
            env = {"XDG_CONFIG_HOME": str(home), "APPDATA": str(home), "IAEDU_ENDPOINT": "real", "OTHER": "x"}
            with mock.patch.object(sys, "platform", "linux"):
                values, used = config.load_values(env, cwd)
            self.assertEqual(values["IAEDU_ENDPOINT"], "real")
            self.assertEqual(values["IAEDU_CHANNEL_ID"], "dotenv")
            self.assertEqual(values["IAEDU_API_KEY"], "user")
            self.assertNotIn("OTHER", values)
            self.assertEqual(len(used), 2)

    def test_settings_errors_and_defaults(self):
        with self.assertRaises(config.ConfigError) as cm:
            config.Settings.from_values({})
        self.assertIn("iaedu-bridge configure", str(cm.exception))
        with self.assertRaises(config.ConfigError):
            config.Settings.from_values({**FULL, "IAEDU_RETRIES": "many"})
        s = config.Settings.from_values({**FULL, "IAEDU_MAX_CONCURRENCY": "4", "IAEDU_MODEL": "iaedu-4o"})
        self.assertEqual((s.max_concurrency, s.model, s.retries), (4, "iaedu-4o", 3))


@contextlib.contextmanager
def isolated_home():
    with tempfile.TemporaryDirectory() as tmp:
        env = {"XDG_CONFIG_HOME": tmp, "APPDATA": tmp, "HOME": tmp, "USERPROFILE": tmp}
        with mock.patch.dict(os.environ, env, clear=False):
            for k in config.REQUIRED + config.OPTIONAL:
                os.environ.pop(k, None)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                yield Path(tmp)
            finally:
                os.chdir(cwd)


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_configure_then_check_never_prints_the_key(self):
        with isolated_home() as home:
            answers = iter(["http://x/stream", "chan"])
            with (
                mock.patch("builtins.input", lambda _="": next(answers)),
                mock.patch("getpass.getpass", lambda _="": "sk-secret-123"),
            ):
                code, out, _ = run_cli("configure", "--model", "iaedu-4o")
            self.assertEqual(code, 0, out)
            path = config.config_file()
            self.assertTrue(path.is_file())
            self.assertIn("IAEDU_API_KEY=sk-secret-123", path.read_text())
            self.assertIn("BRIDGE_API_KEY=", path.read_text())
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            code, out, err = run_cli("check")
            self.assertEqual(code, 0, err)
            self.assertNotIn("sk-secret-123", out + err)
            self.assertIn("iaedu-4o", out)
            self.assertIn(str(home), out)

    def test_configure_with_missing_values_saves_nothing(self):
        with isolated_home():
            with (
                mock.patch("builtins.input", lambda _="": ""),
                mock.patch("getpass.getpass", lambda _="": ""),
            ):
                code, _, err = run_cli("configure")
            self.assertEqual(code, 2)
            self.assertIn("not saved", err)
            self.assertFalse(config.config_file().exists())

    def test_check_without_config_is_a_clean_error(self):
        with isolated_home():
            code, _, err = run_cli("check")
            self.assertEqual(code, 2)
            self.assertIn("configure", err)

    def test_goose_provider_print_and_install(self):
        code, out, _ = run_cli("goose-provider", "--port", "8123")
        data = json.loads(out)
        self.assertEqual((code, data["name"], data["base_url"]), (0, "custom_iaedu", "http://127.0.0.1:8123"))
        self.assertEqual([m["name"] for m in data["models"]], list(goose_mod.DEFAULT_MODELS))
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_cli("goose-provider", "--install", "--dir", str(Path(tmp) / "a" / "b"))
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads((Path(tmp) / "a" / "b" / "custom_iaedu.json").read_text())["engine"], "openai"
            )


def fake_goose(folder: Path, out: Path, port: int) -> None:
    """A `goose` on PATH that records its environment, calls the bridge, and exits 7."""
    script = folder / "fake_goose.py"
    script.write_text(
        "import os, sys, urllib.request\n"
        f"out = {str(out)!r}\n"
        "keys = ('GOOSE_PROVIDER', 'GOOSE_MODEL', 'IAEDU_BRIDGE_API_KEY')\n"
        "open(out, 'w').write(' '.join([os.environ[k] for k in keys] + sys.argv[1:]))\n"
        f"req = urllib.request.Request('http://127.0.0.1:{port}/v1/models', "
        "headers={'Authorization': 'Bearer bk'})\n"
        "assert urllib.request.urlopen(req).status == 200\n"
        "sys.exit(7)\n"
    )
    if sys.platform.startswith("win"):
        (folder / "goose.cmd").write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = folder / "goose"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(0o755)


class GooseLauncherTest(unittest.TestCase):
    def test_runs_goose_with_provider_env_and_stops_the_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env.txt"
            fake_goose(Path(tmp), out, self.port())
            settings = config.Settings(
                endpoint="http://x", api_key="k", channel_id="c", bridge_key="bk", model="iaedu-4o"
            )
            with mock.patch.dict(os.environ, {"PATH": tmp + os.pathsep + os.environ["PATH"]}):
                code = goose_mod.run_goose(settings, ["run", "-t", "hi"], port=self.port())
            self.assertEqual(code, 7)
            self.assertEqual(out.read_text().split(), ["custom_iaedu", "iaedu-4o", "bk", "run", "-t", "hi"])
            self.assertFalse(goose_mod.bridge_alive(self.port(), "bk"))

    _port: int | None = None

    def port(self) -> int:
        if self._port is None:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self._port = s.getsockname()[1]
        return self._port

    def test_missing_goose_is_reported(self):
        settings = config.Settings(endpoint="http://x", api_key="k", channel_id="c")
        with mock.patch.dict(os.environ, {"PATH": ""}):
            self.assertEqual(goose_mod.run_goose(settings, []), 127)


if __name__ == "__main__":
    unittest.main()
