from __future__ import annotations

from pathlib import Path

from hipengine import cli


def test_top_level_cli_help_lists_primary_commands(capsys) -> None:
    assert cli.main(["--help"]) == 0

    out = capsys.readouterr().out
    assert "usage: hipengine <command>" in out
    assert "serve" in out
    assert "bench" in out


def test_cli_version_returns_success(capsys) -> None:
    assert cli.main(["version"]) == 0

    assert capsys.readouterr().out.strip()


def test_cli_serve_forwards_to_server_main(monkeypatch) -> None:
    import hipengine.server.__main__ as server_main

    seen = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return 7

    monkeypatch.setattr(server_main, "main", fake_main)

    assert cli.main(["serve", "--model", "fake-model", "--port", "9000"]) == 7
    assert seen["argv"] == ["--model", "fake-model", "--port", "9000"]


def test_cli_bench_help_lists_packaged_launchers(capsys) -> None:
    assert cli.main(["bench", "--help"]) == 0

    out = capsys.readouterr().out
    assert "usage: hipengine bench <benchmark>" in out
    assert "paro" in out
    assert "gguf" in out


def test_cli_rejects_unknown_command(capsys) -> None:
    assert cli.main(["nope"]) == 2

    err = capsys.readouterr().err
    assert "unknown command" in err


def test_pyproject_exposes_only_top_level_console_script() -> None:
    text = Path("pyproject.toml").read_text()

    assert 'hipengine = "hipengine.cli:main"' in text
    assert "hipengine-server" not in text
