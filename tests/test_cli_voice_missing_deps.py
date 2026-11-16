"""`apyrobo listen` and `apyrobo voice --listen` without the voice packages.

A bare `pip install apyrobo` has no numpy, sounddevice, whisper or openai.
`apyrobo listen` (whisper by default) and `apyrobo voice --listen --adapter
whisper|openai` crashed with a ModuleNotFoundError traceback, and
`apyrobo listen --adapter piper` with NotImplementedError. They now print one
error naming what to install (or that piper only speaks) and exit 1, like
`apyrobo dashboard` does without uvicorn.
"""

from __future__ import annotations

import sys

import pytest

from apyrobo import cli


@pytest.fixture
def no_voice_packages(monkeypatch):
    # A None entry makes `import numpy` raise ModuleNotFoundError(name="numpy"),
    # as on a bare install, whether or not numpy is installed here.
    for name in ("numpy", "sounddevice", "whisper", "openai"):
        monkeypatch.setitem(sys.modules, name, None)


def _run(monkeypatch, capsys, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["apyrobo", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code, capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, install",
    [
        (("listen",), "pip install openai-whisper sounddevice numpy"),
        (("listen", "--adapter", "whisper"), "pip install openai-whisper sounddevice numpy"),
        (("listen", "--adapter", "openai"), "pip install openai sounddevice numpy"),
    ],
)
def test_listen_without_packages_names_the_install(
    monkeypatch, capsys, no_voice_packages, argv, install
):
    code, err = _run(monkeypatch, capsys, *argv)
    assert code == 1
    assert "Traceback" not in err
    assert "needs packages that are not installed" in err
    assert "numpy" in err
    assert f"Install them with: {install}" in err
    assert "apyrobo listen --adapter mock" in err


@pytest.mark.parametrize(
    "adapter, install",
    [
        ("whisper", "pip install openai-whisper sounddevice numpy"),
        ("openai", "pip install openai sounddevice numpy"),
    ],
)
def test_voice_listen_without_packages_names_the_install(
    monkeypatch, capsys, no_voice_packages, adapter, install
):
    code, err = _run(
        monkeypatch, capsys, "voice", "--listen", "--adapter", adapter, "--max-turns", "1"
    )
    assert code == 1
    assert "Traceback" not in err
    assert f"Install them with: {install}" in err
    assert "apyrobo voice --listen --adapter mock" in err


def test_listen_with_piper_says_it_only_speaks(monkeypatch, capsys):
    code, err = _run(monkeypatch, capsys, "listen", "--adapter", "piper")
    assert code == 1
    assert "piper adapter only speaks; it cannot listen" in err
    assert "--adapter whisper or --adapter openai" in err


def test_listen_mock_still_plans(monkeypatch, capsys, no_voice_packages):
    monkeypatch.setattr(sys, "argv", ["apyrobo", "listen", "--adapter", "mock"])
    cli.main()
    out = capsys.readouterr().out
    assert "Heard: 'navigate to (1.0, 2.0)'" in out
    assert "Plan: 1 skill(s)" in out


def test_other_import_errors_are_not_hidden():
    exc = ModuleNotFoundError("No module named 'yaml'", name="yaml")
    with pytest.raises(ModuleNotFoundError):
        cli._voice_listen_failed(exc, "whisper", "listen")
    with pytest.raises(NotImplementedError):
        cli._voice_listen_failed(NotImplementedError("x"), "whisper", "listen")
