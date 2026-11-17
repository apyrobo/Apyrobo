"""apyrobo registry start says where listings live: in memory, not a database.

The registry keeps listings in a module-level dict, so the old
'Database: ./registry.db' start-up line and the --db help ('Path to SQLite
database') told users their listings were saved when nothing was written.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from apyrobo.cli import cmd_registry_start


def _start(capsys, **kwargs) -> str:
    args = SimpleNamespace(port=8080, host="127.0.0.1", seed=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    fake_uvicorn = MagicMock()
    fake_server = MagicMock()
    fake_server.create_app.return_value = MagicMock()
    with patch.dict(
        "sys.modules",
        {"uvicorn": fake_uvicorn, "apyrobo.registry.server": fake_server},
    ):
        cmd_registry_start(args)
    fake_uvicorn.run.assert_called_once()
    return capsys.readouterr().out


def test_startup_says_listings_are_in_memory(capsys):
    out = _start(capsys, db=None)
    assert "in memory" in out
    assert "lost when the server stops" in out
    assert "Database:" not in out
    assert "registry.db" not in out


def test_db_flag_is_accepted_and_reported_as_ignored(capsys):
    out = _start(capsys, db="/tmp/listings.db")
    assert "Database:" not in out
    assert "--db /tmp/listings.db is ignored" in out


def test_start_sets_no_registry_db_env(capsys, monkeypatch):
    monkeypatch.delenv("REGISTRY_DB_PATH", raising=False)
    _start(capsys, db="./registry.db")
    assert "REGISTRY_DB_PATH" not in os.environ


def test_help_says_in_memory_and_db_is_ignored():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "registry", "start", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    text = " ".join(proc.stdout.split())
    assert "keeps listings in memory" in text
    assert "lost when the server stops" in text
    assert "SQLite" not in text
    assert "--db PATH" in text and "Ignored" in text
