"""Talk to a real ``apyrobo mcp`` process the way a client does.

The server refuses motion once stdin closes, so stdin stays open until the
reply with ``last_id`` is in. ``subprocess.run(input=...)`` closes it at once.
"""
from __future__ import annotations

import json
import subprocess
import threading


def talk(
    cmd: list[str], lines: str, *, last_id: object, timeout: float = 60.0
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="latin-1",
    )
    assert proc.stdin and proc.stdout and proc.stderr
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    try:
        proc.stdin.write(lines)
        proc.stdin.flush()
        seen = []
        for line in proc.stdout:
            seen.append(line)
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(reply, dict) and reply.get("id") == last_id:
                break
        proc.stdin.close()
        seen.append(proc.stdout.read())
        stderr = proc.stderr.read()
        return subprocess.CompletedProcess(cmd, proc.wait(), "".join(seen), stderr)
    finally:
        watchdog.cancel()
