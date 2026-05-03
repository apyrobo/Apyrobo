"""Interactive skill composition REPL — ``apyrobo skill compose``.

Start the interactive loop::

    from apyrobo.skills.compose import ComposeREPL
    from apyrobo import Robot
    from apyrobo.skills.library import SkillLibrary

    robot = Robot.discover("mock://turtlebot4")
    ComposeREPL(robot, SkillLibrary()).run()

Or call single commands programmatically (useful for tests)::

    repl = ComposeREPL(robot, library)
    out = repl.run_command("add navigate_to x=3.0 y=5.0")
    out = repl.run_command("show")
    out = repl.run_command("run")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.library import SkillLibrary
from apyrobo.skills.skill import Skill, SkillStatus

_HELP_TEXT = """\
Available commands:
  help                          Show this help message
  skills                        List available skills with descriptions
  add <skill_id> [key=val ...]  Add a skill step to the current plan
  show                          Print the current plan as a numbered list
  remove <n>                    Remove step n from the plan
  clear                         Reset the plan to empty
  run                           Execute the current plan on the robot
  test <skill_id> [key=val ...] Run a single skill immediately (no plan)
  save <filename.json>          Save the plan to a JSON file
  load <filename.json>          Load a plan from a JSON file
  robot <uri>                   Switch to a different robot adapter
  quit / exit                   Exit the REPL"""


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

def _parse_kwargs(tokens: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` tokens into a dict, auto-converting numeric types."""
    params: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected key=value, got: {token!r}")
        key, _, raw = token.partition("=")
        key = key.strip()
        raw = raw.strip()
        try:
            value: Any = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                if (raw.startswith('"') and raw.endswith('"')) or \
                   (raw.startswith("'") and raw.endswith("'")):
                    value = raw[1:-1]
                else:
                    value = raw
        params[key] = value
    return params


# ---------------------------------------------------------------------------
# ComposeREPL
# ---------------------------------------------------------------------------

class ComposeREPL:
    """
    Interactive skill composition REPL.

    Importable and fully testable without any real I/O: call
    ``run_command(line)`` to process one command and receive the
    output as a plain string.  Call ``run()`` for the interactive loop.

    Args:
        robot:      Any robot object returned by ``Robot.discover()``.
        library:    ``SkillLibrary`` with the skills to compose from.
                    Defaults to a library containing only built-in skills.
        output_fn:  Output sink; defaults to ``print``.  Override in tests.
    """

    def __init__(
        self,
        robot: Any,
        library: SkillLibrary | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._robot = robot
        self._library = library or SkillLibrary()
        self._plan: list[tuple[str, dict[str, Any]]] = []
        self._output: Callable[[str], None] = output_fn or print
        self._exit_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_command(self, line: str) -> str:
        """Process one REPL command; return the full output as a string."""
        parts = line.strip().split()
        if not parts:
            return ""

        captured: list[str] = []

        saved = self._output
        self._output = captured.append
        try:
            self._dispatch(parts[0], parts[1:])
        finally:
            self._output = saved

        return "\n".join(captured)

    def run(self) -> None:
        """Start the interactive REPL loop (blocks until exit)."""
        self._setup_readline()
        self._out("APYROBO Skill Composer  —  type 'help' for commands, 'quit' to exit\n")

        while not self._exit_requested:
            try:
                line = input(self._prompt())
            except EOFError:
                self._out("")
                break
            except KeyboardInterrupt:
                self._out("")
                try:
                    answer = input("Exit compose? (y/n) ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if answer in ("y", "yes"):
                    break
                continue

            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd, rest = parts[0], parts[1:]

            if cmd in ("quit", "exit"):
                break

            self._dispatch(cmd, rest)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prompt(self) -> str:
        n = len(self._plan)
        return f"compose[{n}]> " if n else "compose> "

    def _out(self, s: str = "") -> None:
        self._output(s)

    def _dispatch(self, cmd: str, rest: list[str]) -> None:
        handlers = {
            "help":   self._cmd_help,
            "skills": self._cmd_skills,
            "add":    self._cmd_add,
            "show":   self._cmd_show,
            "remove": self._cmd_remove,
            "clear":  self._cmd_clear,
            "run":    self._cmd_run,
            "test":   self._cmd_test,
            "save":   self._cmd_save,
            "load":   self._cmd_load,
            "robot":  self._cmd_robot,
            "quit":   self._cmd_quit,
            "exit":   self._cmd_quit,
        }
        fn = handlers.get(cmd)
        if fn is None:
            self._out(f"Unknown command {cmd!r}. Type 'help' for available commands.")
            return
        fn(rest)

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_help(self, rest: list[str]) -> None:
        self._out(_HELP_TEXT)

    def _cmd_skills(self, rest: list[str]) -> None:
        skills = self._library.all_skills()
        if not skills:
            self._out("No skills available.")
            return
        self._out(f"  {'Skill ID':<24} Description")
        self._out("  " + "─" * 62)
        for sid, skill in sorted(skills.items()):
            desc = (skill.description or "")[:38]
            self._out(f"  {sid:<24} {desc}")

    def _cmd_add(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: add <skill_id> [key=val ...]")
            return
        skill_id = rest[0]
        skill = self._library.get(skill_id)
        if skill is None:
            self._out(
                f"Unknown skill: {skill_id!r}. "
                "Use 'skills' to see available skills."
            )
            return
        try:
            params = _parse_kwargs(rest[1:])
        except ValueError as exc:
            self._out(f"Parameter error: {exc}")
            return

        self._plan.append((skill_id, params))
        step = len(self._plan)
        params_str = (
            ", ".join(f"{k}={v}" for k, v in params.items())
            if params else "(defaults)"
        )
        self._out(f"  [{step}] {skill_id}  {params_str}")

    def _cmd_show(self, rest: list[str]) -> None:
        if not self._plan:
            self._out("Plan is empty. Use 'add <skill_id>' to add steps.")
            return
        self._out(f"Plan ({len(self._plan)} step(s)):")
        for i, (sid, params) in enumerate(self._plan, 1):
            params_str = (
                ", ".join(f"{k}={v}" for k, v in params.items())
                if params else "(defaults)"
            )
            self._out(f"  [{i}] {sid:<22} {params_str}")

    def _cmd_remove(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: remove <step_number>")
            return
        try:
            n = int(rest[0])
        except ValueError:
            self._out(f"Invalid step number: {rest[0]!r}")
            return
        if n < 1 or n > len(self._plan):
            self._out(
                f"Step {n} out of range "
                f"(plan has {len(self._plan)} step(s))"
            )
            return
        removed_id, _ = self._plan.pop(n - 1)
        self._out(f"Removed step {n}: {removed_id}")

    def _cmd_clear(self, rest: list[str]) -> None:
        n = len(self._plan)
        self._plan.clear()
        self._out(f"Plan cleared ({n} step(s) removed).")

    def _cmd_run(self, rest: list[str]) -> None:
        if not self._plan:
            self._out("Plan is empty. Add skills with 'add <skill_id>'.")
            return

        graph = self._build_graph()
        if graph is None:
            return

        self._out(f"Running plan ({len(self._plan)} step(s))...\n")

        executor = SkillExecutor(self._robot)

        def _on_event(event: Any) -> None:
            status = (
                event.status.value
                if hasattr(event.status, "value")
                else str(event.status)
            )
            if status in ("completed", "failed"):
                icon = "✅" if status == "completed" else "❌"
                msg = f"  {event.message}" if event.message else ""
                self._out(f"  {icon} {event.skill_id}: {status}{msg}")

        executor.on_event(_on_event)

        try:
            result = executor.execute_graph(graph)
            self._out("")
            icon = "✅" if result.status.value == "completed" else "❌"
            self._out(
                f"{icon} {result.status.value.upper()}  "
                f"{result.steps_completed}/{result.steps_total} steps"
            )
            if result.error:
                self._out(f"  Error: {result.error}")
        except Exception as exc:
            self._out(f"❌ Execution error: {exc}")

    def _cmd_test(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: test <skill_id> [key=val ...]")
            return
        skill_id = rest[0]
        skill = self._library.get(skill_id)
        if skill is None:
            self._out(f"Unknown skill: {skill_id!r}")
            return
        try:
            params = _parse_kwargs(rest[1:])
        except ValueError as exc:
            self._out(f"Parameter error: {exc}")
            return

        self._out(f"Testing {skill_id}...")
        executor = SkillExecutor(self._robot)
        t0 = time.monotonic()
        try:
            status = executor.execute_skill(skill, params)
            elapsed = time.monotonic() - t0
            icon = "✅" if status == SkillStatus.COMPLETED else "❌"
            self._out(f"  {icon} {status.value}  ({elapsed:.3f}s)")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            self._out(f"  ❌ Error: {exc}  ({elapsed:.3f}s)")

    def _cmd_save(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: save <filename.json>")
            return
        path = Path(rest[0])
        data = {
            "version": 1,
            "steps": [
                {"skill_id": sid, "params": params}
                for sid, params in self._plan
            ],
        }
        try:
            path.write_text(json.dumps(data, indent=2))
            self._out(f"Plan saved to {path}  ({len(self._plan)} step(s))")
        except OSError as exc:
            self._out(f"Save failed: {exc}")

    def _cmd_load(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: load <filename.json>")
            return
        path = Path(rest[0])
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            self._out(f"File not found: {path}")
            return
        except json.JSONDecodeError as exc:
            self._out(f"Invalid JSON: {exc}")
            return

        steps = data.get("steps", [])
        loaded: list[tuple[str, dict[str, Any]]] = []
        for s in steps:
            sid = s.get("skill_id", "")
            params = s.get("params", {})
            if not sid:
                self._out("Warning: skipped a step with no skill_id")
                continue
            loaded.append((sid, params))

        self._plan = loaded
        self._out(f"Loaded {len(self._plan)} step(s) from {path}")
        if self._plan:
            self._cmd_show([])

    def _cmd_robot(self, rest: list[str]) -> None:
        if not rest:
            self._out("Usage: robot <uri>")
            return
        uri = rest[0]
        try:
            from apyrobo.core.robot import Robot
            self._robot = Robot.discover(uri)
            self._out(f"Switched to robot: {uri}")
        except Exception as exc:
            self._out(f"Failed to connect to {uri!r}: {exc}")

    def _cmd_quit(self, rest: list[str]) -> None:
        self._exit_requested = True

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> SkillGraph | None:
        """Convert the plan list to a linear SkillGraph."""
        graph = SkillGraph()
        prev_node_id: str | None = None

        for i, (skill_id, params) in enumerate(self._plan):
            skill = self._library.get(skill_id)
            if skill is None:
                self._out(
                    f"Error: skill {skill_id!r} not found in library "
                    f"(step {i + 1})"
                )
                return None

            # Append step index so the same skill can appear multiple times.
            # The handler registry strips the trailing _N suffix, so dispatch
            # still resolves to the correct handler.
            node_id = f"{skill_id}_{i}"
            node_skill = Skill(
                skill_id=node_id,
                name=skill.name,
                description=skill.description,
                required_capability=skill.required_capability,
                preconditions=skill.preconditions,
                postconditions=skill.postconditions,
                parameters={**skill.parameters, **params},
                timeout_seconds=skill.timeout_seconds,
                retry_count=skill.retry_count,
                handler_module=skill.handler_module,
                handler_fn=skill.handler_fn,
            )
            graph.add_skill(
                node_skill,
                depends_on=[prev_node_id] if prev_node_id else None,
                parameters=params,
            )
            prev_node_id = node_id

        return graph

    # ------------------------------------------------------------------
    # Readline tab completion
    # ------------------------------------------------------------------

    def _setup_readline(self) -> None:
        try:
            import readline
        except ImportError:
            return  # Windows — silently skip

        skill_ids = sorted(self._library.all_skills().keys())
        commands = [
            "help", "skills", "add", "show", "remove", "clear",
            "run", "test", "save", "load", "robot", "quit", "exit",
        ]

        def _completer(text: str, state: int) -> str | None:
            buf = readline.get_line_buffer()
            tokens = buf.lstrip().split()

            if not tokens or (len(tokens) == 1 and not buf.endswith(" ")):
                candidates = [c for c in commands if c.startswith(text)]
            elif tokens[0] in ("add", "test") and (
                len(tokens) == 1 or (len(tokens) == 2 and not buf.endswith(" "))
            ):
                candidates = [s for s in skill_ids if s.startswith(text)]
            else:
                candidates = []

            return candidates[state] if state < len(candidates) else None

        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
