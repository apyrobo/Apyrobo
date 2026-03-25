"""API compatibility checker for apyrobo."""

from __future__ import annotations

import ast
from pathlib import Path


class APICompatibilityChecker:
    """Scan Python source files for usage of deprecated API symbols.

    Example::

        checker = APICompatibilityChecker()
        deprecated = ["apyrobo.memory.MemoryStore", "apyrobo.skills.run_skill"]
        usages = checker.check("src/", deprecated)
        print(checker.report(usages))
    """

    def check(self, source_path: str, deprecated_symbols: list[str]) -> list[str]:
        """Scan *source_path* for usages of deprecated symbols.

        Args:
            source_path: File or directory to scan (recursively for dirs).
            deprecated_symbols: Fully-qualified or short symbol names to look for.

        Returns:
            List of human-readable strings like
            ``"src/foo.py:12 — usage of deprecated 'run_skill'"``.
        """
        usages: list[str] = []
        path = Path(source_path)
        files = list(path.rglob("*.py")) if path.is_dir() else [path]

        short_names = {sym.split(".")[-1] for sym in deprecated_symbols}

        for py_file in files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                name = self._node_name(node)
                if name and name in short_names:
                    line = getattr(node, "lineno", "?")
                    usages.append(
                        f"{py_file}:{line} — usage of deprecated '{name}'"
                    )

        return usages

    def report(self, usages: list[str]) -> str:
        """Render *usages* as a human-readable report.

        Args:
            usages: Output of :meth:`check`.

        Returns:
            Formatted string.
        """
        if not usages:
            return "No deprecated API usages found.\n"

        lines = [f"Found {len(usages)} deprecated API usage(s):\n"]
        for usage in usages:
            lines.append(f"  • {usage}")
        lines.append(
            "\nRun `apyrobo migrate --check` to get automated fix suggestions."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    @staticmethod
    def _node_name(node: ast.AST) -> str | None:
        """Extract a short identifier name from an AST node, if any."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None  # Handled separately
        return None
