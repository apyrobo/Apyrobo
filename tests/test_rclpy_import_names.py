"""Guard: every rclpy submodule referenced anywhere in the repo must exist.

rclpy is only installed inside the integration Docker image, so plain CI
never import-checks these names. That gap let `rclpy.callback_group`
(singular — the real module is `rclpy.callback_groups`) sit in
apyrobo.core.ros2_bridge undetected: the bridge's broad try/except turned
the bad import into a silent _HAS_ROS2=False and the ros2:// adapter
vanished with a misleading "rclpy missing" error.

This test statically scans all Python sources for `import rclpy.X` /
`from rclpy.X import ...` and checks X against the real module list,
captured from ros:humble's apt rclpy via:

    python3 -c "import rclpy, pkgutil; \
        print(sorted(m.name for m in pkgutil.iter_modules(rclpy.__path__)))"

If rclpy ever adds a module we legitimately use, add it to the list below.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# From ros:humble-ros-base (apt ros-humble-rclpy). Public + private modules.
RCLPY_MODULES = frozenset({
    "_rclpy_pybind11", "action", "callback_groups", "client", "clock",
    "constants", "context", "duration", "exceptions", "executors",
    "expand_topic_name", "guard_condition", "impl", "lifecycle", "logging",
    "node", "parameter", "parameter_service", "publisher", "qos",
    "qos_event", "qos_overriding_options", "serialization", "service",
    "signals", "subscription", "task", "time", "time_source", "timer",
    "topic_endpoint_info", "topic_or_service_is_hidden", "type_support",
    "utilities", "validate_full_topic_name", "validate_namespace",
    "validate_node_name", "validate_parameter_name", "validate_topic_name",
    "wait_for_message", "waitable",
})


def _rclpy_submodules_used(tree: ast.AST):
    """Yield the first component after 'rclpy.' for every dotted import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "rclpy" and len(parts) > 1:
                    yield parts[1], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[0] == "rclpy" and len(parts) > 1:
                yield parts[1], node.lineno


def test_all_rclpy_submodule_imports_exist():
    bad = []
    for path in sorted(REPO_ROOT.glob("apyrobo/**/*.py")) + sorted(
        REPO_ROOT.glob("tests/**/*.py")
    ):
        tree = ast.parse(path.read_text(), filename=str(path))
        for submodule, lineno in _rclpy_submodules_used(tree):
            if submodule not in RCLPY_MODULES:
                bad.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: rclpy.{submodule}")

    assert not bad, (
        "Import of rclpy submodule(s) that do not exist in the real rclpy "
        "(remember: it's callback_groups, plural):\n  " + "\n  ".join(bad)
    )


def test_mocked_rclpy_module_names_are_real():
    """The sys.modules keys the mocked tests stub must also be real names,
    or the mocks themselves would hide a bad import (as happened when both
    ros2_bridge and its mock used 'rclpy.callback_group')."""
    bad = []
    for path in sorted(REPO_ROOT.glob("tests/test_ros2_*_mocked.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            parts = node.value.split(".")
            if parts[0] == "rclpy" and len(parts) > 1:
                if parts[1] not in RCLPY_MODULES:
                    bad.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.value!r}"
                    )

    assert not bad, (
        "Mocked test stubs a rclpy module name that does not exist in the "
        "real rclpy:\n  " + "\n  ".join(bad)
    )
