"""Hardware auto-discovery — detect robot model from live ROS node names.

AutoDiscovery queries the active ROS graph (or a supplied node list) and
matches it against registered hardware specs to identify the connected robot.
When rclpy is unavailable it falls back to an empty node list so the registry
still loads cleanly.
"""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apyrobo.hardware.schema import HardwareRegistry, HardwareSpec

logger = logging.getLogger(__name__)

try:
    import rclpy  # type: ignore
    _RCLPY_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
    _RCLPY_AVAILABLE = False


class AutoDiscovery:
    """Detect the connected robot model from the live ROS 2 node graph.

    Usage::

        from apyrobo.hardware import HardwareRegistry, AutoDiscovery

        registry = HardwareRegistry()
        discovery = AutoDiscovery(registry)

        spec = discovery.detect()          # live ROS graph
        spec = discovery.detect(["list"])  # supply node list directly
        if spec:
            discovery.load_skill_package(spec)
    """

    def __init__(self, registry: "HardwareRegistry") -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def detect(
        self, node_names: list[str] | None = None
    ) -> "HardwareSpec | None":
        """Return the best-matching HardwareSpec for *node_names*.

        If *node_names* is ``None`` the live ROS 2 node graph is queried via
        rclpy.  If rclpy is unavailable an empty list is used (no match).
        """
        if node_names is None:
            node_names = self._get_ros_nodes()

        spec = self._registry.detect_from_nodes(node_names)
        if spec:
            logger.info(
                "AutoDiscovery: detected %s (%s) from %d nodes",
                spec.model,
                spec.robot_id,
                len(node_names),
            )
        else:
            logger.debug(
                "AutoDiscovery: no spec matched from %d node(s)", len(node_names)
            )
        return spec

    def _get_ros_nodes(self) -> list[str]:
        """Query the live ROS 2 graph for active node names."""
        if not _RCLPY_AVAILABLE:
            logger.debug("rclpy not available — returning empty node list")
            return []
        try:
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node("_apyrobo_autodiscovery")
            try:
                names_and_ns = node.get_node_names_and_namespaces()
                nodes = []
                for name, ns in names_and_ns:
                    if ns == "/":
                        nodes.append(f"/{name}")
                    else:
                        nodes.append(f"{ns}/{name}")
                return nodes
            finally:
                node.destroy_node()
        except Exception as exc:
            logger.warning("AutoDiscovery: ROS node query failed (%s)", exc)
            return []

    # ------------------------------------------------------------------
    # Skill package loading
    # ------------------------------------------------------------------

    def load_skill_package(self, spec: "HardwareSpec") -> bool:
        """Attempt to import the skill package named in *spec.skill_package*.

        Returns ``True`` if the package was found and imported, ``False``
        otherwise (package not installed or not specified).
        """
        pkg = spec.skill_package
        if not pkg:
            logger.debug("No skill_package specified for %s", spec.robot_id)
            return False
        try:
            importlib.import_module(pkg)
            logger.info("Loaded skill package %r for %s", pkg, spec.robot_id)
            return True
        except ImportError:
            logger.debug(
                "Skill package %r not installed for %s", pkg, spec.robot_id
            )
            return False

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self, spec: "HardwareSpec | None") -> str:
        """Return a one-line human-readable summary of the detection result."""
        if spec is None:
            return "No matching hardware spec detected."
        parts = [f"Detected: {spec.model} ({spec.robot_id})"]
        if spec.manufacturer:
            parts.append(f"by {spec.manufacturer}")
        if spec.sensors:
            parts.append(f"— sensors: {', '.join(spec.sensors)}")
        if spec.skill_package:
            parts.append(f"— skills: {spec.skill_package}")
        return " ".join(parts)
