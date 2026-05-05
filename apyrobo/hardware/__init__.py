"""Hardware knowledge schema, registry, and auto-discovery for apyrobo."""
from apyrobo.hardware.schema import HardwareSpec, HardwareRegistry, spec_to_planner_context
from apyrobo.hardware.autodiscovery import AutoDiscovery

__all__ = [
    "HardwareSpec",
    "HardwareRegistry",
    "spec_to_planner_context",
    "AutoDiscovery",
]
