"""Plugin system for APYROBO — third-party skills, adapters, and providers."""

from apyrobo.plugins.base import ApyroboPlugin
from apyrobo.plugins.loader import PluginLoader
from apyrobo.plugins.registry import PluginRegistry

__all__ = ["ApyroboPlugin", "PluginLoader", "PluginRegistry"]
