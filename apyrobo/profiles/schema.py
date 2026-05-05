"""Compute profile schema and built-in profile registry.

A ComputeProfile describes the hardware constraints of the machine running the
apyrobo planner.  The profile controls which LLM backend and model are used,
what context-window size is safe, and whether GPU acceleration is expected.

Five built-in profiles cover the common deployment tiers::

    jetson-orin     — NVIDIA Jetson Orin (embedded GPU, 8-16 GB unified RAM)
    raspberry-pi    — Raspberry Pi 4/5 or similar ARM SBC (CPU-only, 4-8 GB)
    workstation-gpu — Linux workstation with discrete GPU (≥ 8 GB VRAM)
    cloud           — Cloud VM / container with no GPU constraints
    cpu-only        — Any CPU-only x86 machine (servers, CI)

Profiles can be overridden per-project via ``pyproject.toml`` or the
``APYROBO_PROFILE`` environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComputeProfile:
    """Hardware constraint profile for the apyrobo planner."""

    name: str
    description: str = ""

    # LLM settings
    llm_provider: str = "anthropic"          # "anthropic" | "ollama" | "openai" | ...
    llm_model: str = "claude-haiku-4-5-20251001"   # default model for this tier
    llm_api_base: str = ""                   # override base URL (e.g. local Ollama endpoint)

    # Context / throughput limits
    max_context_tokens: int = 8_192          # safe max for planning prompts
    max_output_tokens: int = 1_024

    # Hardware flags
    gpu_available: bool = False
    gpu_vram_gb: float = 0.0
    ram_gb: float = 8.0
    cpu_cores: int = 4

    # Feature flags
    edge_inference: bool = False             # run model locally on this device
    streaming: bool = True

    # Arbitrary extra settings (passed through to LLM clients)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (e.g. for JSON output)."""
        return {
            "name": self.name,
            "description": self.description,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "gpu_available": self.gpu_available,
            "gpu_vram_gb": self.gpu_vram_gb,
            "ram_gb": self.ram_gb,
            "cpu_cores": self.cpu_cores,
            "edge_inference": self.edge_inference,
            "streaming": self.streaming,
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_BUILTIN_PROFILES: list[ComputeProfile] = [
    ComputeProfile(
        name="jetson-orin",
        description="NVIDIA Jetson Orin — embedded GPU, 8–64 GB unified RAM",
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        llm_api_base="http://localhost:11434",
        max_context_tokens=4_096,
        max_output_tokens=512,
        gpu_available=True,
        gpu_vram_gb=8.0,
        ram_gb=16.0,
        cpu_cores=12,
        edge_inference=True,
        streaming=True,
    ),
    ComputeProfile(
        name="raspberry-pi",
        description="Raspberry Pi 4/5 or equivalent ARM SBC — CPU only, 4–8 GB RAM",
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        llm_api_base="http://localhost:11434",
        max_context_tokens=2_048,
        max_output_tokens=256,
        gpu_available=False,
        ram_gb=4.0,
        cpu_cores=4,
        edge_inference=True,
        streaming=False,
    ),
    ComputeProfile(
        name="workstation-gpu",
        description="Linux workstation with discrete GPU (≥ 8 GB VRAM)",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        max_context_tokens=32_768,
        max_output_tokens=4_096,
        gpu_available=True,
        gpu_vram_gb=16.0,
        ram_gb=32.0,
        cpu_cores=16,
        edge_inference=False,
        streaming=True,
    ),
    ComputeProfile(
        name="cloud",
        description="Cloud VM or container — no GPU constraints, best model available",
        llm_provider="anthropic",
        llm_model="claude-opus-4-7",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        gpu_available=False,
        ram_gb=64.0,
        cpu_cores=32,
        edge_inference=False,
        streaming=True,
    ),
    ComputeProfile(
        name="cpu-only",
        description="CPU-only x86 machine — server, CI, or developer laptop without GPU",
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
        max_context_tokens=8_192,
        max_output_tokens=1_024,
        gpu_available=False,
        ram_gb=8.0,
        cpu_cores=8,
        edge_inference=False,
        streaming=True,
    ),
]


# ---------------------------------------------------------------------------
# ProfileRegistry
# ---------------------------------------------------------------------------

class ProfileRegistry:
    """Registry of ComputeProfiles, seeded with the five built-in profiles.

    Usage::

        reg = ProfileRegistry()
        reg.get("jetson-orin")          # returns ComputeProfile
        reg.register(my_custom_profile)
        reg.all()                       # list of all profiles
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ComputeProfile] = {}
        for p in _BUILTIN_PROFILES:
            self._profiles[p.name] = p

    def all(self) -> list[ComputeProfile]:
        """Return all registered profiles in insertion order."""
        return list(self._profiles.values())

    def get(self, name: str) -> ComputeProfile | None:
        """Return the profile with *name*, or ``None`` if not found."""
        return self._profiles.get(name)

    def register(self, profile: ComputeProfile) -> None:
        """Add or replace a profile."""
        self._profiles[profile.name] = profile

    def names(self) -> list[str]:
        return list(self._profiles.keys())


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_registry = ProfileRegistry()


def get_profile(name: str | None = None) -> ComputeProfile:
    """Return a ComputeProfile by name (or from ``APYROBO_PROFILE`` env var).

    Falls back to ``"cpu-only"`` when *name* is ``None`` and the env var is
    unset, or when the requested profile does not exist.
    """
    resolved = name or os.environ.get("APYROBO_PROFILE", "")
    profile = _default_registry.get(resolved) if resolved else None
    return profile or _default_registry.get("cpu-only")  # type: ignore[return-value]
