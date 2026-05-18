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


# ---------------------------------------------------------------------------
# Profile auto-detection
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of auto-detecting the best compute profile for this machine."""

    recommended_profile: str
    confidence: str        # "high" | "medium" | "low"
    reason: str
    ollama_available: bool = False
    ollama_models: list[str] = field(default_factory=list)
    gpu_detected: bool = False
    gpu_name: str = ""
    ram_gb: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_profile": self.recommended_profile,
            "confidence": self.confidence,
            "reason": self.reason,
            "ollama_available": self.ollama_available,
            "ollama_models": self.ollama_models,
            "gpu_detected": self.gpu_detected,
            "gpu_name": self.gpu_name,
            "ram_gb": self.ram_gb,
            "notes": self.notes,
        }


def _detect_ollama() -> tuple[bool, list[str]]:
    """Probe Ollama at localhost:11434/api/tags. Returns (available, model_names)."""
    import urllib.request
    import json as _json
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = _json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, [m for m in models if m]
    except Exception:
        return False, []


def _detect_gpu() -> tuple[bool, str, float]:
    """Detect GPU presence. Returns (available, name, vram_gb)."""
    # Try nvidia-smi
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",", 1)
            name = parts[0].strip()
            vram_gb = int(parts[1].strip()) / 1024 if len(parts) > 1 else 0.0
            return True, name, round(vram_gb, 1)
    except Exception:
        pass
    # Try platform info for Apple Silicon / Jetson
    try:
        import platform
        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            # Could be Jetson or Apple Silicon
            node = platform.node().lower()
            if "jetson" in node or "orin" in node:
                return True, "NVIDIA Jetson", 8.0  # estimate
    except Exception:
        pass
    return False, "", 0.0


def _detect_ram_gb() -> float:
    """Estimate available RAM in GB."""
    try:
        import resource
        # /proc/meminfo on Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    try:
        import os
        pages = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        return round((pages * phys) / (1024 ** 3), 1)
    except Exception:
        return 8.0


_MODEL_PRIORITY = [
    # Prefer larger, more capable models when available
    "llama3.1:70b", "llama3.1:8b", "llama3:8b",
    "llama3.2:3b", "llama3.2:1b",
    "mistral:7b", "mixtral:8x7b",
    "phi3:14b", "phi3:3.8b",
    "gemma2:9b", "gemma2:2b",
    "qwen2.5:7b", "qwen2.5:1.5b",
    "codellama:13b", "codellama:7b",
]


def detect_profile() -> DetectionResult:
    """Auto-detect the best compute profile for the current machine.

    Probes: Ollama availability + installed models, GPU presence, RAM.
    Returns a :class:`DetectionResult` with the recommended profile name
    and reasoning.

    Example::

        result = detect_profile()
        print(result.recommended_profile)  # e.g. "jetson-orin"
        print(result.reason)
    """
    notes: list[str] = []

    ollama_ok, ollama_models = _detect_ollama()
    gpu_ok, gpu_name, vram_gb = _detect_gpu()
    ram_gb = _detect_ram_gb()

    if ollama_ok:
        notes.append(f"Ollama is running with {len(ollama_models)} model(s) installed")

    # Choose best installed Ollama model
    best_model: str | None = None
    if ollama_ok and ollama_models:
        for priority_model in _MODEL_PRIORITY:
            for installed in ollama_models:
                if installed.startswith(priority_model.split(":")[0]):
                    best_model = installed
                    break
            if best_model:
                break
        if not best_model:
            best_model = ollama_models[0]

    # Recommendation logic
    if ollama_ok and gpu_ok and vram_gb >= 8:
        profile = "workstation-gpu"
        confidence = "high"
        reason = f"GPU detected ({gpu_name}, {vram_gb:.0f} GB VRAM) with Ollama running"
        if best_model:
            notes.append(f"Recommended Ollama model: {best_model}")
            notes.append(f"Override profile model with: --profile local-{best_model.replace(':', '-')}")
    elif ollama_ok and gpu_ok:
        # Jetson-class (ARM + some GPU)
        profile = "jetson-orin"
        confidence = "medium"
        reason = f"GPU detected ({gpu_name or 'embedded'}) with Ollama running — treating as edge device"
        if best_model:
            notes.append(f"Recommended Ollama model: {best_model}")
    elif ollama_ok:
        # CPU-only machine with Ollama
        if ram_gb >= 32:
            profile = "cpu-only"
            notes.append(f"Ollama running on CPU with {ram_gb:.0f} GB RAM — consider a GPU for faster inference")
        else:
            profile = "raspberry-pi"
        confidence = "medium"
        reason = f"Ollama running (CPU mode) with {len(ollama_models)} model(s)"
        if best_model:
            notes.append(f"Recommended Ollama model: {best_model}")
    elif gpu_ok:
        profile = "workstation-gpu"
        confidence = "medium"
        reason = f"GPU detected ({gpu_name}, {vram_gb:.0f} GB VRAM) — install Ollama for local inference"
        notes.append("Install Ollama: https://ollama.com and run: ollama pull llama3.1:8b")
    else:
        # No GPU, no Ollama — cloud or CPU fallback
        if ram_gb >= 64:
            profile = "cloud"
            confidence = "medium"
            reason = f"Large RAM ({ram_gb:.0f} GB) suggests a cloud/server deployment"
        else:
            profile = "cpu-only"
            confidence = "low"
            reason = "No GPU and no Ollama detected — using cloud LLM fallback"
        notes.append("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use cloud LLM providers")

    return DetectionResult(
        recommended_profile=profile,
        confidence=confidence,
        reason=reason,
        ollama_available=ollama_ok,
        ollama_models=ollama_models,
        gpu_detected=gpu_ok,
        gpu_name=gpu_name,
        ram_gb=ram_gb,
        notes=notes,
    )
