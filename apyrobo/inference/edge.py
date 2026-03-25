"""
Edge inference — run small quantised models on robot hardware.

Falls back to stub mode when llama_cpp (or equivalent) is not installed,
so the adapter is safe to import on any platform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeModelConfig:
    model_id: str
    model_path: str
    max_tokens: int = 256
    quantization: str = "int8"
    device: str = "cpu"
    warmup_on_load: bool = True


@dataclass
class EdgeInferenceResult:
    text: str
    latency_ms: float
    model_id: str
    tokens_used: int


class EdgeInferenceAdapter:
    """Load and run a local model via llama_cpp (or stub when unavailable)."""

    def __init__(self, config: EdgeModelConfig) -> None:
        self.config = config
        self._model: object = None
        self._stub_mode: bool = False
        self._total_inferences: int = 0
        self._total_latency_ms: float = 0.0

    def load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore

            self._model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.max_tokens * 4,
                n_gpu_layers=-1 if self.config.device != "cpu" else 0,
            )
            if self.config.warmup_on_load:
                self._model("warmup", max_tokens=1)
        except ImportError:
            self._stub_mode = True
            self._model = object()  # non-None sentinel

    def unload(self) -> None:
        self._model = None
        self._stub_mode = False

    def is_loaded(self) -> bool:
        return self._model is not None

    def infer(self, prompt: str, max_tokens: Optional[int] = None) -> EdgeInferenceResult:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded; call load() first")
        tokens = max_tokens or self.config.max_tokens
        t0 = time.perf_counter()
        if self._stub_mode:
            text = f"[stub:{self.config.model_id}] {prompt[:40]}"
            tokens_used = len(prompt.split()) + 1
        else:
            output = self._model(prompt, max_tokens=tokens)
            text = output["choices"][0]["text"]
            tokens_used = output.get("usage", {}).get("completion_tokens", tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        return EdgeInferenceResult(
            text=text,
            latency_ms=latency_ms,
            model_id=self.config.model_id,
            tokens_used=tokens_used,
        )

    def infer_batch(self, prompts: list[str]) -> list[EdgeInferenceResult]:
        return [self.infer(p) for p in prompts]

    def get_stats(self) -> dict:
        avg = (
            self._total_latency_ms / self._total_inferences if self._total_inferences else 0.0
        )
        return {
            "model_id": self.config.model_id,
            "total_inferences": self._total_inferences,
            "avg_latency_ms": avg,
        }


class MockEdgeInferenceAdapter:
    """Deterministic stub — no model file required; useful for unit tests."""

    def __init__(self, model_id: str = "mock-model") -> None:
        self.model_id = model_id
        self._loaded = False
        self._total_inferences = 0
        self._total_latency_ms = 0.0

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def infer(self, prompt: str, max_tokens: Optional[int] = None) -> EdgeInferenceResult:
        if not self._loaded:
            raise RuntimeError("Adapter not loaded")
        t0 = time.perf_counter()
        text = f"mock response for: {prompt[:50]}"
        tokens_used = len(prompt.split()) + 5
        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        return EdgeInferenceResult(
            text=text,
            latency_ms=latency_ms,
            model_id=self.model_id,
            tokens_used=tokens_used,
        )

    def infer_batch(self, prompts: list[str]) -> list[EdgeInferenceResult]:
        return [self.infer(p) for p in prompts]

    def get_stats(self) -> dict:
        avg = (
            self._total_latency_ms / self._total_inferences if self._total_inferences else 0.0
        )
        return {
            "model_id": self.model_id,
            "total_inferences": self._total_inferences,
            "avg_latency_ms": avg,
        }


class EdgeInferenceRouter:
    """Route inference requests across multiple adapters (round-robin or lowest-latency)."""

    def __init__(self, adapters: list, strategy: str = "round_robin") -> None:
        if strategy not in ("round_robin", "lowest_latency"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self._adapters = adapters
        self.strategy = strategy
        self._rr_index = 0

    def _pick_adapter(self):
        if not self._adapters:
            raise RuntimeError("No adapters registered")
        if self.strategy == "round_robin":
            adapter = self._adapters[self._rr_index % len(self._adapters)]
            self._rr_index += 1
            return adapter
        # lowest_latency: pick adapter with lowest avg_latency_ms
        loaded = [a for a in self._adapters if a.is_loaded()]
        if not loaded:
            raise RuntimeError("No loaded adapters available")
        return min(loaded, key=lambda a: a.get_stats()["avg_latency_ms"] or float("inf"))

    def infer(self, prompt: str, max_tokens: Optional[int] = None) -> EdgeInferenceResult:
        return self._pick_adapter().infer(prompt, max_tokens=max_tokens)

    def infer_batch(self, prompts: list[str]) -> list[EdgeInferenceResult]:
        return [self.infer(p) for p in prompts]
