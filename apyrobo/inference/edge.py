"""Edge inference adapter for on-robot low-latency model execution."""
from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import llama_cpp  # type: ignore
    _LLAMA_AVAILABLE = True
except ImportError:
    llama_cpp = None  # type: ignore
    _LLAMA_AVAILABLE = False


@dataclass
class EdgeModelConfig:
    model_id: str
    model_path: str = ""
    max_tokens: int = 256
    quantization: str = "int8"
    device: str = "cpu"
    warmup_on_load: bool = True


@dataclass
class EdgeInferenceResult:
    text: str
    latency_ms: float
    model_id: str
    tokens_used: int = 0


class EdgeInferenceAdapter:
    """Runs a quantized model locally. Falls back to stub if llama_cpp unavailable."""

    def __init__(self, config: EdgeModelConfig) -> None:
        self.config = config
        self._model: object = None
        self._loaded = False
        self._total_inferences = 0
        self._total_latency_ms = 0.0

    def load(self) -> None:
        if _LLAMA_AVAILABLE and self.config.model_path:
            try:
                self._model = llama_cpp.Llama(model_path=self.config.model_path, n_ctx=512)
                logger.info("Loaded edge model: %s", self.config.model_id)
            except Exception as exc:
                logger.warning("Edge model load failed (%s) — stub mode", exc)
        else:
            logger.info("Edge inference in stub mode (model_id=%s)", self.config.model_id)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def infer(self, prompt: str, max_tokens: Optional[int] = None) -> EdgeInferenceResult:
        t0 = time.perf_counter()
        n_tokens = max_tokens or self.config.max_tokens
        if self._model is not None and _LLAMA_AVAILABLE:
            output = self._model(prompt, max_tokens=n_tokens)
            text = output["choices"][0]["text"]
        else:
            text = f"[stub] response to: {prompt[:50]}"
        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        return EdgeInferenceResult(text=text, latency_ms=latency_ms, model_id=self.config.model_id, tokens_used=n_tokens)

    def infer_batch(self, prompts: list[str]) -> list[EdgeInferenceResult]:
        return [self.infer(p) for p in prompts]

    def is_loaded(self) -> bool:
        return self._loaded

    def get_stats(self) -> dict:
        avg = self._total_latency_ms / self._total_inferences if self._total_inferences else 0.0
        return {
            "model_id": self.config.model_id,
            "total_inferences": self._total_inferences,
            "avg_latency_ms": avg,
        }


class MockEdgeInferenceAdapter(EdgeInferenceAdapter):
    """Deterministic stub for testing — no model files needed."""

    def __init__(self, config: Optional[EdgeModelConfig] = None) -> None:
        if config is None:
            config = EdgeModelConfig(model_id="mock", model_path="")
        super().__init__(config)
        self._responses: list[str] = []

    def load(self) -> None:
        self._loaded = True

    def infer(self, prompt: str, max_tokens: Optional[int] = None) -> EdgeInferenceResult:
        t0 = time.perf_counter()
        text = self._responses.pop(0) if self._responses else f"mock: {prompt[:30]}"
        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        return EdgeInferenceResult(text=text, latency_ms=latency_ms, model_id=self.config.model_id)

    def queue_response(self, text: str) -> None:
        self._responses.append(text)


class EdgeInferenceRouter:
    """Routes inference requests to multiple adapters (round-robin or by latency)."""

    def __init__(self, adapters: list[EdgeInferenceAdapter]) -> None:
        self._adapters = adapters
        self._rr_index = 0

    def infer(self, prompt: str, strategy: str = "round_robin") -> EdgeInferenceResult:
        if not self._adapters:
            raise RuntimeError("No adapters registered")
        if strategy == "round_robin":
            adapter = self._adapters[self._rr_index % len(self._adapters)]
            self._rr_index += 1
        elif strategy == "lowest_latency":
            adapter = min(
                self._adapters,
                key=lambda a: a.get_stats()["avg_latency_ms"] or float("inf"),
            )
        else:
            adapter = self._adapters[0]
        return adapter.infer(prompt)

    def load_all(self) -> None:
        for a in self._adapters:
            a.load()
