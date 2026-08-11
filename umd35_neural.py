"""Local neural embedding backend for UMD 3.5.

The default model is multilingual, symmetric (query/document prefixes are not
required), and produces the same 384 dimensions as HashEncoder. FastEmbed runs
the model through ONNX Runtime, avoiding a PyTorch dependency.
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KNOWN_DIMENSIONS = {
    DEFAULT_MODEL: 384,
    "BAAI/bge-small-zh-v1.5": 512,
    "intfloat/multilingual-e5-small": 384,
}


class FastEmbedEncoder:
    """Lazy, cached FastEmbed adapter implementing UMD's TextEncoder protocol."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimensions: int | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        cache_size: int = 4096,
        threads: int | None = None,
        providers: list[str] | None = None,
        fallback: object | None = None,
    ) -> None:
        inferred_dimensions = dimensions or KNOWN_DIMENSIONS.get(model_name)
        if inferred_dimensions is None:
            raise ValueError("dimensions are required for an unregistered model")
        if batch_size < 1 or cache_size < 0:
            raise ValueError("batch_size must be positive and cache_size non-negative")
        if fallback is not None and getattr(fallback, "dimensions", None) != inferred_dimensions:
            raise ValueError("fallback encoder must use the same dimensions")

        self.model_name = model_name
        self.dimensions = inferred_dimensions
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.threads = threads
        self.providers = providers
        self.fallback = fallback

        self._model = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.neural_calls = 0
        self.fallback_calls = 0
        self.last_error: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "FastEmbed is not installed. Run: pip install -r requirements-neural.txt"
            ) from error

        kwargs: dict[str, object] = {"model_name": self.model_name}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.threads is not None:
            kwargs["threads"] = self.threads
        if self.providers is not None:
            kwargs["providers"] = self.providers
        # FastEmbed 0.8 warns that this model changed from its historical CLS
        # behavior to mean pooling. Mean pooling is the model card's declared
        # architecture and is intentionally what this adapter uses.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The model .* now uses mean pooling instead of CLS embedding.*",
                category=UserWarning,
            )
            self._model = TextEmbedding(**kwargs)
        return self._model

    def _normalize(self, value: Iterable[float]) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        expected = (self.dimensions,)
        if vector.shape != expected:
            raise ValueError(f"neural model returned {vector.shape}, expected {expected}")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def _remember(self, text: str, vector: np.ndarray) -> None:
        if self.cache_size == 0:
            return
        self._cache[text] = vector.copy()
        self._cache.move_to_end(text)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def encode_many(self, texts: Iterable[str]) -> list[np.ndarray]:
        values = list(texts)
        if not values:
            return []
        output: list[np.ndarray | None] = [None] * len(values)
        missing_positions: dict[str, list[int]] = {}

        with self._lock:
            for index, text in enumerate(values):
                if not text.strip():
                    raise ValueError("embedding text cannot be empty")
                cached = self._cache.get(text)
                if cached is not None:
                    self.cache_hits += 1
                    self._cache.move_to_end(text)
                    output[index] = cached.copy()
                else:
                    missing_positions.setdefault(text, []).append(index)

            if missing_positions:
                unique_texts = list(missing_positions)
                self.cache_misses += len(unique_texts)
                try:
                    model = self._load()
                    vectors = list(model.embed(unique_texts, batch_size=self.batch_size))
                    if len(vectors) != len(unique_texts):
                        raise RuntimeError("neural model returned the wrong batch size")
                    normalized = [self._normalize(vector) for vector in vectors]
                    self.neural_calls += len(unique_texts)
                    self.last_error = None
                except Exception as error:
                    self.last_error = f"{type(error).__name__}: {error}"
                    if self.fallback is None:
                        raise RuntimeError(
                            f"neural embedding failed for {self.model_name}: {error}"
                        ) from error
                    normalized = [self._normalize(self.fallback.encode(text)) for text in unique_texts]
                    self.fallback_calls += len(unique_texts)

                for text, vector in zip(unique_texts, normalized):
                    self._remember(text, vector)
                    for index in missing_positions[text]:
                        output[index] = vector.copy()

        if any(vector is None for vector in output):
            raise RuntimeError("encoder failed to populate every output position")
        return [vector for vector in output if vector is not None]

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "fastembed-onnx",
            "model": self.model_name,
            "dimensions": self.dimensions,
            "loaded": self._model is not None,
            "cache_entries": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "neural_calls": self.neural_calls,
            "fallback_calls": self.fallback_calls,
            "last_error": self.last_error,
        }


def create_neural_memory(
    *,
    config=None,
    model_name: str = DEFAULT_MODEL,
    dimensions: int | None = None,
    cache_dir: str | Path | None = None,
    archive_store=None,
    extractor=None,
    fallback_to_hash: bool = False,
):
    """Construct UMD with local neural retrieval and optional LLM extraction."""
    try:
        from .umd35_core import HashEncoder, UMD35Memory
    except ImportError:  # Direct script/module execution.
        from umd35_core import HashEncoder, UMD35Memory

    resolved_dimensions = dimensions or KNOWN_DIMENSIONS.get(model_name)
    if resolved_dimensions is None:
        raise ValueError("dimensions are required for an unregistered model")
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / ".model-cache"
    fallback = HashEncoder(resolved_dimensions) if fallback_to_hash else None
    encoder = FastEmbedEncoder(
        model_name,
        dimensions=resolved_dimensions,
        cache_dir=cache_dir,
        fallback=fallback,
    )
    return UMD35Memory(
        config,
        encoder=encoder,
        archive_store=archive_store,
        extractor=extractor,
    )
