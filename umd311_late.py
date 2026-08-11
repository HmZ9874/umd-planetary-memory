"""UMD 3.11 bounded late-interaction orbit reranker.

The coarse planetary retriever remains responsible for candidate generation.
Only a bounded orbit is token-encoded, and token vectors are request-local so
historical metadata and plaintext do not become another persistent RAM tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class LateEncoder(Protocol):
    def query_embed(self, texts: Sequence[str], **kwargs): ...
    def passage_embed(self, texts: Sequence[str], **kwargs): ...


@dataclass(frozen=True)
class UMD311LateConfig:
    candidate_limit: int = 64
    late_weight: float = 0.60
    force_weight: float = 0.30
    adjacent_weight: float = 0.10
    base_rank_weight: float = 0.10
    batch_size: int = 16


def _unit(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class UMD311LateInteractionReranker:
    """Rerank a bounded coarse orbit using ColBERT-style MaxSim."""

    def __init__(self, encoder: LateEncoder, *, config: UMD311LateConfig | None = None) -> None:
        self.encoder = encoder
        self.config = config or UMD311LateConfig()

    @staticmethod
    def _maxsim(query: np.ndarray, passage: np.ndarray) -> float:
        if not query.size or not passage.size:
            return 0.0
        return float(np.max(query @ passage.T, axis=1).sum())

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        base_order: Sequence[int],
        force: Sequence[float],
        groups: Sequence[int],
    ) -> list[int]:
        if not base_order:
            return []
        limit = min(self.config.candidate_limit, len(base_order))
        candidates = list(base_order[:limit])
        query_vector = list(self.encoder.query_embed([query], batch_size=1))[0]
        passage_vectors = list(self.encoder.passage_embed(
            [texts[index] for index in candidates], batch_size=self.config.batch_size
        ))
        late = _unit([
            self._maxsim(query_vector, passage) for passage in passage_vectors
        ])
        candidate_position = {index: position for position, index in enumerate(candidates)}
        adjacent: list[float] = []
        for index in candidates:
            values = [late[candidate_position[index]]]
            for neighbor in (index - 1, index + 1):
                position = candidate_position.get(neighbor)
                if position is not None and groups[neighbor] == groups[index]:
                    values.append(late[position])
            adjacent.append(max(values))
        force_unit = _unit([force[index] for index in candidates])
        denominator = max(1, len(candidates))
        scores = [
            self.config.late_weight * late[position]
            + self.config.force_weight * force_unit[position]
            + self.config.adjacent_weight * adjacent[position]
            + self.config.base_rank_weight * (1.0 - position / denominator)
            for position in range(len(candidates))
        ]
        head = [
            candidates[position]
            for position in sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        ]
        candidate_set = set(candidates)
        return head + [index for index in base_order if index not in candidate_set]


def load_answerai_colbert(cache_dir, *, threads: int = 4) -> UMD311LateInteractionReranker:
    """Load the optional 130 MB model only when late reranking is enabled."""
    from fastembed import LateInteractionTextEmbedding

    encoder = LateInteractionTextEmbedding(
        model_name="answerdotai/answerai-colbert-small-v1",
        cache_dir=str(cache_dir),
        threads=threads,
    )
    return UMD311LateInteractionReranker(encoder)
