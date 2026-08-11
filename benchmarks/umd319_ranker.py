"""Serializable adapters for learned UMD ranking potentials."""

from __future__ import annotations

import numpy as np
import warnings


class LightGBMProbabilityRanker:
    """Expose a LambdaRank model through the probability-like UMD protocol."""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, rows):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Found 'eval_at' in params.*")
            score = np.asarray(self.model.predict(rows), dtype=np.float64)
        # Only within-query ordering matters; the caller normalizes this field.
        return np.column_stack([-score, score])


class BlendedProbabilityRanker:
    """Conserve old gravity while admitting a bounded learned meson field."""

    def __init__(
        self, old_model, meson_model, meson_weight: float = 0.35,
        old_dimensions: int = 34, conservation_budget: int = 0,
    ) -> None:
        self.old_model = old_model
        self.meson_model = meson_model
        self.meson_weight = min(1.0, max(0.0, float(meson_weight)))
        self.old_dimensions = int(old_dimensions)
        self.conservation_budget = min(4, max(0, int(conservation_budget)))
        self.broad_set_gate = False

    @staticmethod
    def _unit(score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64)
        low, high = float(score.min()), float(score.max())
        if high <= low:
            return np.ones_like(score)
        return (score - low) / (high - low)

    def predict_proba(self, rows):
        old = self.old_model.predict_proba(rows[:, :self.old_dimensions])[:, 1]
        meson = self.meson_model.predict_proba(rows)[:, 1]
        weight = (
            0.0 if getattr(self, "broad_set_gate", False) and rows[0, 15] >= 0.5
            else self.meson_weight
        )
        score = ((1.0 - weight) * self._unit(old)
                 + weight * self._unit(meson))
        return np.column_stack([-score, score])

    def conservation_scores(self, rows):
        """Return the frozen old-field score for provenance shadow slots."""
        return self.old_model.predict_proba(rows[:, :self.old_dimensions])[:, 1]
