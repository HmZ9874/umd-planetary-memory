"""Development-only score fusion for the UMD 3.21 meson field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def unit(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def evaluate(
    old, meson, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    weight: float, broad_gate: bool = False,
) -> dict:
    old_score = old.predict_proba(x[:, :34])[:, 1]
    meson_score = meson.predict_proba(x)[:, 1]
    cursor = any_hits = full_hits = found = total = 0
    for size in groups:
        stop = cursor + int(size)
        labels = y[cursor:stop]
        local_weight = 0.0 if broad_gate and x[cursor, 15] >= 0.5 else weight
        learned = ((1.0 - local_weight) * unit(old_score[cursor:stop])
                   + local_weight * unit(meson_score[cursor:stop]))
        final = 0.88 * unit(learned) + 0.12 * unit(x[cursor:stop, 14])
        chosen = np.argsort(-final, kind="stable")[:10]
        relevant = int(labels.sum())
        selected = int(labels[chosen].sum())
        any_hits += selected > 0
        full_hits += selected == relevant
        found += selected
        total += relevant
        cursor = stop
    queries = len(groups)
    return {
        "meson_weight": weight,
        "broad_gate": broad_gate,
        "r_at_1": any_hits / queries,
        "full_r_at_1": full_hits / queries,
        "micro_r_at_1": found / max(1, total),
        "utility": (0.60 * any_hits / queries + 0.25 * full_hits / queries
                    + 0.15 * found / max(1, total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--old-ranker", type=Path, required=True)
    parser.add_argument("--meson-ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.features)
    old, meson = joblib.load(args.old_ranker), joblib.load(args.meson_ranker)
    results = [
        evaluate(old, meson, data["x"], data["y"], data["group"], value / 20.0, gate)
        for gate in (False, True) for value in range(21)
    ]
    best = max(results, key=lambda row: (row["utility"], row["r_at_1"]))
    payload = {"development_only": True, "best": best, "results": results}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
