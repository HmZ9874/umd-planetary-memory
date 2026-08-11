"""Compose frozen UMD rankers without retraining or evaluator access."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from benchmarks.umd319_ranker import BlendedProbabilityRanker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ranker", type=Path, required=True)
    parser.add_argument("--meson-ranker", type=Path, required=True)
    parser.add_argument("--meson-weight", type=float, default=0.35)
    parser.add_argument("--conservation-budget", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = BlendedProbabilityRanker(
        joblib.load(args.old_ranker), joblib.load(args.meson_ranker),
        args.meson_weight, conservation_budget=args.conservation_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(
        f"saved {args.output} with meson_weight={model.meson_weight:.2f} "
        f"conservation_budget={model.conservation_budget}"
    )


if __name__ == "__main__":
    main()
