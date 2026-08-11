"""UMD 3.3 robustness lab.

Adds genuinely unseen coherent open-set attacks, provenance/density features,
and executable permissions for stable, provisional, quarantined, and floating
memories.  It imports the corrected topology and lineage tests from UMD 3.2.
"""

from __future__ import annotations

import json

import numpy as np

from umd32_formula_lab import binary_metrics, test_lineage, test_star_split_merge


SEED = 20260808
FEATURES = [
    "semantic",
    "relation",
    "scope",
    "contradiction",
    "source_trust",
    "neighborhood_density",
    "known_star_novelty",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_data(n: int, stars: int, seed: int, regime: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hard = regime in {"hard", "coherent_open", "calibration_drift"}
    x = np.stack(
        [
            rng.beta(2.5 if hard else 2.0, 4.5 if hard else 5.0, (n, stars)),
            rng.beta(1.8 if hard else 1.5, 5.0 if hard else 6.0, (n, stars)),
            rng.beta(2.3 if hard else 2.0, 4.2 if hard else 5.0, (n, stars)),
            rng.beta(1.2, 5.5, (n, stars)),
            rng.beta(1.8, 5.5, (n, stars)),
            rng.beta(1.6, 6.0, (n, stars)),
            rng.beta(5.0, 2.0, (n, stars)),
        ],
        axis=-1,
    )
    y = np.zeros((n, stars), dtype=float)
    open_set = rng.random(n) < .20
    for row in np.where(~open_set)[0]:
        count = int(rng.choice([1, 2, 3], p=[.50, .35, .15]))
        target = rng.choice(stars, count, replace=False)
        y[row, target] = 1
        x[row, target, 0] = rng.beta(5.0 if hard else 6.0, 2.8 if hard else 2.0, count)
        x[row, target, 1] = rng.beta(4.0 if hard else 5.0, 3.0 if hard else 2.5, count)
        x[row, target, 2] = rng.beta(5.5 if hard else 7.0, 2.5 if hard else 1.8, count)
        x[row, target, 3] = rng.beta(1.3, 6.0, count)
        x[row, target, 4] = rng.beta(5.5 if hard else 7.0, 2.2, count)
        x[row, target, 5] = rng.beta(5.0 if hard else 7.0, 2.4, count)
        x[row, target, 6] = rng.beta(1.5, 6.0, count)

    # These coherent unknowns are never included in training or calibration.
    # They imitate a known topic in semantic/relation/scope space but lack
    # trusted provenance and neighborhood support inside the known star.
    if regime in {"coherent_open", "calibration_drift"}:
        for row in np.where(open_set)[0]:
            fake = rng.choice(stars, int(rng.integers(1, 3)), replace=False)
            count = len(fake)
            x[row, fake, 0] = rng.beta(5.0, 2.8, count)
            x[row, fake, 1] = rng.beta(4.0, 3.0, count)
            x[row, fake, 2] = rng.beta(5.5, 2.5, count)
            x[row, fake, 3] = rng.beta(1.3, 6.0, count)
            x[row, fake, 4] = rng.beta(1.5, 6.0, count)
            x[row, fake, 5] = rng.beta(1.3, 7.0, count)
            x[row, fake, 6] = rng.beta(6.0, 1.8, count)

    if regime == "calibration_drift":
        negative = y == 0
        x[:, :, 1] = np.clip(x[:, :, 1] + .08 * negative, 0, 1)
        x[:, :, 2] = np.clip(x[:, :, 2] + .08 * negative, 0, 1)
        x[:, :, 4] = np.clip(x[:, :, 4] + .04 * negative, 0, 1)

    corruption = rng.random(x.shape) < (.035 if hard else .012)
    return np.where(corruption, rng.random(x.shape), x), y


def train_model(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    w = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    bias = -2.0
    positive_weight = float((y.size - y.sum()) / max(1.0, y.sum()))
    for step in range(320):
        logits = np.einsum("nsf,f->ns", x, w) + bias
        probability = sigmoid(logits)
        coherent_negative = (y == 0) & (x[:, :, 0] > .58) & ((x[:, :, 1] > .45) | (x[:, :, 2] > .50))
        provenance_negative = (y == 0) & ((x[:, :, 4] < .30) | (x[:, :, 5] < .25))
        scale = np.where(y == 1, positive_weight, 1.0 + 3.0 * coherent_negative + 1.5 * provenance_negative)
        residual = (probability - y) * scale
        grad = np.einsum("ns,nsf->f", residual, x) / residual.size
        grad_bias = float(np.mean(residual))
        lr = .40 / (1 + .007 * step)
        w -= lr * grad
        bias -= lr * grad_bias
    return w, bias


def calibrate(logits: np.ndarray, y: np.ndarray) -> float:
    # Balanced Brier score prevents the many negative labels from dominating
    # probability calibration.
    positives = y == 1
    negatives = ~positives
    best = None
    for temperature in np.linspace(.20, 2.5, 93):
        p = sigmoid(logits / temperature)
        score = .5 * np.mean((p[positives] - 1) ** 2) + .5 * np.mean(p[negatives] ** 2)
        if best is None or score < best[0]: best = (float(score), float(temperature))
    return best[1]


def selections(scores: np.ndarray, threshold: float, top_k: int = 3) -> np.ndarray:
    ranked = np.argsort(-scores, axis=1)[:, :top_k]
    selected = np.zeros_like(scores, dtype=bool)
    rows = np.arange(len(scores))[:, None]
    selected[rows, ranked] = scores[rows, ranked] >= threshold
    return selected


def metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    selected = selections(scores, threshold)
    truth = y.astype(bool)
    open_set = truth.sum(axis=1) == 0
    tp = np.sum(selected & truth); fp = np.sum(selected & ~truth)
    return {
        "recall": float(tp / max(1, np.sum(truth))),
        "contamination": float(fp / max(1, tp + fp)),
        "open_set_false_attachment": float(np.mean(np.any(selected[open_set], axis=1))),
        "open_set_abstention": float(np.mean(~np.any(selected[open_set], axis=1))),
        "memory_query_coverage": float(np.mean(np.any(selected & truth, axis=1)[~open_set])),
        "mean_orbits": float(np.mean(selected.sum(axis=1))),
    }


def choose_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    feasible = []
    all_points = []
    for threshold in np.linspace(.40, .995, 240):
        result = metrics(scores, y, float(threshold))
        violation = max(0.0, result["contamination"] - .08) + max(0.0, result["open_set_false_attachment"] - .04)
        all_points.append((violation - .01 * result["recall"], float(threshold), result))
        if violation == 0:
            feasible.append((result["recall"], float(threshold), result))
    if feasible:
        chosen = max(feasible)
        chosen[2]["constraints_met"] = True
        return chosen[1], chosen[2]
    fallback = min(all_points)
    fallback[2]["constraints_met"] = False
    return fallback[1], fallback[2]


def test_orbits() -> dict:
    stars = 20
    normal_x, normal_y = make_data(35_000, stars, SEED + 301, "normal")
    hard_x, hard_y = make_data(20_000, stars, SEED + 302, "hard")
    train_x = np.concatenate([normal_x, hard_x]); train_y = np.concatenate([normal_y, hard_y])
    valid_x, valid_y = make_data(20_000, stars, SEED + 303, "hard")
    camouflage_x, camouflage_y = make_data(30_000, stars, SEED + 304, "coherent_open")
    drift_x, drift_y = make_data(30_000, stars, SEED + 305, "calibration_drift")
    w, bias = train_model(train_x, train_y)
    valid_logits = np.einsum("nsf,f->ns", valid_x, w) + bias
    temperature = calibrate(valid_logits, valid_y)
    threshold, validation = choose_threshold(sigmoid(valid_logits / temperature), valid_y)

    def evaluate(x, y):
        return metrics(sigmoid((np.einsum("nsf,f->ns", x, w) + bias) / temperature), y, threshold)

    return {
        "formula": "top3 sigmoid(weighted semantic+relation+scope+provenance+density-novelty-contradiction), with abstention",
        "weights": dict(zip(FEATURES, np.round(w, 4))),
        "bias": bias,
        "temperature": temperature,
        "stable_threshold": threshold,
        "validation": validation,
        "unseen_coherent_open": evaluate(camouflage_x, camouflage_y),
        "unseen_calibration_drift": evaluate(drift_x, drift_y),
    }


PERMISSIONS = {
    "stable": {"retrieve", "propagate", "propose_consolidation", "update_parent_mass", "propose_fact_update"},
    "provisional": {"retrieve_capped", "request_confirmation", "collect_independent_evidence"},
    "quarantined": {"inspect_metadata", "request_confirmation"},
    "floating": {"raw_search", "request_classification"},
    "invalidated": {"audit", "archive"},
}


def authorize(state: str, action: str) -> bool:
    return action in PERMISSIONS.get(state, set())


def promote_provisional(probability: float, independent_sources: int, scope_confirmed: bool, explicit_confirmation: bool) -> bool:
    return explicit_confirmation or (probability >= .98 and independent_sources >= 2 and scope_confirmed)


def test_policy() -> dict:
    dangerous = {"propose_consolidation", "update_parent_mass", "propose_fact_update", "propagate"}
    provisional_leaks = sum(authorize("provisional", action) for action in dangerous)
    quarantine_leaks = sum(authorize("quarantined", action) for action in dangerous)
    rng = np.random.default_rng(SEED + 306)
    trials = 100_000
    probability = rng.random(trials)
    sources = rng.integers(0, 5, trials)
    scope = rng.random(trials) < .75
    explicit = rng.random(trials) < .02
    promoted = np.array([promote_provisional(float(probability[i]), int(sources[i]), bool(scope[i]), bool(explicit[i])) for i in range(trials)])
    unsafe = promoted & ~explicit & ((probability < .98) | (sources < 2) | ~scope)
    return {
        "provisional_dangerous_action_leaks": int(provisional_leaks),
        "quarantine_dangerous_action_leaks": int(quarantine_leaks),
        "unsafe_promotions": int(np.sum(unsafe)),
        "promotion_rate": float(np.mean(promoted)),
        "permissions": {key: sorted(value) for key, value in PERMISSIONS.items()},
    }


def main() -> None:
    orbit = test_orbits()
    policy = test_policy()
    topology = test_star_split_merge()
    lineage = test_lineage()
    regression = {
        "coherent_open_false_attachment_below_0.10": orbit["unseen_coherent_open"]["open_set_false_attachment"] < .10,
        "drift_false_attachment_below_0.15": orbit["unseen_calibration_drift"]["open_set_false_attachment"] < .15,
        "drift_contamination_below_0.20": orbit["unseen_calibration_drift"]["contamination"] < .20,
        "provisional_policy_has_no_dangerous_leaks": policy["provisional_dangerous_action_leaks"] == 0,
        "quarantine_policy_has_no_dangerous_leaks": policy["quarantine_dangerous_action_leaks"] == 0,
        "promotion_guards_hold": policy["unsafe_promotions"] == 0,
        "lineage_detects_corruption": lineage["corrupt_derivations_detected"] == lineage["corrupt_derivations_injected"],
        "split_merge_remain_conservative": topology["split"]["holdout"]["false_positive_rate"] < .05 and topology["merge"]["holdout"]["false_positive_rate"] < .05,
    }
    print(json.dumps({
        "methodology": {"seed": SEED, "training_regimes": ["normal", "hard"], "unseen_test_regimes": ["coherent_open", "calibration_drift"], "warning": "Still synthetic; raw-language extraction is not implemented."},
        "robust_orbits": orbit,
        "executable_policy": policy,
        "star_split_merge": topology,
        "lineage": lineage,
        "regression_gates": regression,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
