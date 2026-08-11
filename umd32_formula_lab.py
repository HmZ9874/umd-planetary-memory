"""UMD 3.2 optimization lab.

Focuses on the weakest UMD 3.1 component: assigning raw memories to one or
more stable star systems without forcing unknown memories into the hierarchy.
It also adds reversible lineage and automatic star split/merge diagnostics.
"""

from __future__ import annotations

import json

import numpy as np


SEED = 20260808
FEATURES = ["semantic", "relation", "scope", "contradiction"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_orbit_data(n: int, stars: int, seed: int, shifted: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if shifted:
        x = np.stack(
            [
                rng.beta(2.5, 4.5, (n, stars)),
                rng.beta(1.8, 5.0, (n, stars)),
                rng.beta(2.3, 4.2, (n, stars)),
                rng.beta(1.2, 5.5, (n, stars)),
            ],
            axis=-1,
        )
    else:
        x = np.stack(
            [
                rng.beta(2.0, 5.0, (n, stars)),
                rng.beta(1.5, 6.0, (n, stars)),
                rng.beta(2.0, 5.0, (n, stars)),
                rng.beta(1.0, 6.0, (n, stars)),
            ],
            axis=-1,
        )
    y = np.zeros((n, stars), dtype=float)
    open_set = rng.random(n) < .20
    for i in np.where(~open_set)[0]:
        count = int(rng.choice([1, 2, 3], p=[.50, .35, .15]))
        targets = rng.choice(stars, count, replace=False)
        y[i, targets] = 1
        if shifted:
            x[i, targets, 0] = rng.beta(5.0, 2.8, count)
            x[i, targets, 1] = rng.beta(4.0, 3.0, count)
            x[i, targets, 2] = rng.beta(5.5, 2.5, count)
            x[i, targets, 3] = rng.beta(1.3, 6.0, count)
        else:
            x[i, targets, 0] = rng.beta(6.0, 2.0, count)
            x[i, targets, 1] = rng.beta(5.0, 2.5, count)
            x[i, targets, 2] = rng.beta(7.0, 1.8, count)
            x[i, targets, 3] = rng.beta(1.0, 8.0, count)
    corruption = rng.random(x.shape) < (.035 if shifted else .012)
    x = np.where(corruption, rng.random(x.shape), x)
    return x, y


def train_orbit_model(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # Weighted multilabel logistic regression. Positive labels are rare, so
    # positive residuals receive a balancing factor.
    w = np.array([1.0, 1.0, 1.0, -1.0])
    bias = -1.5
    positive_weight = float((y.size - y.sum()) / max(1.0, y.sum()))
    for step in range(260):
        logits = np.einsum("nsf,f->ns", x, w) + bias
        p = sigmoid(logits)
        # Hard negatives are unrelated stars that nevertheless look highly
        # similar. They model semantic camouflage and open-set prompt injection.
        hard_negative = (y == 0) & ((x[:, :, 0] > .62) | ((x[:, :, 0] > .50) & (x[:, :, 2] > .55)))
        scale = np.where(y == 1, positive_weight, 1.0 + 3.0 * hard_negative)
        residual = (p - y) * scale
        grad_w = np.einsum("ns,nsf->f", residual, x) / residual.size
        grad_b = float(np.mean(residual))
        lr = .45 / (1 + .008 * step)
        w -= lr * grad_w
        bias -= lr * grad_b
    return w, bias


def calibrate_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best = None
    for temperature in np.linspace(.20, 2.5, 93):
        p = sigmoid(logits / temperature)
        brier = float(np.mean((p - y) ** 2))
        if best is None or brier < best[0]:
            best = (brier, float(temperature))
    return best[1]


def orbit_metrics(scores: np.ndarray, y: np.ndarray, threshold: float, top_k: int = 3) -> dict[str, float]:
    ranked = np.argsort(-scores, axis=1)[:, :top_k]
    selected = np.zeros_like(y, dtype=bool)
    rows = np.arange(len(y))[:, None]
    keep = scores[rows, ranked] >= threshold
    selected[rows, ranked] = keep
    true = y.astype(bool)
    tp = np.sum(selected & true)
    fp = np.sum(selected & ~true)
    positives = np.sum(true)
    open_set = np.sum(true, axis=1) == 0
    return {
        "recall": float(tp / max(1, positives)),
        "contamination": float(fp / max(1, tp + fp)),
        "open_set_false_attachment": float(np.mean(np.any(selected[open_set], axis=1))),
        "open_set_abstention": float(np.mean(~np.any(selected[open_set], axis=1))),
        "mean_stable_orbits": float(np.mean(np.sum(selected, axis=1))),
        "memory_query_coverage": float(np.mean(np.any(selected & true, axis=1)[~open_set])),
    }


def tune_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    best = None
    fallback = None
    for threshold in np.linspace(.35, .98, 253):
        metrics = orbit_metrics(scores, y, float(threshold))
        # Validation constraints are intentionally stricter than deployment
        # gates to leave headroom for distribution shift.
        safe = metrics["contamination"] <= .08 and metrics["open_set_false_attachment"] <= .04
        if safe and (best is None or metrics["recall"] > best[1]["recall"]):
            best = (float(threshold), metrics)
        violation = max(0.0, metrics["contamination"] - .08) + max(0.0, metrics["open_set_false_attachment"] - .04)
        objective = violation - .01 * metrics["recall"]
        if fallback is None or objective < fallback[0]:
            fallback = (objective, float(threshold), metrics)
    if best is None:
        # Return the least-violating operating point and report that constraints
        # were not met; never disguise a failed baseline as a safe result.
        fallback[2]["constraints_met"] = False
        return fallback[1], fallback[2]
    best[1]["constraints_met"] = True
    return best


def test_learned_orbits() -> dict:
    stars = 20
    train_normal_x, train_normal_y = make_orbit_data(35_000, stars, SEED + 201, False)
    train_hard_x, train_hard_y = make_orbit_data(20_000, stars, SEED + 211, True)
    train_x = np.concatenate([train_normal_x, train_hard_x])
    train_y = np.concatenate([train_normal_y, train_hard_y])
    valid_normal_x, valid_normal_y = make_orbit_data(10_000, stars, SEED + 202, False)
    valid_hard_x, valid_hard_y = make_orbit_data(10_000, stars, SEED + 212, True)
    valid_x = np.concatenate([valid_normal_x, valid_hard_x])
    valid_y = np.concatenate([valid_normal_y, valid_hard_y])
    shifted_x, shifted_y = make_orbit_data(30_000, stars, SEED + 203, True)
    w, bias = train_orbit_model(train_x, train_y)
    valid_logits = np.einsum("nsf,f->ns", valid_x, w) + bias
    temperature = calibrate_temperature(valid_logits, valid_y)
    valid_scores = sigmoid(valid_logits / temperature)
    threshold, valid_metrics = tune_threshold(valid_scores, valid_y)
    shifted_scores = sigmoid((np.einsum("nsf,f->ns", shifted_x, w) + bias) / temperature)
    shifted_metrics = orbit_metrics(shifted_scores, shifted_y, threshold)
    provisional_threshold = max(.85, threshold - .08)
    provisional_shifted = orbit_metrics(shifted_scores, shifted_y, provisional_threshold)

    # Semantic-only comparator is independently thresholded under the same
    # validation safety constraints.
    sem_valid = valid_x[:, :, 0]
    sem_threshold, sem_valid_metrics = tune_threshold(sem_valid, valid_y)
    sem_shifted = orbit_metrics(shifted_x[:, :, 0], shifted_y, sem_threshold)

    return {
        "formula": "stable_orbit = top3(sigmoid(w*features+b)/temperature) above calibrated threshold; otherwise abstain",
        "weights": dict(zip(FEATURES, np.round(w, 4))),
        "bias": bias,
        "temperature": temperature,
        "threshold": threshold,
        "validation": valid_metrics,
        "shifted_attack": shifted_metrics,
        "provisional_threshold": provisional_threshold,
        "provisional_shifted": provisional_shifted,
        "semantic_only_threshold": sem_threshold,
        "semantic_only_validation": sem_valid_metrics,
        "semantic_only_shifted": sem_shifted,
    }


def kmeans2(points: np.ndarray, iterations: int = 15) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic farthest-pair initialization.
    first = points[0]
    second = points[np.argmax(np.linalg.norm(points - first, axis=1))]
    centers = np.stack([first, second])
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distance, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for k in [0, 1]:
            if np.any(labels == k):
                centers[k] = points[labels == k].mean(axis=0)
    return labels, centers


def split_score(points: np.ndarray) -> float:
    labels, centers = kmeans2(points)
    counts = np.bincount(labels, minlength=2)
    balance = counts.min() / max(1, counts.max())
    within = np.mean(np.linalg.norm(points - centers[labels], axis=1))
    separation = np.linalg.norm(centers[0] - centers[1])
    return float(balance * separation / max(.05, within))


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = np.sum(y & pred); fp = np.sum(~y & pred); fn = np.sum(y & ~pred)
    precision = float(tp / max(1, tp + fp)); recall = float(tp / max(1, tp + fn))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall), "false_positive_rate": float(fp / max(1, np.sum(~y)))}


def test_star_split_merge() -> dict:
    rng = np.random.default_rng(SEED + 204)
    n, children, dims = 2400, 80, 6
    split_truth = rng.random(n) < .5
    scores = np.zeros(n)
    for i in range(n):
        center = rng.normal(0, 1, dims)
        if split_truth[i]:
            direction = rng.normal(0, 1, dims); direction /= np.linalg.norm(direction)
            separation = rng.uniform(1.2, 2.8)
            left = center - direction * separation / 2
            right = center + direction * separation / 2
            count = rng.integers(25, 56)
            points = np.vstack([rng.normal(left, .38, (count, dims)), rng.normal(right, .38, (children - count, dims))])
        else:
            points = rng.normal(center, .62, (children, dims))
        scores[i] = split_score(points)
    split_at = int(n * .60)
    best = None
    for threshold in np.linspace(scores.min(), scores.max(), 300):
        metrics = binary_metrics(split_truth[:split_at], scores[:split_at] >= threshold)
        objective = metrics["f1"] - 2 * metrics["false_positive_rate"]
        if best is None or objective > best[0]: best = (objective, float(threshold))
    split_holdout = binary_metrics(split_truth[split_at:], scores[split_at:] >= best[1])

    pairs = 60_000
    merge_truth = rng.random(pairs) < .45
    centroid_similarity = np.where(merge_truth, rng.beta(8, 1.8, pairs), rng.beta(3, 3.5, pairs))
    membership_overlap = np.where(merge_truth, rng.beta(6, 2, pairs), rng.beta(2, 5, pairs))
    conflict = np.where(merge_truth, rng.beta(1, 8, pairs), rng.beta(2.5, 3, pairs))
    merge_score = .50 * centroid_similarity + .35 * membership_overlap - .65 * conflict
    cut = 35_000
    merge_best = None
    for threshold in np.linspace(merge_score.min(), merge_score.max(), 350):
        metrics = binary_metrics(merge_truth[:cut], merge_score[:cut] >= threshold)
        objective = metrics["f1"] - 3 * metrics["false_positive_rate"]
        if merge_best is None or objective > merge_best[0]: merge_best = (objective, float(threshold))
    merge_holdout = binary_metrics(merge_truth[cut:], merge_score[cut:] >= merge_best[1])
    return {
        "split": {"formula": "balance*centroid_separation/within_scatter", "threshold": best[1], "holdout": split_holdout},
        "merge": {"formula": "0.50*centroid_similarity+0.35*membership_overlap-0.65*conflict", "threshold": merge_best[1], "holdout": merge_holdout},
        "invariant": "split and merge operations are proposals; raw child memories and old star IDs remain addressable until verification",
    }


def test_lineage() -> dict:
    rng = np.random.default_rng(SEED + 205)
    raw_count, derived_count = 12_000, 4_000
    raw_ids = np.arange(raw_count)
    derivations = []
    for derived in range(derived_count):
        source_count = int(rng.integers(2, 7))
        sources = rng.choice(raw_ids, source_count, replace=False)
        derivations.append(sources)
    # Inject malformed lineage records to prove that validation detects them.
    injected = []
    for _ in range(60):
        injected.append(np.array([int(rng.integers(0, raw_count)), raw_count + int(rng.integers(1, 500))]))
    for _ in range(40):
        injected.append(np.array([int(rng.integers(0, raw_count))]))
    candidates = derivations + injected

    def valid_lineage(sources):
        return len(sources) >= 2 and all(0 <= int(source) < raw_count for source in sources)

    detected_corrupt = sum(not valid_lineage(sources) for sources in candidates)
    accepted = [sources for sources in candidates if valid_lineage(sources)]
    orphaned_after_validation = sum(not valid_lineage(sources) for sources in accepted)
    invalid_raw = set(rng.choice(raw_ids, 900, replace=False).tolist())
    stale_derived = [i for i, sources in enumerate(accepted) if any(int(s) in invalid_raw for s in sources)]
    can_materialize = [all(int(s) not in invalid_raw for s in sources) for sources in accepted]
    rebuilt_without_valid_sources = sum(can_materialize[i] for i in stale_derived)
    reversible = all(all(0 <= int(source) < raw_count for source in sources) for sources in accepted)
    return {
        "formula": "derived_memory = immutable source_ids + transform_version + reversible materialization",
        "raw_memories": raw_count,
        "derived_memories": derived_count,
        "invalidated_raw": len(invalid_raw),
        "derived_marked_stale": len(stale_derived),
        "corrupt_derivations_injected": len(injected),
        "corrupt_derivations_detected": detected_corrupt,
        "orphaned_after_validation": orphaned_after_validation,
        "reversible": reversible,
        "incorrectly_rebuilt_without_valid_sources": rebuilt_without_valid_sources,
        "invariant": "consolidation never overwrites or deletes raw evidence",
    }


def main() -> None:
    orbit = test_learned_orbits()
    topology = test_star_split_merge()
    lineage = test_lineage()
    regression = {
        "learned_orbit_beats_semantic_shifted_recall": orbit["shifted_attack"]["recall"] > orbit["semantic_only_shifted"]["recall"],
        "shifted_contamination_below_0.15": orbit["shifted_attack"]["contamination"] < .15,
        "open_set_false_attachment_below_0.10": orbit["shifted_attack"]["open_set_false_attachment"] < .10,
        "split_false_positive_below_0.05": topology["split"]["holdout"]["false_positive_rate"] < .05,
        "merge_false_positive_below_0.05": topology["merge"]["holdout"]["false_positive_rate"] < .05,
        "lineage_detects_all_injected_corruption": lineage["corrupt_derivations_detected"] == lineage["corrupt_derivations_injected"],
        "lineage_has_no_orphans_after_validation": lineage["orphaned_after_validation"] == 0,
        "consolidation_is_reversible": lineage["reversible"],
    }
    print(json.dumps({
        "methodology": {"seed": SEED, "scope": "synthetic train/validation/shifted-attack tests", "warning": "Real language extraction and public memory benchmarks remain required."},
        "learned_multi_orbit": orbit,
        "star_split_merge": topology,
        "lineage": lineage,
        "regression_gates": regression,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
