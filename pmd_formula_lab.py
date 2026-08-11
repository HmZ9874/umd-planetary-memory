"""Reproducible synthetic adversarial tests for Planetary Memory Dynamics.

The lab tests each formula as an engineering scoring rule, not as a claim that
semantic memory literally obeys celestial mechanics.  All datasets are
generated, split by seed, and evaluated again under a shifted attack
distribution.  Run with the bundled Codex Python runtime or Python + NumPy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


SEED = 20260808


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_rate": fp / max(1, int(np.sum(y == 0))),
    }


def test_formation_gate() -> dict:
    """Optimize the write-energy equation and verify a non-negotiable risk veto."""
    rng = np.random.default_rng(SEED + 1)
    # utility, novelty, explicit request, task impact, confidence, redundancy, risk
    x = rng.random((80_000, 7))
    latent = (
        1.8 * x[:, 0]
        + 1.1 * x[:, 1]
        + 2.2 * x[:, 2]
        + 1.7 * x[:, 3]
        + 0.8 * x[:, 4]
        - 1.0 * x[:, 5]
        - 2.0 * x[:, 6]
        + rng.normal(0, 0.35, len(x))
    )
    y = ((latent > 2.15) & (x[:, 6] < 0.82)).astype(int)
    split = 50_000
    train_x, test_x, train_y, test_y = x[:split], x[split:], y[:split], y[split:]
    signs = np.array([1, 1, 1, 1, 1, -1, -1], dtype=float)
    base = np.array([1.5, 1.0, 1.8, 1.5, 0.7, 0.9, 1.8])
    best = None
    for _ in range(2500):
        weights = signs * base * np.exp(rng.normal(0, 0.38, 7))
        scores = train_x @ weights
        for threshold in np.quantile(scores, np.linspace(0.25, 0.75, 15)):
            pred = ((scores > threshold) & (train_x[:, 6] < 0.82)).astype(int)
            metrics = classification_metrics(train_y, pred)
            unsafe = float(np.mean(pred[train_x[:, 6] >= 0.82]))
            # False writes pollute future retrieval, so they cost more than a
            # missed low-value write.  Unsafe writes remain a hard veto.
            objective = metrics["f1"] - 0.75 * metrics["false_positive_rate"] - 8.0 * unsafe
            if best is None or objective > best[0]:
                best = (objective, weights, float(threshold))
    assert best is not None
    weights, threshold = best[1], best[2]
    pred = (((test_x @ weights) > threshold) & (test_x[:, 6] < 0.82)).astype(int)
    metrics = classification_metrics(test_y, pred)
    metrics["unsafe_write_rate"] = float(np.mean(pred[test_x[:, 6] >= 0.82]))
    return {
        "formula": "write = risk_veto AND sigmoid(w*x - threshold)",
        "weights": dict(zip(["utility", "novelty", "explicit", "impact", "confidence", "redundancy", "risk"], np.round(weights, 4))),
        "threshold": threshold,
        "holdout": metrics,
    }


def make_retrieval_set(n: int, seed: int, shifted: bool) -> np.ndarray:
    """Candidates: true, stale, poison, wrong-scope, repeated falsehood, noise."""
    r = np.random.default_rng(seed)
    x = np.zeros((n, 6, 10))
    u = lambda lo, hi: r.uniform(lo, hi, n)
    # similarity, log-mass, confidence, trust, validity, scope, utility, evidence, risk, duplication
    x[:, 0, :] = np.c_[u(.58, .90), np.log1p(u(.50, .90)), u(.72, .99), u(.72, 1), u(.82, 1), u(.86, 1), u(.68, 1), u(.70, 1), u(0, .18), u(0, .22)]
    x[:, 1, :] = np.c_[u(.78, .997), np.log1p(u(.72, 1)), u(.78, .99), u(.68, 1), u(.01, .38 if shifted else .28), u(.86, 1), u(.08, .48), u(.68, 1), u(0, .22), u(.05, .35)]
    x[:, 2, :] = np.c_[u(.84, .999), np.log1p(u(.72, 1)), u(.50, .96), u(.02, .58 if shifted else .45), u(.65, 1), u(.75, 1), u(.20, .78), u(0, .50 if shifted else .38), u(.42, 1), u(.38, 1)]
    x[:, 3, :] = np.c_[u(.74, .99), np.log1p(u(.70, 1)), u(.72, .99), u(.65, 1), u(.75, 1), u(.01, .38 if shifted else .28), u(.20, .75), u(.55, 1), u(0, .28), u(.05, .35)]
    x[:, 4, :] = np.c_[u(.78, .995), np.log1p(u(.88, 1)), u(.68, .98), u(.25, .82 if shifted else .72), u(.55, 1), u(.72, 1), u(.20, .72), u(.02, .48 if shifted else .38), u(.20, .80), u(.72, 1)]
    x[:, 5, :] = np.c_[u(.18, .84), np.log1p(u(.05, .65)), u(.25, .88), u(.30, .90), u(.35, 1), u(.35, 1), u(.05, .60), u(.05, .75), u(0, .45), u(.05, .70)]
    sigma = .065 if shifted else .045
    x = np.clip(x + r.normal(0, sigma, x.shape), 0, 1)
    corruption = r.random(x.shape) < (.035 if shifted else .015)
    return np.where(corruption, r.random(x.shape), x)


def retrieval_accuracy(x: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(np.einsum("ncf,f->nc", x, weights).argmax(axis=1) == 0))


def test_distance_and_attraction() -> dict:
    rng = np.random.default_rng(SEED + 2)
    train = make_retrieval_set(8_000, SEED + 20, False)
    valid = make_retrieval_set(8_000, SEED + 21, False)
    shifted = make_retrieval_set(30_000, SEED + 22, True)
    signs = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1.])
    base = np.array([1.0, .3, .4, 1.5, 2.5, 2.5, 1.6, 2.0, 2.0, 1.8])
    finalists = []
    for _ in range(7000):
        w = signs * base * np.exp(rng.normal(0, .62, 10))
        finalists.append((retrieval_accuracy(train, w), w))
    finalists.sort(key=lambda z: z[0], reverse=True)
    w = max(finalists[:60], key=lambda z: retrieval_accuracy(valid, z[1]))[1]

    # Explicitly measure why inverse-distance gravity is rejected.
    d = np.geomspace(1e-6, 1.0, 1000)
    inverse = 1.0 / np.power(d + 1e-9, 1.35)
    bounded = sigmoid(6 * (1 - d) - 3)
    singularity_ratio = float(inverse.max() / inverse.min())
    bounded_ratio = float(bounded.max() / max(1e-12, bounded.min()))

    old = np.array([3.0, .8, .8, 1.0, 1.2, 1.2, .3, .3, -1.0, -.2])
    ablations = {}
    for index, name in [(6, "utility"), (7, "evidence"), (4, "validity"), (5, "scope")]:
        damaged = shifted.copy()
        damaged[:, :, index] = rng.random(damaged[:, :, index].shape)
        ablations[name + "_random"] = retrieval_accuracy(damaged, w)
    neutral = shifted.copy()
    neutral[:, :, 6] = .5
    ablations["utility_neutral"] = retrieval_accuracy(neutral, w)
    return {
        "formula": "attraction = hard_gate * sigmoid(weighted bounded features / temperature)",
        "feature_order": ["similarity", "log_mass", "confidence", "trust", "validity", "scope", "utility", "evidence", "risk", "duplication"],
        "weights": np.round(w, 4).tolist(),
        "accuracy": {"train": retrieval_accuracy(train, w), "validation": retrieval_accuracy(valid, w), "shifted_attack": retrieval_accuracy(shifted, w), "previous_soft": retrieval_accuracy(shifted, old)},
        "distance_stability": {"inverse_dynamic_range": singularity_ratio, "bounded_dynamic_range": bounded_ratio},
        "ablations": ablations,
    }


def test_mass_update() -> dict:
    """Search conservative evidence/utility/conflict rates against property tests."""
    best = None
    for evidence_rate in np.linspace(.01, .10, 19):
        for utility_rate in np.linspace(.02, .25, 24):
            for conflict_rate in np.linspace(.10, .60, 26):
                def update(m, independent, utility, conflict):
                    return float(np.clip(m + evidence_rate * math.log1p(independent) + utility_rate * utility - conflict_rate * conflict, 0, 1))
                # Repetition from an already-counted source adds zero independent evidence.
                repeated_same_source = update(.25, 0, 0, 0)
                five_independent = update(.25, 5, .3, 0)
                contradicted = update(.80, 0, 0, 1)
                useful = update(.40, 2, .8, 0)
                penalty = (
                    abs(repeated_same_source - .25)
                    + abs(five_independent - .48)
                    + abs(contradicted - .30)
                    + abs(useful - .60)
                )
                if best is None or penalty < best[0]:
                    best = (penalty, evidence_rate, utility_rate, conflict_rate, [repeated_same_source, five_independent, contradicted, useful])
    assert best
    return {
        "formula": "m' = clip(m*decay + a*log(1+independent_sources) + b*counterfactual_utility - c*conflict, 0, 1)",
        "parameters": {"independent_evidence": best[1], "counterfactual_utility": best[2], "conflict": best[3]},
        "property_outputs": dict(zip(["same_source_repeat", "five_independent", "contradicted", "useful"], best[4])),
    }


def test_orbit_mapping() -> dict:
    """Show that orbit radius is a bounded cache-tier visualization, not evidence."""
    mass = np.linspace(0, 1, 100_001)
    epsilon = 1e-6
    inverse_radius = 1.0 / (mass + epsilon)
    gamma = 1.25
    bounded_radius = 1.0 + 9.0 * np.power(1.0 - mass, gamma)
    # Every strictly monotone radius transform has exactly the same top-k set.
    k = 10_000
    top_mass = set(np.argpartition(-mass, k)[:k].tolist())
    top_inverse = set(np.argpartition(inverse_radius, k)[:k].tolist())
    top_bounded = set(np.argpartition(bounded_radius, k)[:k].tolist())
    return {
        "rejected_formula": "r = r_min + k/(mass+epsilon)",
        "replacement": "r = r_min + (r_max-r_min)*(1-mass)^gamma",
        "parameters": {"r_min": 1.0, "r_max": 10.0, "gamma": gamma},
        "inverse_dynamic_range": float(inverse_radius.max() / inverse_radius.min()),
        "bounded_dynamic_range": float(bounded_radius.max() / bounded_radius.min()),
        "top_k_equivalence": {
            "inverse_vs_mass": len(top_mass & top_inverse) / k,
            "bounded_vs_mass": len(top_mass & top_bounded) / k,
        },
        "decision": "radius must not be added back into retrieval scoring; it duplicates mass ordering",
    }


def test_decay() -> dict:
    """Find per-type half-lives minimizing loss on simulated future-use horizons."""
    rng = np.random.default_rng(SEED + 3)
    kinds = {"preference": 540.0, "event": 180.0, "temporary": 7.0}
    results = {"core": {"half_life_days": None, "rule": "no automatic decay"}}
    for kind, target_scale in kinds.items():
        ages = rng.exponential(target_scale, 40_000)
        needed = rng.random(len(ages)) < np.exp(-ages / target_scale)
        best = None
        for half_life in np.geomspace(1, 1500, 250):
            retained = np.exp(-math.log(2) * ages / half_life) >= .25
            false_drop = np.mean(needed & ~retained)
            stale_keep = np.mean(~needed & retained)
            loss = 4 * false_drop + stale_keep
            if best is None or loss < best[0]:
                best = (loss, half_life, false_drop, stale_keep)
        results[kind] = {"half_life_days": best[1], "false_drop": best[2], "stale_keep": best[3]}
    results["warning"] = "expiry and superseded flags are hard state transitions, not decay"
    return results


def test_collision() -> dict:
    rng = np.random.default_rng(SEED + 4)
    n = 80_000
    same = rng.random(n) < .45
    compatible = np.where(same, rng.random(n) < .96, rng.random(n) < .08)
    same_scope = np.where(same, rng.random(n) < .97, rng.random(n) < .35)
    similarity = np.where(same, rng.beta(12, 2, n), rng.beta(5, 4, n))
    best = None
    for threshold in np.linspace(.55, .98, 300):
        pred = (similarity >= threshold) & compatible & same_scope
        metrics = classification_metrics(same.astype(int), pred.astype(int))
        objective = metrics["f1"] - 4 * metrics["false_positive_rate"]
        if best is None or objective > best[0]:
            best = (objective, threshold, metrics)
    sim_only = similarity >= best[1]
    return {
        "formula": "merge iff similarity>=theta AND compatible AND same_scope; conflicts are versioned",
        "similarity_threshold": best[1],
        "guarded": best[2],
        "similarity_only": classification_metrics(same.astype(int), sim_only.astype(int)),
    }


def test_consolidation() -> dict:
    """Optimize conservative gates for turning episodes into a derived summary."""
    rng = np.random.default_rng(SEED + 6)
    n = 100_000
    evidence_count = rng.integers(1, 12, n)
    coherence = rng.beta(5, 2.2, n)
    conflict = rng.beta(1.2, 7, n)
    source_diversity = rng.beta(3, 2, n)
    # Ground truth is deliberately noisy and not identical to the tested gate.
    # This avoids the tautology of generating labels from the same thresholds
    # that the optimizer is asked to recover.
    latent = (
        .45 * np.log1p(evidence_count)
        + 2.2 * coherence
        - 2.6 * conflict
        + 1.0 * source_diversity
        + rng.normal(0, .25, n)
    )
    safe = (latent >= 2.15) & (evidence_count >= 2)
    split = 60_000
    best = None
    for min_evidence in range(2, 7):
        for min_coherence in np.linspace(.55, .90, 36):
            for max_conflict in np.linspace(.05, .30, 26):
                pred = (evidence_count >= min_evidence) & (coherence >= min_coherence) & (conflict <= max_conflict) & (source_diversity >= .35)
                metrics = classification_metrics(safe[:split].astype(int), pred[:split].astype(int))
                objective = metrics["f1"] - 3.0 * metrics["false_positive_rate"]
                if best is None or objective > best[0]:
                    best = (objective, min_evidence, min_coherence, max_conflict)
    assert best
    holdout_pred = (evidence_count[split:] >= best[1]) & (coherence[split:] >= best[2]) & (conflict[split:] <= best[3]) & (source_diversity[split:] >= .35)
    holdout = classification_metrics(safe[split:].astype(int), holdout_pred.astype(int))
    return {
        "formula": "consolidate iff evidence>=n AND coherence>=c AND conflict<=k AND source_diversity>=d",
        "parameters": {"min_evidence": best[1], "min_coherence": best[2], "max_conflict": best[3], "min_source_diversity": .35},
        "holdout": holdout,
        "invariant": "consolidation creates a derived summary and never overwrites raw episodes",
    }


def test_context_selection() -> dict:
    """Compare relevance top-k with diversity-aware greedy selection."""
    rng = np.random.default_rng(SEED + 5)
    trials, candidates, budget = 2500, 24, 900
    lambdas = np.linspace(0, 1.5, 31)
    totals = np.zeros_like(lambdas)
    baseline = 0.0
    for _ in range(trials):
        topic = rng.integers(0, 6, candidates)
        relevance = rng.beta(3, 2, candidates)
        tokens = rng.integers(60, 260, candidates)
        similarity = (topic[:, None] == topic[None, :]).astype(float) * rng.uniform(.65, .98, (candidates, candidates))
        np.fill_diagonal(similarity, 0)

        def utility(selected):
            if not selected:
                return 0.0
            coverage = len(set(topic[selected])) / 6
            redundancy = sum(similarity[i, j] for pos, i in enumerate(selected) for j in selected[pos + 1:]) / max(1, len(selected))
            return float(sum(relevance[selected]) + 1.2 * coverage - .45 * redundancy)

        order = list(np.argsort(-relevance))
        top, used = [], 0
        for i in order:
            if used + tokens[i] <= budget:
                top.append(i); used += tokens[i]
        baseline += utility(top)
        for li, lam in enumerate(lambdas):
            chosen, used = [], 0
            while True:
                feasible = [i for i in range(candidates) if i not in chosen and used + tokens[i] <= budget]
                if not feasible: break
                def gain(i):
                    redundancy = max([similarity[i, j] for j in chosen], default=0)
                    new_topic = 1.0 if topic[i] not in {topic[j] for j in chosen} else 0.0
                    return relevance[i] - lam * redundancy + .20 * new_topic
                pick = max(feasible, key=gain)
                chosen.append(pick); used += tokens[pick]
            totals[li] += utility(chosen)
    best_index = int(np.argmax(totals))
    return {
        "formula": "maximize attraction - lambda*redundancy + coverage under token budget",
        "redundancy_lambda": float(lambdas[best_index]),
        "mean_utility_topk": baseline / trials,
        "mean_utility_optimized": totals[best_index] / trials,
        "relative_gain": float(totals[best_index] / baseline - 1),
    }


def memory_overhead() -> dict:
    n, dims, edges = 1_000_000, 768, 4
    vector_bytes = n * dims * 2
    state_bytes = n * 96
    edge_bytes = n * edges * 24
    return {
        "records": n,
        "fp16_vector_gb": vector_bytes / 1e9,
        "pmd_state_and_edges_gb": (state_bytes + edge_bytes) / 1e9,
        "increment_over_raw_vectors": (state_bytes + edge_bytes) / vector_bytes,
        "note": "excludes text, ANN index, database allocator overhead, and model weights",
    }


def main() -> None:
    formation = test_formation_gate()
    attraction = test_distance_and_attraction()
    mass = test_mass_update()
    orbit = test_orbit_mapping()
    decay = test_decay()
    collision = test_collision()
    consolidation = test_consolidation()
    context = test_context_selection()
    regression = {
        "formation_f1_at_least_0.90": formation["holdout"]["f1"] >= .90,
        "unsafe_write_rate_zero": formation["holdout"]["unsafe_write_rate"] == 0,
        "shifted_retrieval_at_least_0.95": attraction["accuracy"]["shifted_attack"] >= .95,
        "bounded_distance_dynamic_range_below_100": attraction["distance_stability"]["bounded_dynamic_range"] < 100,
        "same_source_does_not_reinforce": mass["property_outputs"]["same_source_repeat"] == .25,
        "orbit_is_bounded": orbit["bounded_dynamic_range"] <= 10,
        "collision_false_positive_below_0.02": collision["guarded"]["false_positive_rate"] < .02,
        "consolidation_false_positive_below_0.10": consolidation["holdout"]["false_positive_rate"] < .10,
        "context_utility_improves": context["relative_gain"] > 0,
    }
    report = {
        "methodology": {
            "seed": SEED,
            "scope": "synthetic adversarial/property tests",
            "warning": "Parameters are prototype priors, not production calibration.",
        },
        "formation_gate": formation,
        "distance_and_attraction": attraction,
        "mass_update": mass,
        "orbit_mapping": orbit,
        "decay": decay,
        "collision": collision,
        "consolidation": consolidation,
        "context_selection": context,
        "memory_overhead": memory_overhead(),
        "regression_gates": regression,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
