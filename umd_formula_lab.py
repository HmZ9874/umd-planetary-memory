"""UMD 3.1 synthetic optimization lab.

Extends the PMD 2.2 scoring layer with hierarchical universe/star/planet
routing, multi-orbit membership, uncertainty-aware confidence, bitemporal
facts, bounded activation budgets, typed relationship propagation, quarantine,
lifecycle guards, star-size homeostasis, and procedural memory.

The tests are deterministic synthetic stress tests.  They reject unsafe
formula shapes and provide prototype priors; they do not establish real-world
benchmark performance.
"""

from __future__ import annotations

import json
import math

import numpy as np


SEED = 20260808


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def test_multi_orbit() -> dict:
    rng = np.random.default_rng(SEED + 101)
    memories, stars = 50_000, 24
    logits = rng.normal(0, 1, (memories, stars))
    memberships = np.empty((memories, 3), dtype=int)
    for i in range(memories):
        memberships[i] = rng.choice(stars, 3, replace=False)
    rows = np.arange(memories)
    logits[rows, memberships[:, 0]] += 2.6
    logits[rows, memberships[:, 1]] += 1.9
    logits[rows, memberships[:, 2]] += 1.35
    query_slot = rng.integers(0, 3, memories)
    query_star = memberships[rows, query_slot]
    ranked = np.argsort(-logits, axis=1)
    candidates = []
    true_sets = [set(x.tolist()) for x in memberships]
    for k in range(1, 7):
        selected = ranked[:, :k]
        coverage = float(np.mean(np.any(selected == query_star[:, None], axis=1)))
        contamination = float(np.mean([sum(int(v not in true_sets[i]) for v in selected[i]) / k for i in range(memories)]))
        objective = coverage - .55 * contamination - .035 * k
        candidates.append((objective, k, coverage, contamination))
    best = max(candidates)
    weights = softmax(logits, temperature=.8)
    # At most three candidate orbits are retained. Only sufficiently probable
    # assignments become stable; the rest remain provisional asteroids.
    selected = ranked[:, :best[1]]
    best_stable = None
    for threshold in np.linspace(.03, .30, 55):
        stable_mask = np.take_along_axis(weights, selected, axis=1) >= threshold
        stable_coverage = float(np.mean(np.any((selected == query_star[:, None]) & stable_mask, axis=1)))
        wrong = []
        counts = []
        for i in range(memories):
            stable = selected[i][stable_mask[i]]
            wrong.extend([v not in true_sets[i] for v in stable])
            counts.append(len(stable))
        contamination = float(np.mean(wrong)) if wrong else 0.0
        # Stable orbits must meet a strict contamination constraint.
        if contamination <= .20 and (best_stable is None or stable_coverage > best_stable[0]):
            best_stable = (stable_coverage, float(threshold), contamination, float(np.mean(counts)))
    assert best_stable
    retained_mass = float(np.mean(np.take_along_axis(weights, selected, axis=1).sum(axis=1)))
    return {
        "formula": "pi(memory,star)=softmax(alpha*semantic+beta*relation+gamma*scope-delta*conflict)",
        "selected_orbits": best[1],
        "single_parent_query_coverage": candidates[0][2],
        "multi_orbit_query_coverage": best[2],
        "multi_orbit_contamination": best[3],
        "stable_orbit_probability_threshold": best_stable[1],
        "stable_orbit_query_coverage": best_stable[0],
        "stable_orbit_contamination": best_stable[2],
        "mean_stable_orbits": best_stable[3],
        "mean_probability_mass_retained": retained_mass,
        "temperature": .8,
    }


def test_uncertainty() -> dict:
    rng = np.random.default_rng(SEED + 102)
    n = 150_000
    true_p = rng.beta(2.2, 2.2, n)
    observations = rng.integers(1, 45, n)
    successes = rng.binomial(observations, true_p)
    a, b = 1 + successes, 1 + observations - successes
    mean = a / (a + b)
    variance = a * b / (((a + b) ** 2) * (a + b + 1))
    std = np.sqrt(variance)
    unsafe = true_p < .55
    useful = true_p >= .72
    threshold = .70

    def metrics(accept):
        false_accept = float(np.mean(accept & unsafe))
        useful_recall = float(np.sum(accept & useful) / max(1, np.sum(useful)))
        return false_accept, useful_recall

    mean_metrics = metrics(mean >= threshold)
    ordinary = None
    high_stakes = None
    for z in np.linspace(0, 3, 121):
        accept = mean - z * std >= threshold
        fa, recall = metrics(accept)
        objective = recall - 12 * fa
        candidate = (objective, float(z), fa, recall, float(np.mean(accept)))
        if ordinary is None or objective > ordinary[0]: ordinary = candidate
        if fa <= .002 and (high_stakes is None or recall > high_stakes[3]): high_stakes = candidate
    assert ordinary and high_stakes
    return {
        "formula": "confidence_lower_bound = beta_mean - z*beta_std",
        "mean_only": {"unsafe_accept_rate": mean_metrics[0], "useful_recall": mean_metrics[1]},
        "ordinary": {"z": ordinary[1], "unsafe_accept_rate": ordinary[2], "useful_recall": ordinary[3], "acceptance_rate": ordinary[4]},
        "high_stakes": {"z": high_stakes[1], "unsafe_accept_rate": high_stakes[2], "useful_recall": high_stakes[3], "acceptance_rate": high_stakes[4]},
    }


def test_bitemporal() -> dict:
    rng = np.random.default_rng(SEED + 103)
    entities, versions = 30_000, 5
    valid_from = np.tile(np.arange(versions) * 20.0, (entities, 1))
    valid_to = np.c_[valid_from[:, 1:], np.full(entities, 100.0)]
    ingested = valid_from + rng.uniform(0, 15, (entities, versions))
    event_time = rng.uniform(0, 100, entities)
    knowledge_time = np.minimum(100, event_time + rng.uniform(0, 25, entities))
    valid = (valid_from <= event_time[:, None]) & (event_time[:, None] < valid_to)
    known = ingested <= knowledge_time[:, None]
    eligible = valid & known
    has_answer = np.any(eligible, axis=1)
    # Independent row-wise oracle. The implementation below is vectorized;
    # keeping the oracle separate makes mutation tests meaningful.
    oracle = np.full(entities, -1, dtype=int)
    for entity in range(entities):
        candidates = [
            version for version in range(versions)
            if valid_from[entity, version] <= event_time[entity] < valid_to[entity, version]
            and ingested[entity, version] <= knowledge_time[entity]
        ]
        if candidates:
            oracle[entity] = max(candidates, key=lambda version: valid_from[entity, version])
    recency_scores = np.where(known, ingested, -np.inf)
    recency_pick = np.argmax(recency_scores, axis=1)
    bitemporal_scores = np.where(eligible, valid_from, -np.inf)
    bitemporal_pick = np.argmax(bitemporal_scores, axis=1)
    return {
        "formula": "eligible = event_time in [valid_from,valid_to) AND ingested_at<=knowledge_cutoff",
        "queries": int(np.sum(has_answer)),
        "ingestion_recency_accuracy": float(np.mean(recency_pick[has_answer] == oracle[has_answer])),
        "bitemporal_accuracy": float(np.mean(bitemporal_pick[has_answer] == oracle[has_answer])),
        "oracle_implementation_disagreements": int(np.sum(bitemporal_pick[has_answer] != oracle[has_answer])),
        "state_rule": "expired and superseded are explicit transitions, never inferred from age alone",
    }


def test_activation_budget() -> dict:
    rng = np.random.default_rng(SEED + 104)
    budget = 12.0
    results = {}
    for n in [100, 1_000, 10_000, 100_000]:
        logits = rng.normal(-1.0, 1.2, n)
        raw = sigmoid(logits)
        allocated = budget * softmax(logits)
        k = min(64, n)
        indices = np.argpartition(-allocated, k - 1)[:k]
        sparse = np.zeros(n)
        sparse[indices] = allocated[indices]
        sparse *= budget / max(1e-12, sparse.sum())
        results[str(n)] = {
            "raw_activation_sum": float(raw.sum()),
            "budgeted_sum": float(allocated.sum()),
            "sparse_budgeted_sum": float(sparse.sum()),
            "active_memories": int(np.sum(sparse > 0)),
        }
    return {
        "formula": "activation_i = budget*softmax(logit_i/temperature), followed by sparse top-k",
        "budget": budget,
        "top_k": 64,
        "scaling": results,
    }


def relation_trial(rng: np.random.Generator, eta: float, steps: int, typed: bool) -> float:
    nodes = 32
    causal = np.zeros((nodes, nodes))
    semantic = np.zeros((nodes, nodes))
    contradiction = np.zeros((nodes, nodes))
    causal[0, 1] = causal[1, 2] = causal[2, 3] = 1
    # Semantically similar distractor hub and spokes.
    for j in range(8, 20):
        semantic[0, j] = rng.uniform(.6, 1)
        semantic[j, 20] = rng.uniform(.5, 1)
    contradiction[2, 21] = 1
    noise = rng.random((nodes, nodes)) < .025
    semantic += noise * rng.uniform(0, .4, (nodes, nodes))
    def normalize(matrix):
        row_sum = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    if typed:
        # Normalize each relation channel independently so many weak semantic
        # edges cannot drown out one strong causal edge.
        transition = 1.0 * normalize(causal) + .12 * normalize(semantic)
    else:
        transition = normalize(causal + semantic + contradiction)
    base = np.zeros(nodes)
    base[0] = 1
    base[8:20] = rng.uniform(.05, .18, 12)
    activation = base.copy()
    for _ in range(steps):
        activation = (1 - eta) * base + eta * (activation @ transition)
        if typed:
            activation -= eta * .55 * (activation @ normalize(contradiction))
        activation = np.clip(activation, 0, None)
    top = set(np.argsort(-activation)[:6].tolist())
    return len(top & {1, 2, 3}) / 3


def test_typed_relations() -> dict:
    rng = np.random.default_rng(SEED + 105)
    best = None
    for eta in np.linspace(.08, .45, 20):
        for steps in range(1, 5):
            scores = [relation_trial(rng, float(eta), steps, True) for _ in range(120)]
            score = float(np.mean(scores)) - .005 * steps
            if best is None or score > best[0]:
                best = (score, float(eta), steps, float(np.mean(scores)))
    assert best
    typed = np.mean([relation_trial(rng, best[1], best[2], True) for _ in range(1000)])
    untyped = np.mean([relation_trial(rng, best[1], best[2], False) for _ in range(1000)])
    return {
        "formula": "A(k+1)=(1-eta)A0+eta*sum_r(theta_r*P_r*A(k)), with signed contradiction edges",
        "eta": best[1],
        "steps": best[2],
        "typed_chain_recall_at_6": float(typed),
        "untyped_chain_recall_at_6": float(untyped),
        "relation_weights": {"causal": 1.0, "semantic": .12, "contradiction": -.55},
    }


def test_quarantine() -> dict:
    rng = np.random.default_rng(SEED + 106)
    n = 80_000
    trusted = rng.uniform(.62, .92, n)
    poison = rng.uniform(.82, .999, n)
    no_quarantine_attack = float(np.mean(poison > trusted))
    poison_cap = .45
    quarantined_attack = float(np.mean(np.minimum(poison, poison_cap) > trusted))
    same_source_repeats = rng.integers(1, 1000, n)
    independent_sources = rng.integers(0, 5, n)

    def promote(source_ids, explicitly_confirmed=False):
        return explicitly_confirmed or len(set(source_ids)) >= 2

    sample = 12_000
    same_source_promotions = [promote(["source-A"] * int(count)) for count in same_source_repeats[:sample]]
    independent_promotions = [promote([f"source-{j}" for j in range(int(count))]) for count in independent_sources[:sample]]
    naive_same_source_false_release = float(np.mean(same_source_repeats[:sample] >= 2))
    released_wrongly = float(np.mean(same_source_promotions))
    released = float(np.mean(independent_promotions))
    return {
        "formula": "quarantined activation <= cap until independent_sources>=2 or explicit confirmation",
        "activation_cap": poison_cap,
        "attack_top1_without_quarantine": no_quarantine_attack,
        "attack_top1_with_quarantine": quarantined_attack,
        "release_rate_with_evidence": released,
        "naive_repeat_based_false_release_rate": naive_same_source_false_release,
        "same_source_false_release_rate": released_wrongly,
    }


def test_lifecycle() -> dict:
    transitions = {
        "floating": {"provisional", "quarantined", "deleted"},
        "provisional": {"stable", "floating", "quarantined", "deleted"},
        "stable": {"structured", "invalidated", "quarantined", "archived"},
        "structured": {"consolidated", "invalidated", "quarantined", "archived"},
        "consolidated": {"invalidated", "archived"},
        "quarantined": {"provisional", "deleted"},
        "archived": {"stable", "deleted"},
        "invalidated": {"archived", "deleted"},
        "deleted": set(),
    }
    def execute_transition(source, target, evidence, conflict, confirmed):
        if target not in transitions[source]:
            return False
        if source == "provisional" and target == "stable":
            return evidence >= 2 or confirmed
        if source == "structured" and target == "consolidated":
            return evidence >= 3 and conflict <= .22
        if source == "quarantined" and target == "provisional":
            return evidence >= 2 or confirmed
        return True

    rng = np.random.default_rng(SEED + 107)
    states = list(transitions)
    attempts = 100_000
    accepted = rejected = illegal_accepted = unsafe_promotions = 0
    for _ in range(attempts):
        source, target = rng.choice(states, 2, replace=True)
        evidence = int(rng.integers(0, 5)); conflict = float(rng.random()); confirmed = bool(rng.random() < .05)
        executed = execute_transition(source, target, evidence, conflict, confirmed)
        if executed:
            accepted += 1
            if source == "provisional" and target == "stable" and evidence < 2 and not confirmed:
                unsafe_promotions += 1
            if target not in transitions[source]:
                illegal_accepted += 1
        else:
            rejected += 1
    required_path = ["floating", "provisional", "stable", "structured", "consolidated", "archived"]
    path_valid = all(required_path[i + 1] in transitions[required_path[i]] for i in range(len(required_path) - 1))
    return {
        "formula": "state(t+1)=guarded_transition(state,evidence,utility,conflict,time)",
        "attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "illegal_transitions_accepted": illegal_accepted,
        "unsafe_provisional_promotions": unsafe_promotions,
        "required_path_valid": path_valid,
    }


def test_star_homeostasis() -> dict:
    rng = np.random.default_rng(SEED + 108)
    trials, stars = 100_000, 16
    sizes = np.exp(rng.uniform(math.log(5), math.log(200_000), (trials, stars)))
    semantic = rng.normal(0, 1, (trials, stars))
    relevant = rng.integers(0, stars, trials)
    semantic[np.arange(trials), relevant] += 2.2
    raw = semantic + .28 * np.log1p(sizes)
    raw_accuracy = float(np.mean(np.argmax(raw, axis=1) == relevant))
    best = None
    for zeta in np.linspace(0, .5, 101):
        score = raw - zeta * np.log1p(sizes)
        accuracy = float(np.mean(np.argmax(score, axis=1) == relevant))
        largest_bias = float(np.mean(np.argmax(score, axis=1) == np.argmax(sizes, axis=1)))
        objective = accuracy - .10 * largest_bias
        if best is None or objective > best[0]:
            best = (objective, float(zeta), accuracy, largest_bias)
    assert best
    return {
        "formula": "star_activation = local_activation - zeta*log(1+child_count)",
        "zeta": best[1],
        "raw_accuracy": raw_accuracy,
        "normalized_accuracy": best[2],
        "largest_star_selection_rate": best[3],
        "split_rule": "split a star when topic entropy remains above a calibrated threshold",
    }


def test_procedural_memory() -> dict:
    rng = np.random.default_rng(SEED + 109)
    tasks, skills = 60_000, 8
    context = rng.random((tasks, skills))
    base_ability = rng.beta(3, 2, (tasks, skills))
    true_success = sigmoid(4 * (context - .5) + 2 * (base_ability - .5))
    history_n = rng.integers(1, 80, (tasks, skills))
    history_success = rng.binomial(history_n, true_success)
    a, b = 1 + history_success, 1 + history_n - history_success
    mean, std = a / (a + b), np.sqrt(a * b / (((a + b) ** 2) * (a + b + 1)))
    optimum = np.argmax(true_success, axis=1)
    usage_pick = np.argmax(history_n, axis=1)
    best = None
    for z in np.linspace(0, 2.5, 101):
        score = (mean - z * std) * (.35 + .65 * context)
        pick = np.argmax(score, axis=1)
        accuracy = float(np.mean(pick == optimum))
        expected_success = float(np.mean(true_success[np.arange(tasks), pick]))
        if best is None or expected_success > best[0]:
            best = (expected_success, float(z), accuracy)
    assert best
    return {
        "formula": "skill_score=(beta_mean-z*beta_std)*(0.35+0.65*context_match)",
        "z": best[1],
        "best_skill_identification": best[2],
        "expected_success": best[0],
        "usage_only_identification": float(np.mean(usage_pick == optimum)),
        "invariant": "usage count alone never reinforces a skill; outcomes and applicability do",
    }


def main() -> None:
    multi = test_multi_orbit()
    uncertainty = test_uncertainty()
    bitemporal = test_bitemporal()
    budget = test_activation_budget()
    relations = test_typed_relations()
    quarantine = test_quarantine()
    lifecycle = test_lifecycle()
    homeostasis = test_star_homeostasis()
    procedural = test_procedural_memory()
    regression = {
        "multi_orbit_improves_coverage": multi["multi_orbit_query_coverage"] > multi["single_parent_query_coverage"],
        "uncertainty_reduces_unsafe_accepts": uncertainty["high_stakes"]["unsafe_accept_rate"] < uncertainty["mean_only"]["unsafe_accept_rate"],
        "bitemporal_beats_recency": bitemporal["bitemporal_accuracy"] > bitemporal["ingestion_recency_accuracy"],
        "bitemporal_matches_independent_oracle": bitemporal["oracle_implementation_disagreements"] == 0,
        "activation_budget_is_constant": all(abs(v["budgeted_sum"] - budget["budget"]) < 1e-8 for v in budget["scaling"].values()),
        "typed_relations_beat_untyped": relations["typed_chain_recall_at_6"] > relations["untyped_chain_recall_at_6"],
        "quarantine_blocks_attack": quarantine["attack_top1_with_quarantine"] < .01,
        "same_source_repetition_does_not_promote": quarantine["same_source_false_release_rate"] == 0,
        "lifecycle_rejects_illegal_transitions": lifecycle["illegal_transitions_accepted"] == 0,
        "lifecycle_blocks_unsafe_promotions": lifecycle["unsafe_provisional_promotions"] == 0,
        "homeostasis_improves_routing": homeostasis["normalized_accuracy"] > homeostasis["raw_accuracy"],
        "procedural_memory_beats_usage": procedural["best_skill_identification"] > procedural["usage_only_identification"],
    }
    report = {
        "methodology": {
            "seed": SEED,
            "scope": "synthetic adversarial and property tests",
            "warning": "Prototype priors only; real conversation benchmarks and calibrated feature extractors remain required.",
        },
        "multi_orbit": multi,
        "uncertainty": uncertainty,
        "bitemporal": bitemporal,
        "activation_budget": budget,
        "typed_relations": relations,
        "quarantine": quarantine,
        "lifecycle": lifecycle,
        "star_homeostasis": homeostasis,
        "procedural_memory": procedural,
        "regression_gates": regression,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
