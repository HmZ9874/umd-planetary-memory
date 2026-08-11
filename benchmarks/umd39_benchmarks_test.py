"""Fast checks for the UMD 3.9 public benchmark adapter."""

from __future__ import annotations

import unittest

from benchmarks.public_benchmarks import normalize_locomo_evidence_ids, peak_working_set_mib
from benchmarks.umd39_benchmarks import (
    CapsuleRecallAccumulator,
    _anchored_capsule_order,
    _benchmark_capsules,
    _cognitive_order,
    _directive_metadata,
    _flatten_source_ids,
    _force,
    _orbit_expand,
)
from benchmarks.umd314_adapter import (
    fuse_transient_bridge,
    lagrange_capsules,
    relative_gravity_order,
    transient_directive_bridge,
)
from benchmarks.umd315_adapter import (
    compress_event_horizon,
    event_field_entropy_budget,
    event_satellite_budget,
    rank_event_satellites,
)


class UMD39BenchmarkAdapterTests(unittest.TestCase):
    def test_packed_locomo_evidence_ids_are_normalized(self) -> None:
        value = ["D9:1 D4:4 D4:6", "D1:2", {"latest": "D8:7"}]
        self.assertEqual(
            normalize_locomo_evidence_ids(value),
            {"D9:1", "D4:4", "D4:6", "D1:2", "D8:7"},
        )

    def test_umd315_entropy_budget_is_bounded(self) -> None:
        flat = event_field_entropy_budget(
            list(range(8)), [(8,)], list(range(9)),
            [0.5] * 9, [0.5] * 9, [0.5] * 9,
        )
        peaked = event_field_entropy_budget(
            list(range(8)), [(8,)], list(range(9)),
            [1.0] + [0.0] * 8, [1.0] + [0.0] * 8,
            [1.0] + [0.0] * 8,
        )
        self.assertTrue(4 <= flat <= 8)
        self.assertEqual(peaked, 4)

    def test_umd315_secondary_orbit_rerank_excludes_stable_sources(self) -> None:
        result = rank_event_satellites(
            [0, 1, 2, 3], [(0,), (1,)], [0, 0, 1, 2],
            [1.0, 0.9, 0.2, 0.8], [1.0, 0.9, 0.2, 0.8],
            [1.0, 0.9, 0.2, 0.8],
        )
        self.assertEqual(result, [3, 2])

    def test_umd315_budget_is_query_derived_and_bounded(self) -> None:
        self.assertEqual(event_satellite_budget("Would Maya likely enjoy hiking?"), 20)
        self.assertEqual(event_satellite_budget("When did Maya arrive?"), 6)
        self.assertEqual(event_satellite_budget("Why did Maya move?"), 12)
        self.assertEqual(event_satellite_budget("What color is the car?"), 8)
        self.assertEqual(event_satellite_budget("Would Maya travel?", ceiling=10), 10)

    def test_umd315_event_horizon_preserves_stable_prefixes(self) -> None:
        stable = [(0, 1), (2,), (3, 4)]
        result = compress_event_horizon(stable, list(range(20)), max_extra_sources=6)
        for cutoff in range(1, len(stable) + 1):
            old = {source for capsule in stable[:cutoff] for source in capsule}
            new = {source for capsule in result[:cutoff] for source in capsule}
            self.assertTrue(old <= new)

    def test_umd315_event_horizon_is_gold_blind_and_bounded(self) -> None:
        stable = [(0,), (1,)]
        result = compress_event_horizon(stable, list(range(100)), max_extra_sources=5)
        added = {source for capsule in result for source in capsule} - {0, 1}
        self.assertEqual(added, {2, 3, 4, 5, 6})
    def test_nested_evidence_ids_are_normalized(self) -> None:
        value = {"old": [3, 7], "new": {"latest": [11]}}
        self.assertEqual(_flatten_source_ids(value), {"3", "7", "11"})

    def test_orbit_expansion_stays_inside_session(self) -> None:
        self.assertEqual(_orbit_expand([1, 3, 0, 2], [0, 0, 1, 1]), [1, 0, 3, 2])

    def test_frozen_force_weights_sum_to_one(self) -> None:
        ones = [1.0]
        self.assertAlmostEqual(_force(ones, ones, ones, ones, ones, ones, ones)[0], 1.0)

    def test_peak_memory_probe_works_on_windows(self) -> None:
        value = peak_working_set_mib()
        self.assertIsNotNone(value)
        self.assertGreater(value or 0.0, 0.0)

    def test_cognitive_router_preserves_unmatched_query(self) -> None:
        base = [2, 0, 1]
        result = _cognitive_order(
            "Where is the key?", ["a", "b", "c"], [0, 0, 1], ["d0", "d1"],
            base, [0.1, 0.2, 0.9], [0.1, 0.2, 0.9], [0.1, 0.2, 0.9],
            [None, None, None], top_k=2,
        )
        self.assertEqual(result, base)

    def test_latest_trusted_preference_is_reserved(self) -> None:
        texts = ["answer", "I prefer concise replies", "I prefer detailed replies"]
        metadata = [_directive_metadata(text, trusted=True) for text in texts]
        result = _cognitive_order(
            "What style should you recommend?", texts, [0, 1, 2], ["1", "2", "3"],
            [0, 1, 2], [1.0, 0.2, 0.1], [1.0, 0.2, 0.1], [1.0, 0.2, 0.1],
            metadata, top_k=3,
        )
        self.assertEqual(result[:2], [0, 2])

    def test_capsule_expansion_never_loses_atomic_top_ten(self) -> None:
        groups = [0] * 20
        capsules, by_source = _benchmark_capsules(groups)
        base = list(range(20))
        selected = _anchored_capsule_order(
            "Summarize all milestones", base, [1.0 - i / 40 for i in base],
            capsules, by_source, top_k=10,
        )
        expanded = {source for capsule in selected for source in capsule}
        self.assertTrue(set(base[:10]) <= expanded)
        self.assertTrue(all(len(capsule) <= 4 for capsule in selected))

    def test_capsule_metric_can_count_more_than_ten_provenance_sources(self) -> None:
        metric = CapsuleRecallAccumulator((10,))
        capsules = [tuple(range(index, index + 2)) for index in range(0, 20, 2)]
        # Real IDs are strings in public datasets.
        string_capsules = [tuple(str(value) for value in capsule) for capsule in capsules]
        metric.add(string_capsules, {str(index) for index in range(16)})
        result = metric.result()
        self.assertEqual(result["full_evidence_recall"]["10"], 1.0)

    def test_umd314_reserves_applicable_directive_without_gold(self) -> None:
        texts = [
            "The login feature uses Flask-Login.",
            "Always format code snippets with syntax highlighting.",
            "Unrelated deployment note.",
        ]
        metadata = [_directive_metadata(text, trusted=True) for text in texts]
        order, _ = relative_gravity_order(
            "Could you show me how to implement login?", texts, [0, 1, 2],
            ["1", "2", "3"], [0, 2, 1], [1.0, 0.05, 0.2],
            [1.0, 0.05, 0.2], [1.0, 0.05, 0.2], metadata, top_k=3,
        )
        bridge = transient_directive_bridge(
            "Could you show me how to implement login?", texts, order, metadata,
        )
        self.assertEqual(order, [0, 2, 1])
        self.assertEqual(bridge, (0, 1))

    def test_umd314_bridge_preserves_every_atomic_anchor(self) -> None:
        groups = [0] * 12
        capsules, by_source = _benchmark_capsules(groups)
        selected = _anchored_capsule_order(
            "Show code", list(range(12)), [1.0] * 12,
            capsules, by_source, top_k=10,
        )
        bridged = fuse_transient_bridge(selected, (0, 11))
        for cutoff in range(1, 11):
            old = {source for capsule in selected[:cutoff] for source in capsule}
            new = {source for capsule in bridged[:cutoff] for source in capsule}
            self.assertTrue(old <= new)
        self.assertIn(11, set(bridged[0]))

    def test_umd314_entity_capsules_can_cross_sessions(self) -> None:
        capsules, _ = lagrange_capsules(
            ["Dr Vega chose Lisbon.", "Unrelated note.", "Dr Vega approved launch."],
            [0, 1, 2],
        )
        self.assertTrue(any({0, 2} <= set(capsule) for capsule in capsules))

    def test_umd314_locks_multi_evidence_orbit(self) -> None:
        texts = ["primary evidence", "Always format code as markdown.", "second evidence"]
        metadata = [_directive_metadata(text, trusted=True) for text in texts]
        base = [0, 2, 1]
        order, _ = relative_gravity_order(
            "Considering all sessions, how many different changes were made?",
            texts, [0, 1, 2], ["1", "2", "3"], base,
            [1.0, 0.05, 0.8], [1.0, 0.05, 0.8], [1.0, 0.05, 0.8],
            metadata, top_k=3,
        )
        self.assertEqual(order, base)

    def test_umd314_implicit_preference_does_not_perturb_orbit(self) -> None:
        texts = ["primary evidence", "I prefer concise responses", "second evidence"]
        metadata = [_directive_metadata(text, trusted=True) for text in texts]
        base = [0, 2, 1]
        order, _ = relative_gravity_order(
            "How should I structure the report?", texts, [0, 1, 2],
            ["1", "2", "3"], base, [1.0, 0.05, 0.8],
            [1.0, 0.05, 0.8], [1.0, 0.05, 0.8], metadata, top_k=3,
        )
        self.assertEqual(order, base)


if __name__ == "__main__":
    unittest.main()
