from __future__ import annotations

import unittest

from benchmarks.ai_memory_extended import escape_velocity_budget
from benchmarks.public_benchmarks import BM25
from benchmarks.umd316_adapter import (
    apply_state_field,
    conserved_constellation_orbit,
    constellation_budget,
    first_capsule_lagrange_order,
    query_physics,
    rank_constellation_coverage,
    state_mass_adjustment,
    split_state_orbits,
)
from benchmarks.umd315_adapter import compress_event_horizon


class UMD316PhysicsTests(unittest.TestCase):
    def test_escape_velocity_budget_is_adaptive_and_bounded(self) -> None:
        self.assertEqual(escape_velocity_budget("What is Jo's favorite color?", 96), 96)
        self.assertEqual(escape_velocity_budget("Why did Jo move to that country?", 96), 288)
        self.assertEqual(escape_velocity_budget("Why?", 10), 30)

    def test_bm25_prefix_matches_fresh_visible_index(self) -> None:
        documents = ["alpha beta", "beta gamma gamma", "future alpha alpha"]
        full = BM25(documents)
        fresh = BM25(documents[:2])
        self.assertEqual(len(full.scores_prefix("alpha gamma", 2)), 2)
        for left, right in zip(full.scores_prefix("alpha gamma", 2), fresh.scores("alpha gamma")):
            self.assertAlmostEqual(left, right, places=12)

    def test_query_physics_detects_broad_current_collection(self) -> None:
        mode = query_physics("What tasks remain on my current todo list?")
        self.assertTrue(mode.broad_set)
        self.assertTrue(mode.current_state)
        self.assertFalse(mode.historical)

    def test_historical_query_keeps_deleted_provenance_neutral(self) -> None:
        texts = ["I deleted the old trip plan.", "The new trip is active."]
        self.assertEqual(
            state_mass_adjustment("How did my trip plan change?", texts),
            [0.0, 0.0],
        )

    def test_current_query_uses_signed_state_mass(self) -> None:
        texts = ["I deleted the old task.", "The new task remains active."]
        adjustment = state_mass_adjustment("What tasks remain now?", texts)
        adjusted = apply_state_field([0.5, 0.5], adjustment)
        self.assertLess(adjusted[0], adjusted[1])

    def test_state_orbits_keep_deleted_fact_for_audit_only(self) -> None:
        result = split_state_orbits(
            "What tasks remain now?", [("old", "new")], ["old", "new"],
            ["I deleted the old task.", "The new task remains active."],
        )
        self.assertEqual(result["audit"], [("old", "new")])
        self.assertEqual(result["answer"], [("new",)])

    def test_first_capsule_lagrange_preserves_top_ten_set(self) -> None:
        capsules = [(0,), (1,), (2,)]
        result = first_capsule_lagrange_order(
            "Where did Maya camp?", capsules,
            ["unrelated note", "Maya camped near Tahoe", "other"],
            [0.7, 0.6, 0.5], [0.1, 1.0, 0.1], [0.1, 1.0, 0.1],
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
        )
        self.assertIn(1, result[0])
        self.assertEqual(result[1:], capsules[1:])
        self.assertEqual(
            {source for capsule in result for source in capsule},
            {source for capsule in capsules for source in capsule},
        )

    def test_constellation_budget_is_query_derived_and_bounded(self) -> None:
        self.assertEqual(constellation_budget("List all remaining tasks."), 40)
        self.assertEqual(constellation_budget("Where is the key?"), 16)
        self.assertEqual(constellation_budget("List every item.", ceiling=20), 20)

    def test_constellation_conserves_frozen_secondary_evidence(self) -> None:
        old = [0, 1, 2, 3]
        result = conserved_constellation_orbit(old, [4, 2, 1, 5], conserved=4)
        self.assertEqual(set(result[:4]), set(old))
        self.assertEqual(result[4:], [4, 5])

    def test_constellation_conserves_every_striped_prefix(self) -> None:
        stable = [(index,) for index in range(10)]
        old_orbit = list(range(10, 26))
        merged = conserved_constellation_orbit(
            old_orbit, list(reversed(range(10, 40))), conserved=16,
        )
        old = compress_event_horizon(stable, old_orbit, max_extra_sources=16)
        new = compress_event_horizon(stable, merged, max_extra_sources=40)
        for k in (1, 5, 10):
            old_sources = {source for capsule in old[:k] for source in capsule}
            new_sources = {source for capsule in new[:k] for source in capsule}
            self.assertTrue(old_sources <= new_sources)

    def test_constellation_coverage_prefers_new_fact_over_duplicate(self) -> None:
        # Source 1 is a near duplicate of the stable source. Source 2 adds a
        # second relevant project fact and should have higher marginal energy.
        result = rank_constellation_coverage(
            "List all current project tasks",
            [1, 2], [(0,)],
            [
                "Current project task: write report",
                "Current project task: write report",
                "Current project task: schedule field study",
            ],
            [0, 1, 2], ["1", "2", "3"],
            [1.0, 0.8, 0.8], [1.0, 0.8, 0.8], [1.0, 0.8, 0.8],
        )
        self.assertEqual(result[0], 2)


if __name__ == "__main__":
    unittest.main()
