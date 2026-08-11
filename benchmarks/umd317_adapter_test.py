from __future__ import annotations

import unittest

from benchmarks.public_benchmarks import BM25
from benchmarks.umd317_adapter import (
    compile_fact_planets,
    conserved_fact_constellation,
    extract_fact_atoms,
    first_capsule_focus_tether,
    rank_fact_planet_coverage,
)


class UMD317PhysicsTests(unittest.TestCase):
    def test_prefix_bm25_matches_fresh_index_at_every_boundary(self) -> None:
        documents = ["alpha beta", "beta gamma gamma", "future alpha alpha", "delta"]
        full = BM25(documents)
        for size in range(1, len(documents) + 1):
            fresh = BM25(documents[:size])
            for left, right in zip(full.scores_prefix("alpha gamma delta", size), fresh.scores("alpha gamma delta")):
                self.assertAlmostEqual(left, right, places=12)

    def test_fact_compiler_splits_independent_and_signed_atoms(self) -> None:
        atoms = extract_fact_atoms(
            "I planned a Tahoe trip; I deleted the old hotel. The train remains active."
        )
        self.assertGreaterEqual(len(atoms), 3)
        self.assertIn(-1, {atom.state for atom in atoms})
        self.assertIn(1, {atom.state for atom in atoms})

    def test_focus_tether_is_additive_and_bounded(self) -> None:
        facts = compile_fact_planets([
            "We discussed cooking.",
            "Maya camped beside Lake Tahoe.",
            "Maya bought a notebook.",
        ])
        capsules = [(0,), (1,), (2,)]
        output = first_capsule_focus_tether(
            "Where did Maya camp?", capsules, facts,
            [0.5, 0.8, 0.4], [0.2, 0.9, 0.3], [0.1, 0.9, 0.2],
        )
        self.assertEqual(output[1:], capsules[1:])
        self.assertEqual(len(set(output[0]) - set(capsules[0])), 1)
        self.assertIn(1, output[0])

    def test_fact_coverage_prefers_distinct_relevant_atom(self) -> None:
        facts = compile_fact_planets([
            "Current project task: write report.",
            "Current project task: write report.",
            "Current project task: schedule field study.",
        ])
        output = rank_fact_planet_coverage(
            "List all current project tasks", [1, 2], [(0,)], facts,
            [0, 1, 2], [1.0, 0.8, 0.8], [1.0, 0.8, 0.8], [1.0, 0.8, 0.8],
        )
        self.assertEqual(output[0], 2)

    def test_fact_constellation_preserves_complete_legacy_budget(self) -> None:
        old = list(range(16))
        legacy = list(range(16, 60))
        facts = [70, 71, 72]
        output = conserved_fact_constellation(old, legacy, facts)
        self.assertEqual(output[:40], list(range(40)))
        self.assertEqual(output[40:43], facts)


if __name__ == "__main__":
    unittest.main()
