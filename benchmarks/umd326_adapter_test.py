from __future__ import annotations

import unittest

from benchmarks.umd326_adapter import (
    EmergentDomainGraph,
    collapse_census_singularity,
    compact_census_payload,
    compile_versioned_facts,
    is_census_query,
    singularity_budget,
)


class UMD326AdapterTest(unittest.TestCase):
    def test_natural_recommendation_is_a_census(self) -> None:
        self.assertTrue(is_census_query("Where should I travel next?"))
        self.assertEqual(singularity_budget("Where should I travel next?", 900), 900)
        self.assertEqual(singularity_budget("What is my total spending?", 900), 768)
        self.assertEqual(singularity_budget("Any music suggestions?", 900), 900)

    def test_singularity_conserves_and_collapses_sources(self) -> None:
        capsules = [(1, 2), (3,), (4,)]
        output = collapse_census_singularity("List all tasks.", capsules)
        self.assertEqual(output[0], (1, 2, 3, 4))
        self.assertEqual(output[1:], [(3,), (4,)])

    def test_fact_versions_and_payload_exclude_old_values(self) -> None:
        records = [{
            "operation": "update", "date": "2025-01-01",
            "session_type": "preference",
            "operation_details": {
                "subcategory": "genres", "item": "jazz", "old_item": "rock",
            },
        }]
        facts = compile_versioned_facts(records, ["7"])
        self.assertIn("superseded", {fact.state for fact in facts})
        payload = compact_census_payload("Suggest music", ["7"], facts)
        rendered = str(payload)
        self.assertIn("jazz", rendered)
        self.assertNotIn("rock", rendered)

    def test_nested_old_item_is_superseded(self) -> None:
        facts = compile_versioned_facts([{
            "operation": "update",
            "operation_details": {
                "item": {"step_count": 9000},
                "old_item": {"step_count": 7000},
            },
        }], ["9"])
        states = {fact.field: fact.state for fact in facts}
        self.assertEqual(states["item.step_count"], "current")
        self.assertEqual(states["old_item.step_count"], "superseded")

    def test_emergent_graph_learns_local_relation(self) -> None:
        graph = EmergentDomainGraph([
            frozenset({"travel", "climate", "monsoon"}),
            frozenset({"travel", "climate", "coastal"}),
            frozenset({"music", "album", "jazz"}),
        ])
        self.assertIn("climate", graph.expand("travel"))
        self.assertNotIn("album", graph.expand("travel"))


if __name__ == "__main__":
    unittest.main()
