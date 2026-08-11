from __future__ import annotations

import unittest

from benchmarks.umd325_adapter import (
    galactic_census_budget,
    is_explicit_antimatter,
    split_antimatter_orbits,
)
from benchmarks.ai_memory_extended import _memora_session_text


class UMD325AdapterTest(unittest.TestCase):
    def test_budget_distinguishes_aggregate_and_direct_queries(self) -> None:
        self.assertEqual(galactic_census_budget("What is my total food spending?", 900), 512)
        self.assertEqual(galactic_census_budget("List all remaining tasks.", 900), 160)
        self.assertEqual(galactic_census_budget("Where is my key?", 900), 32)

    def test_structured_delete_is_antimatter(self) -> None:
        self.assertTrue(is_explicit_antimatter("session operation delete category todo list"))
        self.assertFalse(is_explicit_antimatter("session operation add category todo list"))

    def test_answer_orbit_filters_but_audit_conserves(self) -> None:
        capsules = [("1", "2"), ("3",)]
        channels = split_antimatter_orbits(
            "What remains on my todo list?", capsules, ["1", "2", "3"],
            ["operation add", "operation delete", "operation add"],
        )
        self.assertEqual(channels["audit"], capsules)
        self.assertEqual(channels["answer"], [("1",), ("3",)])

    def test_update_projection_excludes_superseded_value(self) -> None:
        text = _memora_session_text({
            "session_type": "preference",
            "operation": "update",
            "operation_details": {
                "item": "romantic comedy", "old_item": "western",
                "preference": "like", "old_preference": "dislike",
            },
            "conversation": [{
                "speaker": "user", "message": "I replaced western with romantic comedy.",
                "share_memory": True,
            }],
        })
        self.assertIn("romantic comedy", text)
        self.assertNotIn("western", text)
        self.assertNotIn("old preference", text)


if __name__ == "__main__":
    unittest.main()
