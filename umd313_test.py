from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from umd35_core import UMD35Config
from umd36_persistent import UMD36Database, UMD36TenantMemory, generate_master_key
from umd313_gravity import UMD313Config, UMD313RelativeGravityMemory


class UMD313RelativeGravityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.database = UMD36Database(Path(self.directory.name) / "umd313.sqlite3", generate_master_key())
        self.database.create_tenant("gravity-lab", "owner")
        base = UMD35Config(
            max_active_total=256,
            max_active_per_star=256,
            max_planets_per_star=64,
            duplicate_threshold=0.9999,
            relation_similarity_threshold=1.1,
        )
        self.durable = UMD36TenantMemory(
            self.database, "gravity-lab", "owner", config=base,
            validation_source="synthetic",
        )
        self.memory = UMD313RelativeGravityMemory(
            self.durable,
            config=UMD313Config(
                gravity_capture_threshold=0.50,
                gravity_relative_margin=0.0,
            ),
        )

    def tearDown(self) -> None:
        self.database.close()
        self.directory.cleanup()

    def test_strong_query_field_captures_bound_satellite_without_reparenting(self) -> None:
        target = self.memory.write(
            "Project Atlas launch city is Lisbon.", scope="atlas",
            session_key="city-session", source="user", explicit_importance=1.0,
        )
        self.memory.write(
            "Garden irrigation uses twelve liters of water.", scope="atlas",
            session_key="garden-session", source="user",
        )
        before = self.durable.engine.nodes[target.memory_id].planet_id
        hits = self.memory.retrieve_gravity("What is the Project Atlas launch city?", scope="atlas")
        expanded = {source for hit in hits for source in hit.source_ids}
        after = self.durable.engine.nodes[target.memory_id].planet_id
        self.assertIn(target.memory_id, expanded)
        self.assertEqual(before, after)
        self.assertTrue(all(hit.explanation["persistent_reparenting"] is False for hit in hits))

    def test_tidal_bridge_connects_complementary_cross_session_evidence(self) -> None:
        city = self.memory.write(
            "Project Atlas launch city is Lisbon.", scope="atlas",
            session_key="atlas-city", source="user", explicit_importance=1.0,
        )
        year = self.memory.write(
            "Project Atlas launch year is 2028.", scope="atlas",
            session_key="atlas-year", source="user", explicit_importance=1.0,
        )
        hits = self.memory.retrieve_gravity(
            "Which city hosts Project Atlas launch and what year is Project Atlas launch?",
            scope="atlas", top_k=4, budget_chars=5000,
        )
        self.assertTrue(any(
            {city.memory_id, year.memory_id} <= set(hit.source_ids) for hit in hits
        ))
        bridge = next(hit for hit in hits if {city.memory_id, year.memory_id} <= set(hit.source_ids))
        self.assertTrue(bridge.explanation["transient_bridge"])
        self.assertGreaterEqual(len(bridge.explanation["captured_goal_indices"]), 2)

    def test_unrelated_satellite_does_not_displace_stronger_memory(self) -> None:
        target = self.memory.write(
            "Project Atlas launch year is 2028.", scope="atlas",
            session_key="atlas", source="user", explicit_importance=1.0,
        )
        distractor = self.memory.write(
            "The ceramic studio closes on Mondays.", scope="atlas",
            session_key="ceramics", source="user", explicit_importance=1.0,
        )
        hits = self.memory.retrieve_gravity("When is the Project Atlas launch year?", scope="atlas", top_k=1)
        self.assertTrue(hits)
        self.assertIn(target.memory_id, hits[0].source_ids)
        self.assertNotIn(distractor.memory_id, hits[0].source_ids)

    def test_scope_isolation_survives_dynamic_capture(self) -> None:
        allowed = self.memory.write(
            "Atlas access token label is PUBLIC-DEMO.", scope="atlas",
            session_key="security", source="user", explicit_importance=1.0,
        )
        forbidden = self.memory.write(
            "Borealis access token is SECRET-991.", scope="borealis",
            session_key="security", source="user", explicit_importance=1.0,
        )
        hits = self.memory.retrieve_gravity("What is the access token?", scope="atlas")
        expanded = {source for hit in hits for source in hit.source_ids}
        self.assertIn(allowed.memory_id, expanded)
        self.assertNotIn(forbidden.memory_id, expanded)

    def test_explanation_exposes_elements_and_optimizer(self) -> None:
        self.memory.write(
            "Project Atlas launch city is Lisbon.", scope="atlas",
            session_key="atlas", source="user", explicit_importance=1.0,
        )
        hit = self.memory.retrieve_gravity("What is the Project Atlas launch city?", scope="atlas")[0]
        self.assertIn("net_potential", hit.elements)
        self.assertIn("Omega", hit.explanation["ai_language"])
        self.assertIn("Pcapture", hit.explanation["formula"])
        self.assertTrue(hit.explanation["immutable_provenance"])


if __name__ == "__main__":
    unittest.main()
