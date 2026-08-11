"""Small repeatable RAM/index probe for UMD 3.12 capsules."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from umd35_core import UMD35Config
from umd36_persistent import UMD36Database, UMD36TenantMemory, generate_master_key
from umd312_capsules import UMD312ConstellationMemory


def run(count: int = 200) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        database = UMD36Database(Path(directory) / "scale.sqlite3", generate_master_key())
        database.create_tenant("scale", "owner")
        memory = UMD312ConstellationMemory(UMD36TenantMemory(
            database, "scale", "owner",
            config=UMD35Config(
                max_active_total=count + 16,
                max_active_per_star=count + 16,
                max_planets_per_star=16,
                planet_attach_threshold=1.1,
                duplicate_threshold=0.99999,
                relation_similarity_threshold=1.1,
            ),
            validation_source="synthetic",
        ))
        for index in range(count):
            memory.write(
                f"Atlas milestone {index}: component C-{index} passed deployment validation.",
                scope="atlas", session_key="scale-chain", fact_key=f"atlas:milestone:{index}",
                source="user" if index % 2 == 0 else "assistant",
                explicit_importance=1.0,
            )
        snapshot = memory.snapshot()
        started = time.perf_counter()
        hits = memory.retrieve_capsules(
            "Summarize all Atlas deployment milestones", scope="atlas", top_k=10,
            budget_chars=20_000,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        expanded = {source for hit in hits for source in hit.source_ids}
        result = {
            "active_memories": len(memory.durable.engine.active_ids),
            "capsules": snapshot["capsules"],
            "source_links": snapshot["capsule_source_links"],
            "capsule_index_container_bytes": snapshot["capsule_index_container_bytes"],
            "capsule_index_container_bytes_per_active_memory": snapshot[
                "capsule_index_container_bytes_per_active_memory"
            ],
            "retrieval_latency_ms": latency_ms,
            "returned_capsules": len(hits),
            "expanded_unique_sources": len(expanded),
            "plaintext_cache": snapshot["capsule_plaintext_cache"],
        }
        database.close()
        return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
