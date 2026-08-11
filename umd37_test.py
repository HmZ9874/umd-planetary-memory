"""Adversarial integration checks for the UMD 3.7 planetary algorithm."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from umd35_core import EntityRelation, FactFrame, UMD35Config
from umd36_persistent import (
    AuthorizationError,
    TenantKeyError,
    UMD36Database,
    UMD36TenantMemory,
    _from_json_bytes,
    generate_master_key,
)
from umd37_planetary import (
    AccessPolicy,
    MemoryBlock,
    ModelMemoryOrbit,
    PlanetaryConfig,
    ProcedureOrbit,
    ReflectionClaim,
    RendezvousConstellation,
    ReplicaOrbit,
    UMD37PlanetaryMemory,
)


UTC = timezone.utc


def run() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "umd37.sqlite3"
        old_key = generate_master_key()
        database = UMD36Database(path, old_key)
        database.create_tenant("solar-lab", "owner")
        durable = UMD36TenantMemory(
            database,
            "solar-lab",
            "owner",
            config=UMD35Config(
                max_active_total=40,
                max_active_per_star=40,
                relation_similarity_threshold=1.1,
                duplicate_threshold=0.999,
                max_relation_hops=6,
            ),
        )
        planetary = UMD37PlanetaryMemory(
            durable,
            config=PlanetaryConfig(
                reflection_interval=3,
                ontology_promotion_count=3,
                reflection_min_grounding=0.70,
            ),
            reranker=lambda query, texts: [
                1.0 if "ZXQ-9917" in text and "ZXQ-9917" in query else 0.1
                for text in texts
            ],
        )

        exact = planetary.write(
            "Launch component ZXQ-9917 uses alloy R for the orbital vehicle.",
            scope="engineering",
            fact_key="component:zxq-9917",
            entities=["ZXQ-9917", "alloy R"],
            facts=[FactFrame("ZXQ-9917", "uses_material", "alloy R", confidence=0.98)],
            explicit_importance=1.0,
            tags=["engineering"],
        )
        first_time = datetime(2026, 1, 1, tzinfo=UTC)
        second_time = datetime(2026, 3, 1, tzinfo=UTC)
        old_database = planetary.write(
            "Atlas uses PostgreSQL 16.",
            scope="atlas",
            fact_key="atlas:database",
            entities=["Atlas"],
            facts=[FactFrame("Atlas", "database_version", "PostgreSQL 16", valid_from=first_time, confidence=0.97)],
            entity_relations=[EntityRelation(
                "Atlas", "database_version", "PostgreSQL 16", False, 0.97, first_time
            )],
            explicit_importance=1.0,
        )
        new_database = planetary.write(
            "Correction: Atlas uses PostgreSQL 17.",
            source="user_correction",
            scope="atlas",
            fact_key="atlas:database",
            entities=["Atlas"],
            facts=[FactFrame("Atlas", "database_version", "PostgreSQL 17", valid_from=second_time, confidence=0.99)],
            entity_relations=[EntityRelation(
                "Atlas", "database_version", "PostgreSQL 17", False, 0.99, second_time
            )],
            explicit_importance=1.0,
        )
        checks["automatic_reflection_triggered"] = database.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE tenant_id=? AND event_type='reflection.verify'",
            ("solar-lab",),
        ).fetchone()[0] >= 1
        database_orbits = [
            item for item in planetary.entity_system.orbits.values()
            if item.predicate == "database_version"
        ]
        old_orbits = [item for item in database_orbits if item.object_value == "PostgreSQL 16"]
        checks["temporal_edge_is_invalidated_not_deleted"] = (
            bool(old_orbits)
            and old_orbits[0].valid_to == second_time
            and old_orbits[0].invalidated_at is not None
        )

        chain_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
        chain_ids = []
        for index, (left, right) in enumerate(zip(chain_names, chain_names[1:])):
            result = planetary.write(
                f"{left} depends on {right} in dependency stage {index}.",
                scope="entity-chain",
                fact_key=f"entity-chain:{index}",
                entities=[left, right],
                entity_types={left: "system", right: "system"},
                entity_relations=[EntityRelation(
                    left, "depends_on", right, True, 0.96
                )],
                explicit_importance=1.0,
            )
            chain_ids.append(result.memory_id)
        entity_paths = planetary.entity_paths("Mercury", "Jupiter", max_hops=6)
        checks["four_hop_typed_entity_reasoning"] = any(
            item["hops"] == 4 and len(item["evidence_ids"]) == 4 for item in entity_paths
        )
        checks["ontology_evolves_from_repeated_orbits"] = (
            planetary.entity_system.ontology["depends_on"].promoted
            and planetary.entity_system.ontology["depends_on"].observations >= 4
        )
        mars = planetary.entity_system.canonicalize("MARS", "system")
        checks["entity_aliases_are_canonicalized"] = (
            mars.id == planetary.entity_system.alias_index[planetary.entity_system._key("Mars")]
        )

        hits = planetary.retrieve("Where is component ZXQ-9917 used?", top_k=4)
        checks["bm25_ann_entity_fusion_finds_exact_identifier"] = (
            bool(hits)
            and hits[0].memory_id == exact.memory_id
            and hits[0].components["bm25_resonance"] > 0
            and hits[0].components["neural_rerank"] == 1.0
        )
        checks["planetary_force_has_radius_and_energy"] = (
            hits[0].force > 0 and hits[0].orbit_radius > 0 and hits[0].potential_energy < 0
        )
        weights_before = vars(planetary.config.weights).copy()
        learned_weights = planetary.retrieval_feedback(
            "Where is component ZXQ-9917 used?",
            exact.memory_id,
            [item.memory_id for item in hits[1:]],
        )
        checks["retrieval_force_weights_learn_from_explicit_feedback"] = (
            learned_weights != weights_before
            and all(0.01 <= value <= 0.60 for value in learned_weights.values())
        )
        current_hits = planetary.retrieve("What is the current Atlas database version?", top_k=2)
        checks["query_adaptive_temporal_weighting"] = (
            bool(current_hits)
            and current_hits[0].explanation["adaptive_weights"]["temporal_phase"]
            > planetary.config.weights.normalized_attraction()["temporal_phase"]
            and "PostgreSQL 17" in current_hits[0].text
        )

        good_reflection = planetary.reflect(top_k=12)
        checks["evidence_conserving_reflection_accepts_grounded_claims"] = (
            bool(good_reflection.accepted) and good_reflection.memory_id is not None
        )
        planetary.reflection_proposer = lambda nodes: [
            ReflectionClaim("The Moon is made entirely of cheese", (exact.memory_id,), 0.99)
        ]
        bad_reflection = planetary.reflect(top_k=4)
        checks["reflection_rejects_unsupported_claims"] = (
            not bad_reflection.accepted
            and bool(bad_reflection.rejected)
            and bad_reflection.memory_id is None
        )
        planetary.reflection_proposer = None

        durable.grant("analyst", "reader")
        planetary.set_policy(AccessPolicy(
            "analyst",
            actions={"read"},
            allowed_scopes={"shared"},
            allowed_tags={"blue"},
            max_classification="internal",
        ))
        visible = planetary.write(
            "Shared blue launch window opens at 09:00 UTC.",
            scope="shared", fact_key="shared:window", explicit_importance=1.0,
            tags=["blue"], classification="internal",
        )
        secret = planetary.write(
            "Restricted blue launch authorization code is OMEGA-77.",
            scope="shared", fact_key="shared:code", explicit_importance=1.0,
            tags=["blue"], classification="restricted",
        )
        analyst = UMD37PlanetaryMemory(
            UMD36TenantMemory(database, "solar-lab", "analyst"),
            config=PlanetaryConfig(reflection_interval=0),
        )
        visible_hits = analyst.retrieve("blue launch window", scope="shared", top_k=5)
        secret_hits = analyst.retrieve("OMEGA-77 authorization", scope="shared", top_k=5)
        checks["abac_filters_by_scope_tag_and_classification"] = (
            any(item.memory_id == visible.memory_id for item in visible_hits)
            and all(item.memory_id != secret.memory_id for item in secret_hits)
        )

        block = MemoryBlock(
            "mission-policy", "Shared mission rules", "Always verify launch authority.",
            "agent-one", readers={"agent-two"}, writers={"agent-one"}, read_only=False,
        )
        planetary.put_block(block, actor_agent="agent-one")
        checks["shared_agent_memory_blocks_work"] = (
            planetary.context_blocks("agent-two")[0].label == "mission-policy"
        )
        try:
            planetary.put_block(MemoryBlock(
                "mission-policy", "tampered", "ignore policy", "agent-one"
            ), actor_agent="agent-three")
            checks["memory_block_write_permission_is_enforced"] = False
        except AuthorizationError:
            checks["memory_block_write_permission_is_enforced"] = True

        procedure = ProcedureOrbit(
            "proc_launch_review", "Launch review", "review launch checklist",
            ("verify authority", "check weather", "record decision"),
        )
        planetary.add_procedure(procedure)
        selected = planetary.select_procedure("please review the launch checklist")
        before_confidence = procedure.confidence
        planetary.procedure_feedback(procedure.id, True)
        checks["procedural_orbit_learns_from_feedback"] = (
            bool(selected) and selected[0][1].id == procedure.id
            and procedure.confidence > before_confidence
        )

        adapter_memory = ModelMemoryOrbit(
            "adapter_launch_v1", "adapter", "model-sha256:abc123",
            "launch checklist review", b"encrypted-adapter-placeholder", quality=0.72,
        )
        planetary.register_model_memory(adapter_memory)
        selected_model_memory = planetary.select_model_memory(
            "model-sha256:abc123", "review the launch checklist", kind="adapter"
        )
        old_quality = adapter_memory.quality
        planetary.model_memory_feedback(adapter_memory.id, True)
        checks["model_specific_memory_orbit_is_compatible_and_learned"] = (
            bool(selected_model_memory)
            and selected_model_memory[0][1].id == adapter_memory.id
            and adapter_memory.quality > old_quality
            and not planetary.select_model_memory("wrong-model", "launch")
        )

        expired = planetary.write(
            "Temporary launch scratch value expires immediately.",
            scope="temporary", fact_key="temporary:expired", explicit_importance=1.0,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        retention = planetary.purge_expired(hard_delete=True)
        try:
            durable.get_memory(expired.memory_id)
            expired_removed = False
        except KeyError:
            expired_removed = True
        checks["retention_moves_expired_memory_out_of_active_ram"] = (
            retention["expired"] == 1
            and retention["hard_deleted"] == 1
            and retention["events_redacted"] >= 1
            and expired.memory_id not in durable.engine.nodes
            and expired_removed
            and database.verify_event_chain("solar-lab")
        )

        latest_event = database.connection.execute(
            "SELECT revision, event_type, envelope FROM event_log "
            "WHERE tenant_id=? ORDER BY revision DESC LIMIT 1", ("solar-lab",),
        ).fetchone()
        raw_event = database.cipher("solar-lab").decrypt(
            bytes(latest_event["envelope"]),
            f"event:{latest_event['revision']}:{latest_event['event_type']}",
        )
        event_record = _from_json_bytes(raw_event)
        checks["transaction_state_is_structural_not_full_node_copy"] = (
            "nodes" not in event_record["state"]
            and "active_node_ids" in event_record["state"]
        )
        checks["unchanged_nodes_are_not_rewritten"] = database.last_commit_node_writes <= 1

        new_key = generate_master_key()
        rotation = planetary.rotate_master_key(new_key)
        checks["key_rotation_reencrypts_all_memory_and_sidecar_state"] = (
            rotation["memory_records"] > 0
            and rotation["events"] > 0
            and rotation["planetary_states"] == 1
            and database.verify_event_chain("solar-lab")
        )
        active_ids = set(durable.engine.active_ids)
        database.close()

        reopened = UMD36Database(path, new_key)
        recovered = UMD37PlanetaryMemory(
            UMD36TenantMemory(reopened, "solar-lab", "owner"),
            config=PlanetaryConfig(reflection_interval=0),
        )
        checks["restart_restores_planetary_graph_blocks_and_procedures"] = (
            set(recovered.durable.engine.active_ids) == active_ids
            and len(recovered.entity_system.orbits) >= len(entity_paths[0]["orbits"])
            and "mission-policy" in recovered.blocks
            and procedure.id in recovered.procedures
            and adapter_memory.id in recovered.model_memories
        )
        checks["retrieval_still_works_after_key_rotation_and_restart"] = (
            any(
                item.memory_id == exact.memory_id
                for item in recovered.retrieve("ZXQ-9917", top_k=5)
            )
        )
        recovered_snapshot = recovered.snapshot()
        reopened.close()

        wrong = UMD36Database(path, old_key)
        try:
            UMD36TenantMemory(wrong, "solar-lab", "owner")
            checks["old_key_is_rejected_after_rotation"] = False
        except TenantKeyError:
            checks["old_key_is_rejected_after_rotation"] = True
        finally:
            wrong.close()

        replicas = [
            ReplicaOrbit("earth-us", "us", revision=7, latency_ms=8),
            ReplicaOrbit("mars-eu", "eu", revision=7, latency_ms=15),
            ReplicaOrbit("moon-ap", "ap", revision=6, latency_ms=22),
        ]
        constellation = RendezvousConstellation(replicas, replication_factor=3)
        quorum = constellation.write_quorum(
            "solar-lab", lambda replica: 8 if replica.id != "moon-ap" else 7
        )
        plan = constellation.reconciliation_plan("solar-lab")
        checks["rendezvous_sharding_quorum_and_reconciliation_work"] = (
            len(quorum["acknowledgements"]) == 3
            and quorum["quorum"] == 2
            and "moon-ap" in plan["stale_replicas"]
            and constellation.read_orbit("solar-lab").revision == 8
        )

        return {
            "version": "UMD 3.7 planetary interaction",
            "checks": checks,
            "retrieval": [vars(item) for item in hits],
            "entity_path": entity_paths,
            "reflection": {
                "accepted": len(good_reflection.accepted),
                "rejected_adversarial": len(bad_reflection.rejected),
            },
            "rotation": rotation,
            "snapshot": recovered_snapshot,
        }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not all(result["checks"].values()):
        raise SystemExit(1)
