"""Adversarial and integration tests for UMD 3.8 cosmic memory."""

from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umd35_core import EntityRelation, UMD35Config
from umd36_persistent import UMD36Database, UMD36TenantMemory, generate_master_key
from umd37_planetary import MemoryBlock, ModelMemoryOrbit, ProcedureOrbit
from umd38_cosmic import (
    CapabilityToken,
    CosmicCheckpointManager,
    CosmicTelemetry,
    GravityPacket,
    IdentityGravitySystem,
    MultimodalMeteor,
    NetworkConstellation,
    UMD38Config,
    UMD38CosmicMemory,
    UMDGateway,
)
from umd38_eval import AnswerOutput, EndToEndQAEvaluator, QACase


class ControlledEncoder:
    def encode(self, text: str):
        values = {
            "Nova Alpha": [1.0, 0.0, 0.0, 0.0],
            "Nova Beta": [0.0, 1.0, 0.0, 0.0],
            "Nova": [0.7, 0.7, 0.0, 0.0],
        }
        return np.asarray(values.get(text, [0.0, 0.0, 1.0, 0.0]), dtype=np.float32)


class FakeMultimodalEncoder:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def encode_payload(self, payload: bytes, metadata: dict[str, object]):
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[0] = 1.0
        return vector, "red satellite above a blue ocean"


class FakeRuntime:
    def __init__(self):
        self.applied = []

    def fingerprint(self):
        return "model-sha256:abc"

    def apply_prompt(self, payload):
        self.applied.append(("prompt", payload))
        return "prompt-loaded"

    def apply_kv_cache(self, payload):
        self.applied.append(("kv_cache", payload))
        return "cache-loaded"

    def apply_adapter(self, payload):
        self.applied.append(("adapter", payload))
        return "adapter-loaded"


class FakeReplicaTransport:
    def __init__(self):
        self.revisions = {"http://r1": 4, "http://r2": 3, "http://r3": 4}
        self.packets = {}

    def apply(self, endpoint, packet):
        if packet.packet_id in self.packets.get(endpoint, set()):
            return {"revision": self.revisions[endpoint], "duplicate": True}
        self.packets.setdefault(endpoint, set()).add(packet.packet_id)
        self.revisions[endpoint] += 1
        return {"revision": self.revisions[endpoint], "duplicate": False}

    def status(self, endpoint, tenant_id):
        return {"tenant": tenant_id, "revision": self.revisions[endpoint]}


def _event_count(database, tenant="cosmic-lab"):
    return database.connection.execute(
        "SELECT COUNT(*) FROM event_log WHERE tenant_id=?", (tenant,)
    ).fetchone()[0]


def run() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "umd38.sqlite3"
        key = generate_master_key()
        database = UMD36Database(path, key)
        database.create_tenant("cosmic-lab", "owner")
        durable = UMD36TenantMemory(
            database, "cosmic-lab", "owner",
            config=UMD35Config(
                max_active_total=80, max_active_per_star=80,
                duplicate_threshold=0.999, relation_similarity_threshold=1.1,
            ),
            validation_source="synthetic",
        )
        telemetry = CosmicTelemetry(64)
        cosmic = UMD38CosmicMemory(
            durable, config=UMD38Config(reflection_interval=0), telemetry=telemetry
        )

        before = _event_count(database)
        alpha = cosmic.write(
            "Project Helios uses engine ZXQ-9917 for the lunar transfer vehicle.",
            scope="engineering", entities=["Project Helios", "ZXQ-9917"],
            explicit_importance=1.0,
        )
        after = _event_count(database)
        event_type = database.connection.execute(
            "SELECT event_type FROM event_log WHERE tenant_id=? ORDER BY revision DESC LIMIT 1",
            ("cosmic-lab",),
        ).fetchone()[0]
        checks["gravity_capsule_is_one_database_transaction"] = after - before == 1 and event_type == "cosmic.write"
        checks["incremental_index_contains_new_planet"] = alpha.memory_id in cosmic.index.tokens
        checks["write_does_not_trigger_full_index_rebuild"] = cosmic.index.full_rebuilds == 1

        cosmic.write(
            "Project Selene depends on Project Helios and launches in 2027.",
            scope="engineering", entities=["Project Selene", "Project Helios"],
            entity_relations=[EntityRelation(
                "Project Selene", "depends_on", "Project Helios", True, 0.96
            )],
            explicit_importance=1.0,
        )
        checks["multiple_revisions_still_use_incremental_index"] = (
            cosmic.index.full_rebuilds == 1 and cosmic.index.incremental_upserts >= 2
        )
        hits = cosmic.retrieve("Which vehicle uses ZXQ-9917?", top_k=3)
        checks["tidal_reranker_participates_in_force"] = (
            hits[0].memory_id == alpha.memory_id
            and hits[0].components["neural_rerank"] > 0.0
        )
        checks["retrieval_exposes_force_explanation"] = (
            hits[0].orbit_radius > 0 and hits[0].potential_energy < 0
            and "adaptive_weights" in hits[0].explanation
        )

        positive_before = cosmic.neural_tide.score("ZXQ-9917 vehicle", [hits[0].text])[0]
        for _ in range(25):
            cosmic.neural_tide.learn_pairwise(
                "ZXQ-9917 vehicle", hits[0].text,
                ["Recipe calls for two eggs and warm milk."], learning_rate=0.05,
            )
        positive_after = cosmic.neural_tide.score("ZXQ-9917 vehicle", [hits[0].text])[0]
        checks["neural_tide_learns_online"] = cosmic.neural_tide.updates == 25 and positive_after > positive_before
        cosmic._commit_planetary("test.tide.persist", alpha.memory_id, {"test": True})

        identity = IdentityGravitySystem(
            ControlledEncoder(), identity_threshold=0.55, identity_margin=0.30,
        )
        first = identity.canonicalize("Nova Alpha", "project")
        second = identity.canonicalize("Nova Beta", "project")
        ambiguous = identity.canonicalize("Nova", "project")
        checks["entity_margin_prevents_false_merge"] = (
            len({first.id, second.id, ambiguous.id}) == 3 and bool(identity.ambiguities)
        )
        communities = cosmic.entity_communities()
        checks["entity_graph_discovers_evidence_communities"] = any(
            len(item["entity_ids"]) >= 2 and item["evidence_ids"] for item in communities
        )

        cosmic.put_block(MemoryBlock(
            "mission", "current mission", "Protect the Helios launch window.",
            "agent-1", readers={"agent-1"}, writers={"agent-1"},
        ), actor_agent="agent-1")
        cosmic.add_procedure(ProcedureOrbit(
            "procedure-1", "engine lookup", "find engine vehicle",
            ("retrieve component", "verify evidence"),
        ))
        context = cosmic.assemble_context(
            "Which vehicle uses ZXQ-9917?", agent_id="agent-1", budget_chars=1200
        )
        kinds = {item["kind"] for item in context["items"]}
        checks["context_lagrange_packs_multiple_memory_species"] = (
            {"declarative", "block", "procedure"}.issubset(kinds)
            and context["used_chars"] <= context["budget_chars"]
        )

        events_before_failure = _event_count(database)
        nodes_before_failure = set(cosmic.durable.engine.nodes)
        try:
            cosmic.write(
                "Impossible interval memory", explicit_importance=1.0,
                valid_from=__import__("datetime").datetime(2028, 1, 2, tzinfo=__import__("datetime").timezone.utc),
                valid_to=__import__("datetime").datetime(2028, 1, 1, tzinfo=__import__("datetime").timezone.utc),
            )
        except ValueError:
            pass
        checks["failed_capsule_leaves_no_event_or_memory"] = (
            _event_count(database) == events_before_failure
            and set(cosmic.durable.engine.nodes) == nodes_before_failure
        )

        cosmic.register_multimodal_encoder(
            "image", FakeMultimodalEncoder(cosmic.durable.engine.config.embedding_dimensions)
        )
        multimodal_events_before = _event_count(database)
        image_result = cosmic.ingest_multimodal(
            MultimodalMeteor("image", b"fake-image-bytes", {"format": "test"}),
            scope="media", explicit_importance=1.0,
        )
        image_node = cosmic.durable.get_memory(image_result.memory_id)
        checks["multimodal_adapter_stores_digest_not_raw_payload"] = (
            image_node.extraction_metadata["multimodal"]["sha256"]
            and b"fake-image-bytes" not in database.connection.execute(
                "SELECT envelope FROM memory_records WHERE tenant_id=? AND memory_id=?",
                ("cosmic-lab", image_result.memory_id),
            ).fetchone()[0]
        )
        checks["multimodal_vector_uses_same_gravity_capsule"] = (
            _event_count(database) - multimodal_events_before == 1
        )

        artifact = ModelMemoryOrbit(
            "adapter-1", "adapter", "model-sha256:abc", "lunar planning", b"adapter-weights"
        )
        cosmic.register_model_memory(artifact)
        runtime = FakeRuntime()
        activation = cosmic.activate_model_memory("adapter-1", runtime)
        checks["model_artifact_is_applied_to_runtime"] = (
            activation == "adapter-loaded" and runtime.applied == [("adapter", b"adapter-weights")]
        )
        incompatible = FakeRuntime()
        incompatible.fingerprint = lambda: "wrong-model"
        try:
            cosmic.activate_model_memory("adapter-1", incompatible)
            rejected = False
        except ValueError:
            rejected = True
        checks["incompatible_model_artifact_is_rejected"] = rejected

        transport = FakeReplicaTransport()
        constellation = NetworkConstellation(
            {"r1": "http://r1", "r2": "http://r2", "r3": "http://r3"},
            transport, b"r" * 32, replication_factor=3,
        )
        packet = GravityPacket("packet-1", "cosmic-lab", "memory.write", 4, {"id": alpha.memory_id})
        quorum = constellation.replicate(packet)
        duplicate = constellation.replicate(packet)
        checks["network_constellation_reaches_signed_quorum"] = (
            len(quorum["acknowledgements"]) == 3 and packet.verify(b"r" * 32)
        )
        checks["replica_packets_are_idempotent"] = all(
            item["duplicate"] for item in duplicate["acknowledgements"].values()
        )
        transport.revisions["http://r2"] = 1
        repair = constellation.read_repair_plan("cosmic-lab")
        checks["constellation_plans_read_repair"] = "r2" in repair["repair"]

        authority = CapabilityToken(b"t" * 32)
        token = authority.issue("cosmic-lab", "owner")
        gateway = UMDGateway(lambda tenant, principal: cosmic, authority)
        headers = {"Authorization": "Bearer " + token}
        status, body, _ = gateway.handle(
            "POST", "/v1/retrieve", headers,
            json.dumps({"query": "ZXQ-9917", "top_k": 2}).encode(),
        )
        checks["authenticated_rest_gateway_retrieves"] = status == 200 and body[0]["memory_id"] == alpha.memory_id
        mcp = gateway.mcp({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "umd_retrieve", "arguments": {"query": "ZXQ-9917", "top_k": 1}},
        }, headers)
        checks["mcp_gateway_calls_same_memory_core"] = (
            mcp["id"] == 7 and mcp["result"]["content"][0]["memory_id"] == alpha.memory_id
        )
        evaluator = EndToEndQAEvaluator(
            lambda question, evidence: (
                AnswerOutput("ZXQ-9917", (evidence[0]["memory_id"],))
                if "engine" in question else AnswerOutput("unknown")
            ),
            top_k=3,
        )
        qa = evaluator.evaluate([
            QACase("qa-1", "What engine identifier is used?", "ZXQ-9917", frozenset({alpha.memory_id})),
            QACase("qa-2", "What is the pilot's birthday?", None, answerable=False),
        ], cosmic)
        checks["end_to_end_qa_uses_answers_citations_and_abstention"] = (
            qa["exact_match"] == 0.5 and qa["grounded_citation_rate"] == 1.0
            and qa["abstention_accuracy"] == 1.0
        )

        server = gateway.make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=2) as response:
                health = json.loads(response.read())
            checks["real_http_boundary_serves_health"] = health["version"] == "UMD 3.8"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        metrics = telemetry.prometheus()
        checks["prometheus_telemetry_is_bounded_and_exportable"] = (
            "umd_write_total" in metrics and "umd_retrieve_total" in metrics
            and len(telemetry.spans) <= 64
        )
        checks["encrypted_event_chain_remains_valid"] = database.verify_event_chain("cosmic-lab")

        checkpoint_path = Path(directory) / "checkpoint.sqlite3"
        manifest = CosmicCheckpointManager.create(database, checkpoint_path)
        verified = CosmicCheckpointManager.verify(checkpoint_path, manifest["sha256"])
        recovery_path = Path(directory) / "recovered.sqlite3"
        recovered = CosmicCheckpointManager.recover_to(checkpoint_path, recovery_path)
        checks["online_checkpoint_is_integral_and_digest_verified"] = (
            verified["integrity"] == "ok" and verified["digest_matches"]
        )
        checks["recovery_drill_uses_new_non_overwriting_database"] = (
            recovered["integrity"] == "ok" and recovery_path.exists()
        )

        revision = cosmic.durable.revision
        tidal_updates = cosmic.neural_tide.updates
        database.close()

        reopened_db = UMD36Database(path, key)
        reopened_durable = UMD36TenantMemory(
            reopened_db, "cosmic-lab", "owner", validation_source="synthetic"
        )
        reopened = UMD38CosmicMemory(reopened_durable, config=UMD38Config())
        checks["restart_restores_cosmic_revision"] = reopened.durable.revision == revision
        checks["restart_restores_neural_tide"] = reopened.neural_tide.updates == tidal_updates
        checks["restart_rebuilds_searchable_orbits"] = (
            reopened.retrieve("ZXQ-9917", top_k=1)[0].memory_id == alpha.memory_id
        )
        snapshot = reopened.snapshot()
        checks["snapshot_reports_umd38_constructions"] = (
            snapshot["version"] == "UMD 3.8"
            and snapshot["index_full_rebuilds"] == 1
            and snapshot["tidal_training_updates"] == tidal_updates
        )
        reopened_db.close()

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("UMD 3.8 checks failed: " + ", ".join(failed))
    return checks


if __name__ == "__main__":
    results = run()
    print(json.dumps({"passed": sum(results.values()), "checks": results}, indent=2))
