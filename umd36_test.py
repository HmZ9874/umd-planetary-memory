"""End-to-end UMD 3.6 persistence, isolation, graph, and recovery checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from umd35_core import UMD35Config
from umd36_persistent import (
    AuthorizationError,
    ConcurrencyError,
    TenantKeyError,
    UMD36Database,
    UMD36TenantMemory,
    generate_master_key,
)


def run() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "umd36.sqlite3"
        key = generate_master_key()
        database = UMD36Database(path, key)
        database.create_tenant("alpha", "alice")
        database.create_tenant("beta", "bob")
        config = UMD35Config(
            max_active_total=8,
            max_active_per_star=8,
            max_relation_degree=8,
            relation_similarity_threshold=1.1,
            max_relation_hops=5,
        )
        alpha = UMD36TenantMemory(
            database,
            "alpha",
            "alice",
            config=config,
            reflector=lambda nodes: "项目知识已复核，共引用%d条活动记忆。" % len(nodes),
            reflection_interval=3,
        )
        alpha.grant("eve", "reader")
        alpha.grant("carol", "writer")
        stale_writer = UMD36TenantMemory(database, "alpha", "carol")

        chain_texts = [
            "Alpha chain 0: Mercury connects the launch note.",
            "Alpha chain 1: Venus connects the design note.",
            "Alpha chain 2: Earth connects the review note.",
            "Alpha chain 3: Mars connects the approval note.",
            "Alpha chain 4: Jupiter holds the final decision.",
        ]
        chain_ids = []
        for index, text in enumerate(chain_texts):
            result = alpha.write(
                text,
                scope="deep-graph",
                entities=[f"chain-{index}"],
                fact_key=f"deep-graph:{index}",
                explicit_importance=1.0,
            )
            chain_ids.append(result.memory_id)
        try:
            stale_writer.write("This stale write must not replace newer state.", explicit_importance=1.0)
            checks["stale_multiuser_writer_is_rejected"] = False
        except ConcurrencyError:
            checks["stale_multiuser_writer_is_rejected"] = True
        for left, right in zip(chain_ids, chain_ids[1:]):
            alpha.link_memories(left, right, weight=0.92)
        paths = alpha.graph_paths(chain_ids[0], target_id=chain_ids[-1], max_hops=5)
        checks["four_hop_graph_reasoning"] = any(item["hops"] == 4 for item in paths)

        reflection_rows = database.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE tenant_id='alpha' AND event_type='memory.reflection'"
        ).fetchone()[0]
        checks["automatic_reflection_is_audited"] = reflection_rows >= 1

        secret = "ALPHA-ONLY-SECRET-7f3b"
        secret_result = alpha.write(
            f"Tenant alpha confidential marker is {secret}.",
            scope="tenant-secret",
            fact_key="tenant-secret:marker",
            explicit_importance=1.0,
        )
        for index in range(24):
            alpha.write(
                f"Alpha capacity record {index}: durable operation value {1000 + index}.",
                scope="capacity",
                fact_key=f"capacity:{index}",
                explicit_importance=1.0,
            )
        before = alpha.snapshot()
        checks["cold_metadata_is_absent_from_ram"] = (
            before["ram_cold_nodes"] == 0
            and before["ram_active_nodes"] <= config.max_active_total
            and before["cold_records"] > 0
        )
        first_cold = alpha.get_memory(chain_ids[0])
        checks["cold_text_and_metadata_are_recoverable"] = first_cold.text == chain_texts[0]
        checks["event_hash_chain_is_valid"] = before["event_chain_valid"]
        checks["encrypted_database_hides_memory_plaintext"] = secret.encode("utf-8") not in path.read_bytes()

        reader = UMD36TenantMemory(database, "alpha", "eve")
        reader_result = reader.retrieve("durable operation", scope="capacity", reinforce=False)
        checks["reader_can_retrieve"] = bool(reader_result)
        try:
            reader.write("reader must not write", explicit_importance=1.0)
            checks["reader_cannot_write"] = False
        except AuthorizationError:
            checks["reader_cannot_write"] = True
        try:
            UMD36TenantMemory(database, "alpha", "mallory")
            checks["unknown_principal_is_denied"] = False
        except AuthorizationError:
            checks["unknown_principal_is_denied"] = True

        beta = UMD36TenantMemory(database, "beta", "bob", config=config)
        beta_result = beta.write(
            "Beta tenant owns an unrelated private memory.",
            explicit_importance=1.0,
        )
        try:
            beta.get_memory(secret_result.memory_id)
            checks["cross_tenant_memory_ids_are_isolated"] = False
        except KeyError:
            checks["cross_tenant_memory_ids_are_isolated"] = True
        checks["tenant_records_are_separate"] = beta.get_text(beta_result.memory_id).startswith("Beta")

        event_count = database.counts("alpha")["events"]
        try:
            database.commit(
                "alpha", "invalid.test", None, {"bad": object()},
                alpha._active_state(), [],
            )
            checks["failed_transaction_rolls_back"] = False
        except TypeError:
            checks["failed_transaction_rolls_back"] = (
                database.counts("alpha")["events"] == event_count
            )

        active_before_restart = set(alpha.engine.active_ids)
        revision_before_restart = alpha.revision
        database.close()

        reopened = UMD36Database(path, key)
        recovered = UMD36TenantMemory(reopened, "alpha", "alice")
        after = recovered.snapshot()
        checks["restart_restores_exact_active_ids"] = set(recovered.engine.active_ids) == active_before_restart
        checks["restart_restores_revision"] = recovered.revision == revision_before_restart
        checks["restart_retrieval_works"] = bool(
            recovered.retrieve("durable operation", scope="capacity", reinforce=False)
        )
        checks["cold_history_survives_restart"] = (
            recovered.get_memory(chain_ids[0]).text == chain_texts[0]
        )
        cold_probe = recovered.cold_history(limit=100)[0]
        cold_search_ids = {
            node.id for node in recovered.search_cold(cold_probe.text, limit=8)
        }
        checks["blind_cold_index_survives_restart"] = cold_probe.id in cold_search_ids
        term_rows = reopened.connection.execute(
            "SELECT term_digest FROM memory_search_terms WHERE tenant_id='alpha' LIMIT 32"
        ).fetchall()
        checks["blind_cold_index_contains_no_plaintext_terms"] = (
            bool(term_rows)
            and all(
                b"mercury" not in bytes(row[0]).lower()
                and b"launch" not in bytes(row[0]).lower()
                for row in term_rows
            )
        )
        with reopened.connection:
            reopened.connection.execute(
                "DELETE FROM memory_search_terms WHERE tenant_id=? AND memory_id=?",
                ("alpha", cold_probe.id),
            )
        migrated_ids = {
            node.id for node in recovered.search_cold(cold_probe.text, limit=8)
        }
        checks["legacy_cold_records_are_lazily_backfilled"] = cold_probe.id in migrated_ids
        checks["validation_metrics_accumulate"] = (
            recovered.snapshot()["validation"]["samples"] >= 1
        )
        checks["short_test_is_not_mislabeled_long_running"] = (
            recovered.snapshot()["validation"]["is_long_running_evidence"] is False
        )
        with reopened.connection:
            reopened.connection.execute("DELETE FROM tenant_state WHERE tenant_id='alpha'")
        reopened.close()

        log_reopened = UMD36Database(path, key)
        log_recovered = UMD36TenantMemory(log_reopened, "alpha", "alice")
        checks["transaction_log_recovers_missing_materialized_state"] = (
            log_recovered.revision == revision_before_restart
            and set(log_recovered.engine.active_ids) == active_before_restart
        )
        rotated_key = generate_master_key()
        rotation = log_reopened.rotate_master_key(rotated_key)
        checks["blind_index_is_rekeyed_with_tenant_payloads"] = (
            rotation["search_terms"] > 0
            and cold_probe.id in {
                node.id for node in log_recovered.search_cold(cold_probe.text, limit=8)
            }
        )
        log_reopened.close()

        wrong = UMD36Database(path, key)
        try:
            UMD36TenantMemory(wrong, "alpha", "alice")
            checks["wrong_master_key_is_rejected"] = False
        except TenantKeyError:
            checks["wrong_master_key_is_rejected"] = True
        finally:
            wrong.close()

        return {
            "version": "UMD 3.6 persistence and isolation",
            "checks": checks,
            "before_restart": before,
            "after_restart": after,
            "four_hop_paths": paths,
        }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not all(result["checks"].values()):
        raise SystemExit(1)
