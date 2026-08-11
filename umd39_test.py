"""End-to-end platform, SDK, authentication, and console tests for UMD 3.9."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from umd35_core import EntityRelation, UMD35Config
from umd36_persistent import UMD36Database, UMD36TenantMemory, generate_master_key
from umd38_cosmic import UMD38Config, UMD38CosmicMemory
from umd39_platform import (
    APIKeyAuthority,
    OrbitQuota,
    OrbitRateLimiter,
    PlatformError,
    Principal,
    UMD39Platform,
)
from umd_sdk import AsyncUMDClient, UMDAPIError, UMDClient


def run() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "umd39.sqlite3"
        database = UMD36Database(path, generate_master_key())
        database.create_tenant("observatory", "owner")
        database.create_tenant("other", "other-owner")
        config = UMD35Config(
            max_active_total=100, max_active_per_star=100,
            duplicate_threshold=0.999, relation_similarity_threshold=1.1,
        )
        sessions = {}
        for tenant, principal in (("observatory", "owner"), ("other", "other-owner")):
            sessions[(tenant, principal)] = UMD38CosmicMemory(
                UMD36TenantMemory(
                    database, tenant, principal, config=config,
                    validation_source="synthetic",
                ),
                config=UMD38Config(),
            )
        authority = APIKeyAuthority(database, b"platform-pepper-" * 3)
        created = authority.create("observatory", "owner", "integration")
        other_created = authority.create("other", "other-owner", "isolation")
        platform = UMD39Platform(lambda tenant, principal: sessions[(tenant, principal)], authority)
        transport = lambda method, path, headers, body: platform.handle(method, path, headers, body)
        client = UMDClient("local://observatory", created["token"], transport=transport)

        raw_database = path.read_bytes()
        checks["api_key_secret_is_never_stored_raw"] = created["token"].encode() not in raw_database
        principal = authority.authenticate(created["token"])
        checks["hashed_api_key_restores_tenant_principal"] = (
            principal.tenant_id == "observatory" and principal.principal_id == "owner"
        )

        health_status, health, _ = platform.handle("GET", "/health", {}, b"")
        openapi_status, openapi, _ = platform.handle("GET", "/openapi.json", {}, b"")
        checks["health_and_openapi_are_public"] = (
            health_status == 200 and health["data"]["version"] == "UMD 3.9"
            and openapi_status == 200 and openapi["openapi"] == "3.1.0"
        )
        denied_status, denied, _ = platform.handle("GET", "/v1/system", {}, b"")
        checks["private_routes_require_bearer_auth"] = (
            denied_status == 401 and denied["error"]["code"] == "authentication_required"
        )

        events_before = database.counts("observatory")["events"]
        first = client.add(
            "Project Atlas uses PostgreSQL 17 for production workloads.",
            scope="atlas", tags=["engineering"], importance=1.0,
            entities=["Project Atlas", "PostgreSQL 17"],
            idempotency_key="write-atlas-0001",
        )
        replay = client.add(
            "Project Atlas uses PostgreSQL 17 for production workloads.",
            scope="atlas", tags=["engineering"], importance=1.0,
            entities=["Project Atlas", "PostgreSQL 17"],
            idempotency_key="write-atlas-0001",
        )
        events_after = database.counts("observatory")["events"]
        checks["sdk_write_is_idempotent_across_retries"] = (
            first.memory_id == replay.memory_id and events_after - events_before == 1
        )
        try:
            client.add("Different request", idempotency_key="write-atlas-0001")
            conflict = False
        except UMDAPIError as error:
            conflict = error.status == 409 and error.code == "idempotency_conflict"
        checks["idempotency_key_reuse_conflict_is_explicit"] = conflict

        second = client.add(
            "Project Selene depends on Project Atlas for launch telemetry.",
            scope="selene", importance=1.0,
            entities=["Project Selene", "Project Atlas"],
            idempotency_key="write-selene-001",
        )
        # Add a typed relation through the core to make graph/community views nontrivial.
        sessions[("observatory", "owner")].write(
            "Selene control service reports to Atlas mission control.",
            scope="selene", entities=["Selene control", "Atlas mission control"],
            entity_relations=[EntityRelation(
                "Selene control", "reports_to", "Atlas mission control", True, 0.96
            )], explicit_importance=1.0,
        )
        retrieved = client.get(first.memory_id)
        checks["python_sdk_returns_typed_memory"] = (
            retrieved.id == first.memory_id and retrieved.scope == "atlas"
            and retrieved.tags == ("engineering",)
        )
        page1, cursor = client.list_memories(limit=1)
        page2, _ = client.list_memories(limit=1, cursor=cursor)
        checks["cursor_pagination_is_stable"] = (
            len(page1) == len(page2) == 1 and page1[0].id != page2[0].id and cursor is not None
        )

        hits = client.search("Which project uses PostgreSQL 17?", top_k=4)
        checks["sdk_search_exposes_planetary_force"] = (
            hits[0].memory_id == first.memory_id and hits[0].force > 0
            and hits[0].components["neural_rerank"] > 0
        )
        feedback = client.feedback(
            "Which project uses PostgreSQL 17?", first.memory_id,
            [item.memory_id for item in hits[1:]],
        )
        checks["sdk_feedback_trains_force_and_tidal_network"] = (
            "semantic_gravity" in feedback and "tidal_pairwise_loss" in feedback
        )
        graph = client.graph()
        entities = client.entities()
        communities = client.communities()
        audit = client.audit()
        snapshot = client.system()
        checks["observatory_management_surfaces_are_live"] = (
            bool(graph["nodes"]) and bool(graph["edges"]) and bool(entities)
            and bool(communities) and bool(audit) and snapshot["version"] == "UMD 3.8"
        )

        other_client = UMDClient("local://other", other_created["token"], transport=transport)
        other_client.add("Other tenant private memory.", importance=1.0, idempotency_key="other-write-0001")
        other_memories, _ = other_client.list_memories(limit=20)
        primary_memories, _ = client.list_memories(limit=20)
        checks["sdk_preserves_tenant_isolation"] = (
            len(other_memories) == 1
            and all(item.text != "Other tenant private memory." for item in primary_memories)
        )

        too_small = OrbitRateLimiter(OrbitQuota(requests_per_minute=1))
        p = Principal("observatory", "owner", created["key_id"])
        too_small.check(p, "read")
        try:
            too_small.check(p, "read")
            limited = False
        except PlatformError as error:
            limited = error.status == 429
        checks["per_principal_orbit_quota_is_enforced"] = limited

        async def async_probe():
            async_client = AsyncUMDClient("local://observatory", created["token"], transport=transport)
            return await async_client.search("PostgreSQL 17", top_k=1)
        async_hits = asyncio.run(async_probe())
        checks["async_python_sdk_uses_same_contract"] = async_hits[0].memory_id == first.memory_id

        console_status, console_raw, console_type = platform._static("/")
        traversal = platform._static("/console/../umd39_platform.py")
        checks["console_assets_are_served_without_path_traversal"] = (
            console_status == 200 and b"UMD Observatory" in console_raw
            and console_type == "text/html" and traversal[0] == 404
        )

        server = platform.make_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/console/app.js", timeout=2
            ) as response:
                console_js = response.read()
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/system",
                headers={"Authorization": "Bearer " + created["token"]},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                remote_system = json.loads(response.read())
            checks["real_http_server_hosts_console_and_api"] = (
                b"function drawGraph" in console_js
                and remote_system["data"]["version"] == "UMD 3.8"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        usage = database.connection.execute(
            "SELECT COUNT(*) FROM umd_usage WHERE tenant_id=?", ("observatory",)
        ).fetchone()[0]
        checks["platform_usage_ledger_records_operations"] = usage >= 10
        revoked = authority.revoke("observatory", "owner", created["key_id"])
        try:
            authority.authenticate(created["token"])
            revoke_enforced = False
        except PlatformError:
            revoke_enforced = True
        checks["api_key_revocation_is_immediate"] = revoked and revoke_enforced
        checks["event_chain_remains_valid_after_platform_workflow"] = database.verify_event_chain("observatory")
        checks["typescript_sdk_has_runtime_and_types"] = (
            (Path(__file__).with_name("sdk-typescript") / "src" / "index.mjs").is_file()
            and (Path(__file__).with_name("sdk-typescript") / "src" / "index.d.ts").is_file()
        )
        database.close()

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("UMD 3.9 checks failed: " + ", ".join(failed))
    return checks


if __name__ == "__main__":
    results = run()
    print(json.dumps({"passed": sum(results.values()), "checks": results}, indent=2))

