"""UMD 3.6 durable, encrypted, multi-tenant runtime.

The UMD 3.5 engine remains the bounded active working set. This module moves
every inactive node (including its metadata) to SQLite, persists an encrypted
active-state image, and commits an append-only audit event in the same database
transaction. Tenant payloads use independent HKDF-derived AES-256-GCM keys.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

try:
    from .umd35_core import (
        EntityRelation,
        FactFrame,
        MemoryNode,
        Planet,
        Star,
        UMD35Config,
        UMD35Memory,
        WriteFeatures,
        normalized_tokens,
        utcnow,
    )
except ImportError:
    from umd35_core import (
        EntityRelation,
        FactFrame,
        MemoryNode,
        Planet,
        Star,
        UMD35Config,
        UMD35Memory,
        WriteFeatures,
        normalized_tokens,
        utcnow,
    )


Role = Literal["reader", "writer", "admin"]
ROLE_LEVEL = {"reader": 1, "writer": 2, "admin": 3}
ACTIVE_STATES = {"stable", "provisional"}
SERIALIZABLE_TYPES = {
    item.__name__: item
    for item in (FactFrame, EntityRelation, WriteFeatures, MemoryNode, Planet, Star, UMD35Config)
}


class AuthorizationError(PermissionError):
    pass


class TenantKeyError(ValueError):
    pass


class ConcurrencyError(RuntimeError):
    pass


def generate_master_key() -> bytes:
    """Return a new 256-bit master key. Store it outside the database."""
    return AESGCM.generate_key(bit_length=256)


def master_key_from_base64(value: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(value.encode("ascii"))
    except Exception as error:
        raise TenantKeyError("master key is not valid URL-safe base64") from error
    if len(key) != 32:
        raise TenantKeyError("master key must decode to exactly 32 bytes")
    return key


def master_key_to_base64(key: bytes) -> str:
    if len(key) != 32:
        raise TenantKeyError("master key must be exactly 32 bytes")
    return base64.urlsafe_b64encode(key).decode("ascii")


def _pack(value):
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, np.ndarray):
        return {
            "$ndarray": base64.b64encode(value.tobytes()).decode("ascii"),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if is_dataclass(value):
        return {
            "$type": type(value).__name__,
            "fields": {item.name: _pack(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, dict):
        return {"$dict": [[_pack(key), _pack(item)] for key, item in value.items()]}
    if isinstance(value, set):
        return {"$set": [_pack(item) for item in sorted(value, key=str)]}
    if isinstance(value, tuple):
        return {"$tuple": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported persistent value: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$datetime" in value:
        return datetime.fromisoformat(value["$datetime"])
    if "$ndarray" in value:
        raw = base64.b64decode(value["$ndarray"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
    if "$set" in value:
        return {_unpack(item) for item in value["$set"]}
    if "$tuple" in value:
        return tuple(_unpack(item) for item in value["$tuple"])
    if "$dict" in value:
        return {_unpack(key): _unpack(item) for key, item in value["$dict"]}
    if "$type" in value:
        cls = SERIALIZABLE_TYPES[value["$type"]]
        return cls(**{key: _unpack(item) for key, item in value["fields"].items()})
    return {key: _unpack(item) for key, item in value.items()}


def _json_bytes(value) -> bytes:
    return json.dumps(_pack(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _from_json_bytes(value: bytes):
    return _unpack(json.loads(value.decode("utf-8")))


def _encoder_identity(encoder) -> dict[str, object]:
    return {
        "class": type(encoder).__name__,
        "dimensions": getattr(encoder, "dimensions", None),
        "model": getattr(encoder, "model_name", None),
    }


class TenantCipher:
    def __init__(self, master_key: bytes, tenant_id: str) -> None:
        if len(master_key) != 32:
            raise TenantKeyError("master key must be exactly 32 bytes")
        self.tenant_id = tenant_id
        self.key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=hashlib.sha256(("umd36:" + tenant_id).encode("utf-8")).digest(),
            info=b"umd36-tenant-data-key-v1",
        ).derive(master_key)
        self.aes = AESGCM(self.key)

    def encrypt(self, plaintext: bytes, purpose: str) -> bytes:
        nonce = os.urandom(12)
        aad = f"{self.tenant_id}:{purpose}".encode("utf-8")
        return nonce + self.aes.encrypt(nonce, plaintext, aad)

    def decrypt(self, envelope: bytes, purpose: str) -> bytes:
        if len(envelope) < 29:
            raise TenantKeyError("encrypted payload is truncated")
        nonce, ciphertext = envelope[:12], envelope[12:]
        aad = f"{self.tenant_id}:{purpose}".encode("utf-8")
        try:
            return self.aes.decrypt(nonce, ciphertext, aad)
        except InvalidTag as error:
            raise TenantKeyError("wrong key, tenant, or tampered encrypted payload") from error


class UMD36Database:
    """SQLite persistence with encrypted payloads and transactional audit log."""

    def __init__(self, path: str | Path, master_key: bytes) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.master_key = master_key
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS acl (
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('reader','writer','admin')),
                    PRIMARY KEY(tenant_id, principal_id),
                    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS memory_records (
                    tenant_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    digest TEXT,
                    envelope BLOB NOT NULL,
                    PRIMARY KEY(tenant_id, memory_id),
                    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS memory_cold_idx
                    ON memory_records(tenant_id, active, updated_at DESC);
                CREATE TABLE IF NOT EXISTS memory_search_terms (
                    tenant_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    term_digest BLOB NOT NULL,
                    PRIMARY KEY(tenant_id, memory_id, term_digest),
                    FOREIGN KEY(tenant_id, memory_id)
                        REFERENCES memory_records(tenant_id, memory_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS memory_term_lookup_idx
                    ON memory_search_terms(tenant_id, term_digest, memory_id);
                CREATE TABLE IF NOT EXISTS tenant_state (
                    tenant_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    envelope BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS event_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    envelope BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(tenant_id, revision),
                    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS event_tenant_idx
                    ON event_log(tenant_id, revision);
                CREATE TABLE IF NOT EXISTS validation_samples (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    detail TEXT,
                    source TEXT NOT NULL DEFAULT 'real' CHECK(source IN ('real','synthetic')),
                    created_at TEXT NOT NULL
                );
                """
            )
            validation_columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(validation_samples)")
            }
            if "source" not in validation_columns:
                self.connection.execute(
                    "ALTER TABLE validation_samples ADD COLUMN source TEXT NOT NULL DEFAULT 'real'"
                )
            memory_columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(memory_records)")
            }
            if "digest" not in memory_columns:
                self.connection.execute("ALTER TABLE memory_records ADD COLUMN digest TEXT")
        self.last_commit_node_writes = 0
        self.total_node_writes = 0

    def cipher(self, tenant_id: str) -> TenantCipher:
        return TenantCipher(self.master_key, tenant_id)

    def create_tenant(self, tenant_id: str, admin_principal: str) -> None:
        if not tenant_id.strip() or not admin_principal.strip():
            raise ValueError("tenant and principal cannot be empty")
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO tenants(tenant_id, created_at) VALUES (?, ?) "
                "ON CONFLICT(tenant_id) DO NOTHING",
                (tenant_id, utcnow().isoformat()),
            )
            self.connection.execute(
                "INSERT INTO acl(tenant_id, principal_id, role) VALUES (?, ?, 'admin') "
                "ON CONFLICT(tenant_id, principal_id) DO NOTHING",
                (tenant_id, admin_principal),
            )

    def require(self, tenant_id: str, principal_id: str, minimum: Role) -> Role:
        with self._lock:
            row = self.connection.execute(
                "SELECT role FROM acl WHERE tenant_id=? AND principal_id=?",
                (tenant_id, principal_id),
            ).fetchone()
        if row is None or ROLE_LEVEL[row["role"]] < ROLE_LEVEL[minimum]:
            raise AuthorizationError(
                f"principal {principal_id!r} lacks {minimum} access to tenant {tenant_id!r}"
            )
        return row["role"]

    def load_state(self, tenant_id: str):
        with self._lock:
            row = self.connection.execute(
                "SELECT revision, envelope FROM tenant_state WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
        if row is not None:
            try:
                plaintext = self.cipher(tenant_id).decrypt(bytes(row["envelope"]), "active-state")
                return int(row["revision"]), _from_json_bytes(plaintext)
            except TenantKeyError:
                # A damaged materialized state can be recovered from the
                # transaction log. A wrong key will fail there as well.
                pass
        with self._lock:
            events = self.connection.execute(
                "SELECT revision, event_type, envelope FROM event_log "
                "WHERE tenant_id=? ORDER BY revision DESC", (tenant_id,),
            ).fetchall()
        last_error: TenantKeyError | None = None
        for event in events:
            try:
                raw = self.cipher(tenant_id).decrypt(
                    bytes(event["envelope"]),
                    f"event:{event['revision']}:{event['event_type']}",
                )
                record = _from_json_bytes(raw)
                if isinstance(record, dict) and "state" in record:
                    return int(event["revision"]), record["state"]
            except TenantKeyError as error:
                last_error = error
                continue
        if last_error is not None:
            raise last_error
        return 0, None

    def commit(
        self,
        tenant_id: str,
        event_type: str,
        aggregate_id: str | None,
        event_payload: dict,
        state_payload: dict,
        nodes: Iterable[MemoryNode],
        acl_change: tuple[str, Role] | None = None,
        expected_revision: int | None = None,
        planetary_state_payload: bytes | None = None,
    ) -> int:
        cipher = self.cipher(tenant_id)
        now = utcnow().isoformat()
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT MAX(revision) revision FROM event_log WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
            current_revision = int(row["revision"]) if row and row["revision"] is not None else 0
            if expected_revision is not None and current_revision != expected_revision:
                raise ConcurrencyError(
                    f"stale tenant state: expected revision {expected_revision}, "
                    f"database is at {current_revision}; reopen the tenant session"
                )
            revision = current_revision + 1
            prior = self.connection.execute(
                "SELECT event_hash FROM event_log WHERE tenant_id=? ORDER BY revision DESC LIMIT 1",
                (tenant_id,),
            ).fetchone()
            previous_hash = prior["event_hash"] if prior else "0" * 64
            # Each encrypted log record carries both the operation and its
            # resulting bounded active state. Recovery therefore does not
            # depend solely on the mutable tenant_state materialization.
            event_bytes = _json_bytes({"operation": event_payload, "state": state_payload})
            event_hash = hmac.new(
                cipher.key,
                previous_hash.encode("ascii") + event_type.encode("utf-8") + event_bytes,
                hashlib.sha256,
            ).hexdigest()
            event_envelope = cipher.encrypt(event_bytes, f"event:{revision}:{event_type}")
            self.connection.execute(
                "INSERT INTO event_log(tenant_id, revision, event_type, aggregate_id, "
                "previous_hash, event_hash, envelope, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tenant_id, revision, event_type, aggregate_id, previous_hash,
                    event_hash, sqlite3.Binary(event_envelope), now,
                ),
            )
            node_writes = 0
            for node in nodes:
                node_bytes = _json_bytes(node)
                digest = hashlib.sha256(node_bytes).hexdigest()
                existing = self.connection.execute(
                    "SELECT digest FROM memory_records WHERE tenant_id=? AND memory_id=?",
                    (tenant_id, node.id),
                ).fetchone()
                if existing is not None and existing["digest"] == digest:
                    continue
                envelope = cipher.encrypt(node_bytes, f"memory:{node.id}")
                self.connection.execute(
                    "INSERT INTO memory_records(tenant_id, memory_id, active, state, updated_at, digest, envelope) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(tenant_id, memory_id) DO UPDATE SET "
                    "active=excluded.active, state=excluded.state, updated_at=excluded.updated_at, "
                    "digest=excluded.digest, envelope=excluded.envelope",
                    (
                        tenant_id, node.id, int(node.state in ACTIVE_STATES), node.state,
                        node.updated_at.isoformat(), digest, sqlite3.Binary(envelope),
                    ),
                )
                # Index terms are HMAC digests under the tenant key. They
                # remain searchable after eviction without exposing plaintext.
                self.connection.execute(
                    "DELETE FROM memory_search_terms WHERE tenant_id=? AND memory_id=?",
                    (tenant_id, node.id),
                )
                searchable = " ".join((
                    node.text,
                    node.fact_key or "",
                    " ".join(sorted(node.entities)),
                ))
                term_digests = {
                    self._search_term_digest(cipher, token)
                    for token in normalized_tokens(searchable)
                }
                self.connection.executemany(
                    "INSERT INTO memory_search_terms(tenant_id, memory_id, term_digest) "
                    "VALUES (?, ?, ?)",
                    [
                        (tenant_id, node.id, sqlite3.Binary(term_digest))
                        for term_digest in term_digests
                    ],
                )
                node_writes += 1
            state_envelope = cipher.encrypt(_json_bytes(state_payload), "active-state")
            self.connection.execute(
                "INSERT INTO tenant_state(tenant_id, revision, envelope, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET revision=excluded.revision, "
                "envelope=excluded.envelope, updated_at=excluded.updated_at",
                (tenant_id, revision, sqlite3.Binary(state_envelope), now),
            )
            if acl_change is not None:
                principal_id, role = acl_change
                self.connection.execute(
                    "INSERT INTO acl(tenant_id, principal_id, role) VALUES (?, ?, ?) "
                    "ON CONFLICT(tenant_id, principal_id) DO UPDATE SET role=excluded.role",
                    (tenant_id, principal_id, role),
                )
            if planetary_state_payload is not None:
                planetary_envelope = cipher.encrypt(
                    planetary_state_payload, "planetary-state"
                )
                self.connection.execute(
                    "INSERT INTO planetary_state(tenant_id, revision, envelope) VALUES (?, ?, ?) "
                    "ON CONFLICT(tenant_id) DO UPDATE SET revision=excluded.revision, "
                    "envelope=excluded.envelope",
                    (tenant_id, revision, sqlite3.Binary(planetary_envelope)),
                )
        self.last_commit_node_writes = node_writes
        self.total_node_writes += node_writes
        return revision

    def load_memory(self, tenant_id: str, memory_id: str) -> MemoryNode | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT envelope FROM memory_records WHERE tenant_id=? AND memory_id=?",
                (tenant_id, memory_id),
            ).fetchone()
        if row is None:
            return None
        payload = self.cipher(tenant_id).decrypt(bytes(row["envelope"]), f"memory:{memory_id}")
        node = _from_json_bytes(payload)
        if not isinstance(node, MemoryNode):
            raise TypeError("stored memory payload is not a MemoryNode")
        return node

    def load_active_memories(self, tenant_id: str) -> dict[str, MemoryNode]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT memory_id, envelope FROM memory_records "
                "WHERE tenant_id=? AND active=1", (tenant_id,),
            ).fetchall()
        cipher = self.cipher(tenant_id)
        result: dict[str, MemoryNode] = {}
        for row in rows:
            memory_id = row["memory_id"]
            payload = cipher.decrypt(bytes(row["envelope"]), f"memory:{memory_id}")
            node = _from_json_bytes(payload)
            if not isinstance(node, MemoryNode):
                raise TypeError("stored active payload is not a MemoryNode")
            result[memory_id] = node
        return result

    @staticmethod
    def _search_term_digest(cipher: TenantCipher, token: str) -> bytes:
        """Create a tenant-keyed blind-index term without storing plaintext."""
        return hmac.new(
            cipher.key,
            b"memory-search-v1\0" + token.encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def search_cold_records(
        self, tenant_id: str, query: str, limit: int = 100,
    ) -> list[MemoryNode]:
        """Search the complete cold tier using its encrypted blind token index."""
        tokens = tuple(dict.fromkeys(normalized_tokens(query)))[:32]
        if not tokens:
            return []
        cipher = self.cipher(tenant_id)
        # Lazy, one-time migration for databases created before the blind index
        # existed. Only unindexed cold envelopes are decrypted and backfilled.
        with self._lock:
            missing = self.connection.execute(
                "SELECT r.memory_id, r.envelope FROM memory_records r "
                "WHERE r.tenant_id=? AND r.active=0 AND NOT EXISTS ("
                "SELECT 1 FROM memory_search_terms t WHERE t.tenant_id=r.tenant_id "
                "AND t.memory_id=r.memory_id)",
                (tenant_id,),
            ).fetchall()
        if missing:
            with self._lock, self.connection:
                for row in missing:
                    memory_id = row["memory_id"]
                    node = _from_json_bytes(cipher.decrypt(
                        bytes(row["envelope"]), f"memory:{memory_id}",
                    ))
                    searchable = " ".join((
                        node.text,
                        node.fact_key or "",
                        " ".join(sorted(node.entities)),
                    ))
                    self.connection.executemany(
                        "INSERT OR IGNORE INTO memory_search_terms"
                        "(tenant_id, memory_id, term_digest) VALUES (?, ?, ?)",
                        [
                            (
                                tenant_id, memory_id,
                                sqlite3.Binary(self._search_term_digest(cipher, token)),
                            )
                            for token in set(normalized_tokens(searchable))
                        ],
                    )
        digests = [self._search_term_digest(cipher, token) for token in tokens]
        placeholders = ",".join("?" for _ in digests)
        sql = (
            "SELECT r.memory_id, r.envelope, COUNT(*) AS matches "
            "FROM memory_search_terms t JOIN memory_records r "
            "ON r.tenant_id=t.tenant_id AND r.memory_id=t.memory_id "
            f"WHERE t.tenant_id=? AND r.active=0 AND t.term_digest IN ({placeholders}) "
            "GROUP BY r.memory_id, r.envelope "
            "ORDER BY matches DESC, r.updated_at DESC, r.memory_id LIMIT ?"
        )
        bounded_limit = max(1, min(limit, 1000))
        parameters = (
            tenant_id,
            *[sqlite3.Binary(item) for item in digests],
            bounded_limit,
        )
        with self._lock:
            rows = self.connection.execute(sql, parameters).fetchall()
        return [
            _from_json_bytes(
                cipher.decrypt(bytes(row["envelope"]), f"memory:{row['memory_id']}")
            )
            for row in rows
        ]

    def cold_records(self, tenant_id: str, limit: int = 100) -> list[MemoryNode]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT memory_id, envelope FROM memory_records "
                "WHERE tenant_id=? AND active=0 ORDER BY updated_at DESC LIMIT ?",
                (tenant_id, max(1, min(limit, 5000))),
            ).fetchall()
        cipher = self.cipher(tenant_id)
        return [
            _from_json_bytes(cipher.decrypt(bytes(row["envelope"]), f"memory:{row['memory_id']}"))
            for row in rows
        ]

    def verify_event_chain(self, tenant_id: str) -> bool:
        cipher = self.cipher(tenant_id)
        with self._lock:
            rows = self.connection.execute(
                "SELECT revision, event_type, previous_hash, event_hash, envelope "
                "FROM event_log WHERE tenant_id=? ORDER BY revision", (tenant_id,)
            ).fetchall()
        previous = "0" * 64
        for row in rows:
            if row["previous_hash"] != previous:
                return False
            raw = cipher.decrypt(
                bytes(row["envelope"]), f"event:{row['revision']}:{row['event_type']}"
            )
            expected = hmac.new(
                cipher.key,
                previous.encode("ascii") + row["event_type"].encode("utf-8") + raw,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, row["event_hash"]):
                return False
            previous = row["event_hash"]
        return True

    def counts(self, tenant_id: str) -> dict[str, int]:
        with self._lock:
            memories = self.connection.execute(
                "SELECT COUNT(*) total, COALESCE(SUM(active),0) active FROM memory_records WHERE tenant_id=?",
                (tenant_id,),
            ).fetchone()
            events = self.connection.execute(
                "SELECT COUNT(*) total FROM event_log WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
        return {
            "records": int(memories["total"]),
            "active_records": int(memories["active"]),
            "cold_records": int(memories["total"] - memories["active"]),
            "events": int(events["total"]),
        }

    def record_validation(
        self, tenant_id: str, principal_id: str, operation: str,
        latency_ms: float, success: bool, detail: str | None = None,
        source: Literal["real", "synthetic"] = "real",
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO validation_samples(tenant_id, principal_id, operation, latency_ms, "
                "success, detail, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tenant_id, principal_id, operation, latency_ms, int(success),
                    detail[:500] if detail else None, source, utcnow().isoformat(),
                ),
            )

    def validation_report(self, tenant_id: str) -> dict[str, object]:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) samples, COALESCE(SUM(success),0) successes, "
                "COALESCE(AVG(latency_ms),0) mean_ms, COALESCE(MAX(latency_ms),0) max_ms, "
                "MIN(created_at) started_at, MAX(created_at) last_seen_at "
                "FROM validation_samples WHERE tenant_id=?", (tenant_id,),
            ).fetchone()
            real = self.connection.execute(
                "SELECT COUNT(*) samples, COALESCE(SUM(success),0) successes, "
                "MIN(created_at) started_at, MAX(created_at) last_seen_at "
                "FROM validation_samples WHERE tenant_id=? AND source='real'", (tenant_id,),
            ).fetchone()
            synthetic = self.connection.execute(
                "SELECT COUNT(*) samples FROM validation_samples "
                "WHERE tenant_id=? AND source='synthetic'", (tenant_id,),
            ).fetchone()
        samples = int(row["samples"])
        real_samples = int(real["samples"])
        real_success_rate = float(real["successes"]) / real_samples if real_samples else None
        duration_hours = 0.0
        if real["started_at"] and real["last_seen_at"]:
            duration_hours = max(
                0.0,
                (
                    datetime.fromisoformat(real["last_seen_at"])
                    - datetime.fromisoformat(real["started_at"])
                ).total_seconds() / 3600.0,
            )
        long_running = bool(
            real_samples >= 1000
            and duration_hours >= 168.0
            and real_success_rate is not None
            and real_success_rate >= 0.99
        )
        return {
            "samples": samples,
            "real_samples": real_samples,
            "synthetic_samples": int(synthetic["samples"]),
            "success_rate": float(row["successes"]) / samples if samples else None,
            "real_success_rate": real_success_rate,
            "mean_latency_ms": float(row["mean_ms"]),
            "max_latency_ms": float(row["max_ms"]),
            "started_at": row["started_at"],
            "last_seen_at": row["last_seen_at"],
            "real_duration_hours": duration_hours,
            "long_running_criteria": {
                "minimum_real_samples": 1000,
                "minimum_duration_hours": 168.0,
                "minimum_success_rate": 0.99,
            },
            "is_long_running_evidence": long_running,
        }

    def rotate_master_key(self, new_master_key: bytes) -> dict[str, int]:
        """Atomically re-encrypt every tenant payload and rebuild event HMAC chains."""
        if len(new_master_key) != 32:
            raise TenantKeyError("new master key must be exactly 32 bytes")
        records = states = events = planetary_states = search_terms = 0
        with self._lock, self.connection:
            has_planetary_state = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='planetary_state'"
            ).fetchone() is not None
            tenant_rows = self.connection.execute("SELECT tenant_id FROM tenants").fetchall()
            for tenant_row in tenant_rows:
                tenant_id = tenant_row["tenant_id"]
                old_cipher = self.cipher(tenant_id)
                new_cipher = TenantCipher(new_master_key, tenant_id)
                memory_rows = self.connection.execute(
                    "SELECT memory_id, envelope FROM memory_records WHERE tenant_id=?",
                    (tenant_id,),
                ).fetchall()
                self.connection.execute(
                    "DELETE FROM memory_search_terms WHERE tenant_id=?", (tenant_id,)
                )
                for row in memory_rows:
                    purpose = f"memory:{row['memory_id']}"
                    raw = old_cipher.decrypt(bytes(row["envelope"]), purpose)
                    node = _from_json_bytes(raw)
                    envelope = new_cipher.encrypt(raw, purpose)
                    self.connection.execute(
                        "UPDATE memory_records SET envelope=? WHERE tenant_id=? AND memory_id=?",
                        (sqlite3.Binary(envelope), tenant_id, row["memory_id"]),
                    )
                    searchable = " ".join((
                        node.text,
                        node.fact_key or "",
                        " ".join(sorted(node.entities)),
                    ))
                    digests = {
                        self._search_term_digest(new_cipher, token)
                        for token in normalized_tokens(searchable)
                    }
                    self.connection.executemany(
                        "INSERT INTO memory_search_terms"
                        "(tenant_id, memory_id, term_digest) VALUES (?, ?, ?)",
                        [
                            (tenant_id, row["memory_id"], sqlite3.Binary(item))
                            for item in digests
                        ],
                    )
                    search_terms += len(digests)
                    records += 1
                state_row = self.connection.execute(
                    "SELECT envelope FROM tenant_state WHERE tenant_id=?", (tenant_id,),
                ).fetchone()
                if state_row is not None:
                    raw = old_cipher.decrypt(bytes(state_row["envelope"]), "active-state")
                    envelope = new_cipher.encrypt(raw, "active-state")
                    self.connection.execute(
                        "UPDATE tenant_state SET envelope=? WHERE tenant_id=?",
                        (sqlite3.Binary(envelope), tenant_id),
                    )
                    states += 1
                if has_planetary_state:
                    planetary_row = self.connection.execute(
                        "SELECT envelope FROM planetary_state WHERE tenant_id=?", (tenant_id,),
                    ).fetchone()
                    if planetary_row is not None:
                        raw = old_cipher.decrypt(
                            bytes(planetary_row["envelope"]), "planetary-state"
                        )
                        envelope = new_cipher.encrypt(raw, "planetary-state")
                        self.connection.execute(
                            "UPDATE planetary_state SET envelope=? WHERE tenant_id=?",
                            (sqlite3.Binary(envelope), tenant_id),
                        )
                        planetary_states += 1
                event_rows = self.connection.execute(
                    "SELECT revision, event_type, envelope FROM event_log "
                    "WHERE tenant_id=? ORDER BY revision", (tenant_id,),
                ).fetchall()
                previous_hash = "0" * 64
                for row in event_rows:
                    purpose = f"event:{row['revision']}:{row['event_type']}"
                    raw = old_cipher.decrypt(bytes(row["envelope"]), purpose)
                    event_hash = hmac.new(
                        new_cipher.key,
                        previous_hash.encode("ascii") + row["event_type"].encode("utf-8") + raw,
                        hashlib.sha256,
                    ).hexdigest()
                    envelope = new_cipher.encrypt(raw, purpose)
                    self.connection.execute(
                        "UPDATE event_log SET previous_hash=?, event_hash=?, envelope=? "
                        "WHERE tenant_id=? AND revision=?",
                        (
                            previous_hash, event_hash, sqlite3.Binary(envelope),
                            tenant_id, row["revision"],
                        ),
                    )
                    previous_hash = event_hash
                    events += 1
        self.master_key = new_master_key
        return {
            "memory_records": records,
            "tenant_states": states,
            "planetary_states": planetary_states,
            "search_terms": search_terms,
            "events": events,
        }

    def hard_delete_memory(
        self, tenant_id: str, actor: str, memory_id: str
    ) -> dict[str, int]:
        """Delete a record and redact its content-bearing creation event."""
        self.require(tenant_id, actor, "admin")
        cipher = self.cipher(tenant_id)
        redacted = 0
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM memory_records WHERE tenant_id=? AND memory_id=?",
                (tenant_id, memory_id),
            )
            rows = self.connection.execute(
                "SELECT revision, event_type, aggregate_id, envelope FROM event_log "
                "WHERE tenant_id=? ORDER BY revision", (tenant_id,),
            ).fetchall()
            previous_hash = "0" * 64
            for row in rows:
                purpose = f"event:{row['revision']}:{row['event_type']}"
                raw = cipher.decrypt(bytes(row["envelope"]), purpose)
                record = _from_json_bytes(raw)
                aggregate_id = row["aggregate_id"]
                if aggregate_id == memory_id:
                    record["operation"] = {
                        "redacted": True,
                        "memory_id_sha256": hashlib.sha256(memory_id.encode("utf-8")).hexdigest(),
                    }
                    raw = _json_bytes(record)
                    aggregate_id = None
                    redacted += 1
                event_hash = hmac.new(
                    cipher.key,
                    previous_hash.encode("ascii") + row["event_type"].encode("utf-8") + raw,
                    hashlib.sha256,
                ).hexdigest()
                envelope = cipher.encrypt(raw, purpose)
                self.connection.execute(
                    "UPDATE event_log SET aggregate_id=?, previous_hash=?, event_hash=?, envelope=? "
                    "WHERE tenant_id=? AND revision=?",
                    (
                        aggregate_id, previous_hash, event_hash, sqlite3.Binary(envelope),
                        tenant_id, row["revision"],
                    ),
                )
                previous_hash = event_hash
        return {"records_deleted": 1, "events_redacted": redacted}

    def close(self) -> None:
        with self._lock:
            self.connection.close()


class UMD36TenantMemory:
    """Authorized tenant view over a bounded in-RAM UMD active working set."""

    def __init__(
        self,
        database: UMD36Database,
        tenant_id: str,
        principal_id: str,
        *,
        config: UMD35Config | None = None,
        encoder=None,
        extractor=None,
        reflector: Callable[[list[MemoryNode]], str] | None = None,
        reflection_interval: int = 0,
        validation_source: Literal["real", "synthetic"] = "real",
    ) -> None:
        database.require(tenant_id, principal_id, "reader")
        self.database = database
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.encoder = encoder
        self.extractor = extractor
        self.reflector = reflector
        self.reflection_interval = max(0, reflection_interval)
        if validation_source not in {"real", "synthetic"}:
            raise ValueError("validation_source must be 'real' or 'synthetic'")
        self.validation_source = validation_source
        self._lock = threading.RLock()
        self.revision, state = database.load_state(tenant_id)
        self.engine = self._restore(state, config)
        self._writes_since_reflection = 0

    def _restore(self, state: dict | None, config: UMD35Config | None) -> UMD35Memory:
        if state is None:
            return UMD35Memory(config, encoder=self.encoder, extractor=self.extractor)
        engine = UMD35Memory(state["config"], encoder=self.encoder, extractor=self.extractor)
        expected_encoder = state.get("encoder_identity")
        if expected_encoder is not None and expected_encoder != _encoder_identity(engine.encoder):
            raise ValueError(
                f"encoder mismatch during recovery: expected {expected_encoder}, "
                f"received {_encoder_identity(engine.encoder)}"
            )
        engine.nodes = state.get("nodes") or self.database.load_active_memories(self.tenant_id)
        engine.stars = state["stars"]
        engine.planets = state["planets"]
        engine.scope_stars = state["scope_stars"]
        engine.session_planets = state["session_planets"]
        engine.fact_index = defaultdict(list, state["fact_index"])
        engine.entity_index = defaultdict(set, state["entity_index"])
        engine.relations = defaultdict(dict, state["relations"])
        engine.active_ids = state["active_ids"]
        engine.gate.weights = state["gate_weights"]
        engine.gate.bias = state["gate_bias"]
        engine.gate.updates = state["gate_updates"]
        counters = state.get("extraction_counters", (0, 0, 0))
        engine.extraction_calls, engine.extraction_failures, engine.extraction_skipped_risk = counters
        return engine

    def _active_state(self) -> dict:
        active_nodes = {
            memory_id: node for memory_id, node in self.engine.nodes.items()
            if node.state in ACTIVE_STATES
        }
        active_ids = set(active_nodes) & self.engine.active_ids
        active_fact = {}
        for key, ids in self.engine.fact_index.items():
            kept = [memory_id for memory_id in ids if memory_id in active_nodes]
            if kept:
                active_fact[key] = kept
        active_entities = {
            key: set(ids) & set(active_nodes)
            for key, ids in self.engine.entity_index.items()
            if set(ids) & set(active_nodes)
        }
        active_relations = {
            left: {right: weight for right, weight in edges.items() if right in active_nodes}
            for left, edges in self.engine.relations.items() if left in active_nodes
        }
        active_planets: dict[str, Planet] = {}
        used_planets: set[str] = set()
        for planet_id, planet in self.engine.planets.items():
            ids = set(planet.active_node_ids) & active_ids
            if not ids:
                continue
            copied = _from_json_bytes(_json_bytes(planet))
            copied.node_ids = [item for item in planet.node_ids if item in active_nodes]
            copied.active_node_ids = ids
            active_planets[planet_id] = copied
            used_planets.add(planet_id)
        active_stars: dict[str, Star] = {}
        for star_id, star in self.engine.stars.items():
            planet_ids = [item for item in star.planet_ids if item in used_planets]
            if not planet_ids:
                continue
            copied = _from_json_bytes(_json_bytes(star))
            copied.planet_ids = planet_ids
            active_stars[star_id] = copied
        return {
            "config": self.engine.config,
            "encoder_identity": _encoder_identity(self.engine.encoder),
            "active_node_ids": set(active_nodes),
            "stars": active_stars,
            "planets": active_planets,
            "scope_stars": {
                scope: star_id for scope, star_id in self.engine.scope_stars.items()
                if star_id in active_stars
            },
            "session_planets": {
                key: planet_id for key, planet_id in self.engine.session_planets.items()
                if planet_id in active_planets
            },
            "fact_index": active_fact,
            "entity_index": active_entities,
            "relations": active_relations,
            "active_ids": active_ids,
            "gate_weights": self.engine.gate.weights,
            "gate_bias": self.engine.gate.bias,
            "gate_updates": self.engine.gate.updates,
            "extraction_counters": (
                self.engine.extraction_calls,
                self.engine.extraction_failures,
                self.engine.extraction_skipped_risk,
            ),
        }

    def _evict_cold(self) -> None:
        cold_ids = {
            memory_id for memory_id, node in self.engine.nodes.items()
            if node.state not in ACTIVE_STATES
        }
        if not cold_ids:
            return
        for memory_id in cold_ids:
            self.engine.nodes.pop(memory_id, None)
            self.engine.active_ids.discard(memory_id)
            self.engine.relations.pop(memory_id, None)
        for edges in self.engine.relations.values():
            for memory_id in cold_ids:
                edges.pop(memory_id, None)
        for key in list(self.engine.fact_index):
            self.engine.fact_index[key] = [item for item in self.engine.fact_index[key] if item not in cold_ids]
            if not self.engine.fact_index[key]:
                self.engine.fact_index.pop(key, None)
        for key in list(self.engine.entity_index):
            self.engine.entity_index[key].difference_update(cold_ids)
            if not self.engine.entity_index[key]:
                self.engine.entity_index.pop(key, None)
        for planet in self.engine.planets.values():
            planet.node_ids = [item for item in planet.node_ids if item not in cold_ids]
            planet.active_node_ids.difference_update(cold_ids)
        empty_planets = {
            planet_id for planet_id, planet in self.engine.planets.items()
            if not planet.active_node_ids
        }
        for planet_id in empty_planets:
            self.engine.planets.pop(planet_id, None)
        for star in self.engine.stars.values():
            star.planet_ids = [item for item in star.planet_ids if item not in empty_planets]
        empty_stars = {
            star_id for star_id, star in self.engine.stars.items()
            if not star.planet_ids
        }
        for star_id in empty_stars:
            self.engine.stars.pop(star_id, None)
        self.engine.scope_stars = {
            scope: star_id for scope, star_id in self.engine.scope_stars.items()
            if star_id in self.engine.stars
        }
        self.engine.session_planets = {
            key: planet_id for key, planet_id in self.engine.session_planets.items()
            if planet_id in self.engine.planets
        }
        # UMD35's default archive is an in-memory staging area. Once the full
        # cold record is durable, remove that staging payload as well.
        payloads = getattr(self.engine.archive_store, "payloads", None)
        if isinstance(payloads, dict):
            for memory_id in cold_ids:
                payloads.pop(memory_id, None)

    def _commit(
        self,
        event_type: str,
        aggregate_id: str | None,
        payload: dict,
        *,
        acl_change: tuple[str, Role] | None = None,
        planetary_state_payload: bytes | None = None,
    ) -> None:
        nodes: list[MemoryNode] = []
        for node in self.engine.nodes.values():
            durable = _from_json_bytes(_json_bytes(node))
            if durable.state not in ACTIVE_STATES and not durable.text:
                durable.text = self.engine.get_text(node.id)
            nodes.append(durable)
        state = self._active_state()
        self.revision = self.database.commit(
            self.tenant_id, event_type, aggregate_id, payload, state, nodes,
            acl_change=acl_change,
            expected_revision=self.revision,
            planetary_state_payload=planetary_state_payload,
        )
        self._evict_cold()

    def write(self, text: str, **kwargs):
        self.database.require(self.tenant_id, self.principal_id, "writer")
        started = time.perf_counter()
        try:
            with self._lock:
                result = self.engine.write(text, **kwargs)
                self._commit(
                    "memory.write", result.memory_id,
                    {"text": text, "kwargs": kwargs},
                )
                self._writes_since_reflection += 1
                if (
                    self.reflection_interval
                    and self._writes_since_reflection >= self.reflection_interval
                    and not text.startswith("Reflection summary:")
                ):
                    self.reflect(auto=True)
            self.database.record_validation(
                self.tenant_id, self.principal_id, "write",
                (time.perf_counter() - started) * 1000.0, True,
                source=self.validation_source,
            )
            return result
        except Exception as error:
            self.database.record_validation(
                self.tenant_id, self.principal_id, "write",
                (time.perf_counter() - started) * 1000.0, False, type(error).__name__,
                source=self.validation_source,
            )
            raise

    def grant(self, principal_id: str, role: Role) -> None:
        self.database.require(self.tenant_id, self.principal_id, "admin")
        if role not in ROLE_LEVEL:
            raise ValueError("invalid role")
        with self._lock:
            self._commit(
                "acl.grant", principal_id,
                {"principal_id": principal_id, "role": role},
                acl_change=(principal_id, role),
            )

    def retrieve(self, query: str, **kwargs):
        self.database.require(self.tenant_id, self.principal_id, "reader")
        reinforce = bool(kwargs.get("reinforce", False))
        started = time.perf_counter()
        try:
            result = self.engine.retrieve(query, **kwargs)
            if reinforce:
                self.database.require(self.tenant_id, self.principal_id, "writer")
                self._commit(
                    "memory.retrieve_reinforced", None,
                    {"query": query},
                )
            self.database.record_validation(
                self.tenant_id, self.principal_id, "retrieve",
                (time.perf_counter() - started) * 1000.0, True,
                source=self.validation_source,
            )
            return result
        except Exception as error:
            self.database.record_validation(
                self.tenant_id, self.principal_id, "retrieve",
                (time.perf_counter() - started) * 1000.0, False, type(error).__name__,
                source=self.validation_source,
            )
            raise

    def get_memory(self, memory_id: str) -> MemoryNode:
        self.database.require(self.tenant_id, self.principal_id, "reader")
        node = self.engine.nodes.get(memory_id)
        if node is not None:
            return node
        node = self.database.load_memory(self.tenant_id, memory_id)
        if node is None:
            raise KeyError(memory_id)
        return node

    def get_text(self, memory_id: str) -> str:
        return self.get_memory(memory_id).text

    def cold_history(self, limit: int = 100) -> list[MemoryNode]:
        self.database.require(self.tenant_id, self.principal_id, "reader")
        return self.database.cold_records(self.tenant_id, limit)

    def search_cold(self, query: str, limit: int = 100) -> list[MemoryNode]:
        self.database.require(self.tenant_id, self.principal_id, "reader")
        return self.database.search_cold_records(self.tenant_id, query, limit)

    def decay(self, now: datetime | None = None) -> dict[str, int]:
        self.database.require(self.tenant_id, self.principal_id, "writer")
        with self._lock:
            result = self.engine.decay(now)
            self._commit("memory.decay", None, {"now": now, "result": result})
            return result

    def confirm(self, memory_id: str, *, session_key: str | None = None):
        self.database.require(self.tenant_id, self.principal_id, "writer")
        with self._lock:
            if memory_id not in self.engine.nodes:
                raise ValueError("only an active provisional memory can be confirmed")
            result = self.engine.confirm(memory_id, session_key=session_key)
            self._commit("memory.confirm", memory_id, {"session_key": session_key})
            return result

    def reject(self, memory_id: str):
        self.database.require(self.tenant_id, self.principal_id, "writer")
        with self._lock:
            if memory_id not in self.engine.nodes:
                raise ValueError("only an active provisional memory can be rejected")
            result = self.engine.reject(memory_id)
            self._commit("memory.reject", memory_id, {})
            return result

    def feedback(self, memory_id: str, useful: bool) -> None:
        self.database.require(self.tenant_id, self.principal_id, "writer")
        with self._lock:
            if memory_id not in self.engine.nodes:
                raise ValueError("feedback is only accepted for the active working set")
            self.engine.feedback(memory_id, useful)
            self._commit("memory.feedback", memory_id, {"useful": useful})

    def link_memories(self, left_id: str, right_id: str, *, weight: float = 0.8) -> None:
        """Create an explicit bounded graph edge inside this tenant only."""
        self.database.require(self.tenant_id, self.principal_id, "writer")
        if left_id == right_id:
            raise ValueError("a memory cannot link to itself")
        if left_id not in self.engine.active_ids or right_id not in self.engine.active_ids:
            raise KeyError("both endpoints must be active memories in this tenant")
        bounded = max(0.01, min(1.0, float(weight)))
        with self._lock:
            self.engine.relations[left_id][right_id] = bounded
            self.engine.relations[right_id][left_id] = bounded
            for endpoint in (left_id, right_id):
                if len(self.engine.relations[endpoint]) > self.engine.config.max_relation_degree:
                    weakest = min(
                        self.engine.relations[endpoint],
                        key=self.engine.relations[endpoint].get,
                    )
                    self.engine.relations[endpoint].pop(weakest, None)
                    self.engine.relations[weakest].pop(endpoint, None)
            self._commit(
                "graph.link", left_id,
                {"left_id": left_id, "right_id": right_id, "weight": bounded},
            )

    def graph_paths(
        self,
        start_id: str,
        *,
        target_id: str | None = None,
        target_entities: Iterable[str] = (),
        max_hops: int = 5,
        beam_width: int = 32,
    ) -> list[dict[str, object]]:
        """Return bounded 3+ hop explanations over the active memory graph."""
        self.database.require(self.tenant_id, self.principal_id, "reader")
        if start_id not in self.engine.active_ids:
            raise KeyError("start memory is not active in this tenant")
        max_hops = max(1, min(max_hops, 8))
        beam_width = max(1, min(beam_width, 128))
        wanted = {item.strip().casefold() for item in target_entities if item.strip()}
        frontier = [(1.0, [start_id])]
        matches: list[tuple[float, list[str]]] = []
        for _depth in range(max_hops):
            expanded: list[tuple[float, list[str]]] = []
            for strength, path in frontier:
                for neighbor, weight in self.engine.relations.get(path[-1], {}).items():
                    if neighbor in path or neighbor not in self.engine.active_ids:
                        continue
                    next_path = path + [neighbor]
                    next_strength = strength * weight * self.engine.config.relation_hop_decay
                    node = self.engine.nodes[neighbor]
                    if (target_id and neighbor == target_id) or (wanted and node.entities & wanted):
                        matches.append((next_strength, next_path))
                    expanded.append((next_strength, next_path))
            frontier = sorted(expanded, key=lambda item: item[0], reverse=True)[:beam_width]
            if not frontier:
                break
        return [
            {
                "strength": strength,
                "hops": len(path) - 1,
                "path": path,
                "evidence": [self.engine.nodes[item].text for item in path],
            }
            for strength, path in sorted(matches, key=lambda item: item[0], reverse=True)[:beam_width]
        ]

    def reflect(self, *, auto: bool = False, top_k: int = 12):
        """Create an auditable low-trust summary from the active working set."""
        self.database.require(self.tenant_id, self.principal_id, "writer")
        nodes = sorted(
            self.engine.nodes.values(),
            key=lambda item: (item.utility_ema * item.mass, item.updated_at),
            reverse=True,
        )[: max(1, min(top_k, 32))]
        if not nodes:
            return None
        if self.reflector is not None:
            summary = self.reflector(nodes).strip()
        else:
            claims = []
            for node in nodes:
                fact = node.fact
                if fact.subject and fact.predicate and fact.object_value:
                    claims.append(f"{fact.subject} {fact.predicate} {fact.object_value}")
                else:
                    claims.append(node.text[:180])
            summary = "；".join(dict.fromkeys(claims))
        if not summary:
            return None
        result = self.engine.write(
            "Reflection summary: " + summary[:3000],
            source="assistant_inference",
            kind="observation",
            explicit_importance=0.45,
            auto_extract=False,
        )
        self._commit(
            "memory.reflection", result.memory_id,
            {"auto": auto, "source_ids": [node.id for node in nodes]},
        )
        self._writes_since_reflection = 0
        return result

    def snapshot(self) -> dict[str, object]:
        self.database.require(self.tenant_id, self.principal_id, "reader")
        disk = self.database.counts(self.tenant_id)
        return {
            "version": "UMD 3.6",
            "tenant_id": self.tenant_id,
            "revision": self.revision,
            "ram_active_nodes": len(self.engine.nodes),
            "ram_cold_nodes": sum(node.state not in ACTIVE_STATES for node in self.engine.nodes.values()),
            "event_chain_valid": self.database.verify_event_chain(self.tenant_id),
            "last_commit_node_writes": self.database.last_commit_node_writes,
            "database_node_writes": self.database.total_node_writes,
            **disk,
            "engine": self.engine.snapshot(),
            "validation": self.database.validation_report(self.tenant_id),
        }
