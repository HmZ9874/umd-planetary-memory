"""UMD 3.9 Observatory platform boundary.

This module turns the UMD 3.8 engine into a stable developer surface: hashed
API keys, quotas, idempotent writes, versioned REST envelopes, OpenAPI, graph
and audit inspection, Prometheus metrics, and a static Observatory console.
It deliberately depends only on the Python standard library.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import re
import secrets
import sqlite3
import ssl
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

try:
    from .umd35_core import MemoryNode
    from .umd36_persistent import AuthorizationError
    from .umd38_cosmic import UMD38CosmicMemory
except ImportError:
    from umd35_core import MemoryNode
    from umd36_persistent import AuthorizationError
    from umd38_cosmic import UMD38CosmicMemory


API_VERSION = "2026-08-09"
SERVER_VERSION = "UMD 3.9"


class PlatformError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or {}


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    principal_id: str
    key_id: str


@dataclass
class OrbitQuota:
    requests_per_minute: int = 600
    writes_per_minute: int = 120
    max_write_characters: int = 20_000
    max_page_size: int = 200
    max_top_k: int = 100


class APIKeyAuthority:
    """Tenant API keys stored as peppered hashes; raw secrets are shown once."""

    def __init__(self, database, pepper: bytes) -> None:
        if len(pepper) < 32:
            raise ValueError("API key pepper must be at least 32 bytes")
        self.database, self.pepper = database, pepper
        with database._lock, database.connection:
            database.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS umd_api_keys (
                    key_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS umd_api_key_tenant_idx
                    ON umd_api_keys(tenant_id, principal_id);
                CREATE TABLE IF NOT EXISTS umd_idempotency (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS umd_usage (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    request_bytes INTEGER NOT NULL,
                    response_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS umd_usage_tenant_idx
                    ON umd_usage(tenant_id, created_at);
                """
            )

    def _hash(self, token: str) -> str:
        return hmac.new(self.pepper, token.encode("utf-8"), hashlib.sha256).hexdigest()

    def create(
        self, tenant_id: str, principal_id: str, name: str,
        expires_at: datetime | None = None,
    ) -> dict[str, str | None]:
        self.database.require(tenant_id, principal_id, "reader")
        key_id = "key_" + secrets.token_hex(8)
        token = f"umd_{key_id}.{secrets.token_urlsafe(32)}"
        with self.database._lock, self.database.connection:
            self.database.connection.execute(
                "INSERT INTO umd_api_keys(key_id,tenant_id,principal_id,name,token_hash,created_at,expires_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    key_id, tenant_id, principal_id, name[:100], self._hash(token),
                    datetime.now(timezone.utc).isoformat(), expires_at.isoformat() if expires_at else None,
                ),
            )
        return {"key_id": key_id, "token": token, "expires_at": expires_at.isoformat() if expires_at else None}

    def authenticate(self, token: str) -> Principal:
        digest = self._hash(token)
        with self.database._lock:
            row = self.database.connection.execute(
                "SELECT key_id,tenant_id,principal_id,expires_at,revoked_at FROM umd_api_keys WHERE token_hash=?",
                (digest,),
            ).fetchone()
        if row is None or row["revoked_at"]:
            raise PlatformError(401, "invalid_api_key", "API key is invalid or revoked")
        if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            raise PlatformError(401, "expired_api_key", "API key has expired")
        self.database.require(row["tenant_id"], row["principal_id"], "reader")
        return Principal(row["tenant_id"], row["principal_id"], row["key_id"])

    def revoke(self, tenant_id: str, actor: str, key_id: str) -> bool:
        self.database.require(tenant_id, actor, "admin")
        with self.database._lock, self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE umd_api_keys SET revoked_at=? WHERE tenant_id=? AND key_id=? AND revoked_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), tenant_id, key_id),
            )
        return cursor.rowcount == 1


class OrbitRateLimiter:
    def __init__(self, quota: OrbitQuota | None = None) -> None:
        self.quota = quota or OrbitQuota()
        self.events: dict[tuple[str, str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def check(self, principal: Principal, operation: str) -> None:
        limit = self.quota.writes_per_minute if operation == "write" else self.quota.requests_per_minute
        key = (principal.tenant_id, principal.principal_id, operation)
        now = time.monotonic()
        with self._lock:
            events = self.events[key]
            while events and now - events[0] >= 60.0:
                events.popleft()
            if len(events) >= limit:
                raise PlatformError(429, "orbit_quota_exceeded", "request orbit quota exceeded", {"limit": limit})
            events.append(now)


class IdempotencyLedger:
    def __init__(self, database) -> None:
        self.database = database

    def replay(self, tenant_id: str, key: str, digest: str) -> tuple[int, object] | None:
        with self.database._lock:
            row = self.database.connection.execute(
                "SELECT request_digest,status,response_json FROM umd_idempotency "
                "WHERE tenant_id=? AND idempotency_key=?", (tenant_id, key),
            ).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row["request_digest"], digest):
            raise PlatformError(409, "idempotency_conflict", "idempotency key was used with another request")
        return int(row["status"]), json.loads(row["response_json"])

    def store(self, tenant_id: str, key: str, digest: str, status: int, response: object) -> None:
        with self.database._lock, self.database.connection:
            self.database.connection.execute(
                "INSERT INTO umd_idempotency(tenant_id,idempotency_key,request_digest,status,response_json,created_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(tenant_id,idempotency_key) DO NOTHING",
                (
                    tenant_id, key, digest, status,
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items() if key != "vector"}
    return value


def memory_document(node: MemoryNode) -> dict[str, object]:
    policy = node.extraction_metadata.get("umd37_policy", {})
    return {
        "id": node.id, "text": node.text, "state": node.state,
        "source": node.source, "scope": node.scope, "kind": node.kind,
        "created_at": node.created_at.isoformat(), "updated_at": node.updated_at.isoformat(),
        "valid_from": node.valid_from.isoformat() if node.valid_from else None,
        "valid_to": node.valid_to.isoformat() if node.valid_to else None,
        "star_id": node.star_id, "planet_id": node.planet_id,
        "entities": sorted(node.entities), "entity_types": node.entity_types,
        "mass": node.mass, "confidence": node.confidence, "utility": node.utility_ema,
        "classification": policy.get("classification", "internal"),
        "tags": policy.get("tags", []), "facts": _jsonable(node.facts),
        "relations": _jsonable(node.entity_relations),
    }


class UMD39Platform:
    """Stable REST contract over tenant-specific UMD 3.8 sessions."""

    def __init__(
        self,
        memory_lookup: Callable[[str, str], UMD38CosmicMemory],
        authority: APIKeyAuthority,
        *,
        quota: OrbitQuota | None = None,
        console_directory: str | Path | None = None,
        external_token_verifier: Callable[[str], Principal] | None = None,
    ) -> None:
        self.memory_lookup = memory_lookup
        self.authority = authority
        self.quota = quota or OrbitQuota()
        self.limiter = OrbitRateLimiter(self.quota)
        self.idempotency = IdempotencyLedger(authority.database)
        self.external_token_verifier = external_token_verifier
        self.console_directory = Path(console_directory or Path(__file__).with_name("observatory_console")).resolve()

    @staticmethod
    def _header(headers: dict[str, str], name: str) -> str:
        wanted = name.casefold()
        return next((str(value) for key, value in headers.items() if key.casefold() == wanted), "")

    def _authenticate(self, headers: dict[str, str]) -> Principal:
        raw = self._header(headers, "authorization")
        if not raw.startswith("Bearer "):
            raise PlatformError(401, "authentication_required", "Bearer API key is required")
        token = raw[7:]
        if token.startswith("umd_"):
            return self.authority.authenticate(token)
        if self.external_token_verifier:
            return self.external_token_verifier(token)
        raise PlatformError(401, "unsupported_token", "No verifier is configured for this token")

    @staticmethod
    def _request_id(headers: dict[str, str]) -> str:
        supplied = UMD39Platform._header(headers, "x-request-id")
        return supplied[:80] if re.fullmatch(r"[A-Za-z0-9._:-]{1,80}", supplied) else "req_" + uuid.uuid4().hex

    @staticmethod
    def _success(data: object, request_id: str, **meta) -> dict[str, object]:
        return {"data": data, "meta": {"request_id": request_id, "api_version": API_VERSION, **meta}}

    @staticmethod
    def _error(error: PlatformError, request_id: str) -> dict[str, object]:
        return {"error": {
            "code": error.code, "message": error.message,
            "request_id": request_id, "details": error.details,
        }}

    @staticmethod
    def _body(raw: bytes) -> dict:
        try:
            value = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PlatformError(400, "invalid_json", "request body must be valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise PlatformError(400, "invalid_body", "request body must be a JSON object")
        return value

    @staticmethod
    def _cursor_encode(updated_at: str, memory_id: str) -> str:
        return base64.urlsafe_b64encode(f"{updated_at}|{memory_id}".encode()).decode().rstrip("=")

    @staticmethod
    def _cursor_decode(value: str) -> tuple[str, str]:
        try:
            raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
            return tuple(raw.split("|", 1))  # type: ignore[return-value]
        except Exception as error:
            raise PlatformError(400, "invalid_cursor", "pagination cursor is malformed") from error

    def _list_memories(self, memory: UMD38CosmicMemory, query: dict[str, list[str]]) -> tuple[list[dict], str | None]:
        try:
            limit = max(1, min(self.quota.max_page_size, int(query.get("limit", [50])[0])))
        except ValueError as error:
            raise PlatformError(400, "invalid_limit", "limit must be an integer") from error
        clauses, values = ["tenant_id=?"], [memory.durable.tenant_id]
        state, scope = query.get("state", [""])[0], query.get("scope", [""])[0]
        if state:
            clauses.append("state=?")
            values.append(state)
        cursor = query.get("cursor", [""])[0]
        if cursor:
            updated_at, memory_id = self._cursor_decode(cursor)
            clauses.append("(updated_at < ? OR (updated_at = ? AND memory_id < ?))")
            values.extend([updated_at, updated_at, memory_id])
        rows = memory.durable.database.connection.execute(
            "SELECT memory_id,updated_at FROM memory_records WHERE " + " AND ".join(clauses)
            + " ORDER BY updated_at DESC,memory_id DESC LIMIT ?",
            (*values, limit + 1),
        ).fetchall()
        documents = []
        usable = rows[:limit]
        for row in usable:
            node = memory.durable.get_memory(row["memory_id"])
            if scope and node.scope != scope:
                continue
            if memory._allowed_node(node):
                documents.append(memory_document(node))
        next_cursor = None
        if len(rows) > limit and usable:
            next_cursor = self._cursor_encode(usable[-1]["updated_at"], usable[-1]["memory_id"])
        return documents, next_cursor

    def _graph(self, memory: UMD38CosmicMemory, limit: int = 500) -> dict[str, object]:
        engine = memory.durable.engine
        nodes, edges = [], []
        for star in list(engine.stars.values())[:limit]:
            nodes.append({"id": star.id, "type": "star", "label": star.label, "mass": star.mass})
        for planet in list(engine.planets.values())[:limit]:
            nodes.append({"id": planet.id, "type": "planet", "label": planet.label, "mass": planet.mass})
            edges.append({"source": planet.star_id, "target": planet.id, "type": "contains"})
        for item in list(engine.nodes.values())[:limit]:
            if memory._allowed_node(item):
                nodes.append({"id": item.id, "type": "memory", "label": item.text[:100], "state": item.state})
                if item.planet_id:
                    edges.append({"source": item.planet_id, "target": item.id, "type": "contains"})
        for entity in list(memory.entity_system.entities.values())[:limit]:
            nodes.append({"id": entity.id, "type": "entity", "label": entity.canonical_name,
                          "entity_type": entity.entity_type, "confidence": entity.confidence})
        for orbit in list(memory.entity_system.orbits.values())[:limit]:
            if orbit.object_entity_id:
                edges.append({
                    "id": orbit.id, "source": orbit.subject_id, "target": orbit.object_entity_id,
                    "type": orbit.predicate, "confidence": orbit.confidence,
                    "valid_from": orbit.valid_from.isoformat(),
                    "valid_to": orbit.valid_to.isoformat() if orbit.valid_to else None,
                    "evidence_memory_id": orbit.evidence_memory_id,
                })
        return {"nodes": nodes, "edges": edges, "truncated": len(nodes) >= limit}

    def openapi(self) -> dict[str, object]:
        endpoints = {
            "/v1/memories": {"get": "List memories", "post": "Create memory"},
            "/v1/memories/{memory_id}": {"get": "Get one memory"},
            "/v1/search": {"post": "Search by planetary force"},
            "/v1/retrieval-feedback": {"post": "Train force and tidal weights"},
            "/v1/entities": {"get": "List canonical entities"},
            "/v1/ambiguities": {"get": "List identity-margin quarantines"},
            "/v1/graph": {"get": "Inspect hierarchy and temporal graph"},
            "/v1/communities": {"get": "List evidence communities"},
            "/v1/audit": {"get": "List audit events"},
            "/v1/system": {"get": "Read system snapshot"},
            "/metrics": {"get": "Prometheus metrics"},
        }
        return {
            "openapi": "3.1.0", "info": {"title": "UMD Observatory API", "version": API_VERSION},
            "servers": [{"url": "/"}],
            "paths": {path: {method: {"summary": summary} for method, summary in methods.items()}
                      for path, methods in endpoints.items()},
            "components": {"securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}},
            "security": [{"bearerAuth": []}],
        }

    def _route(
        self, method: str, path: str, query: dict[str, list[str]], headers: dict[str, str],
        raw_body: bytes, principal: Principal, memory: UMD38CosmicMemory, request_id: str,
    ) -> tuple[int, object, str]:
        if method == "GET" and path == "/metrics":
            return 200, memory.telemetry.prometheus(), "text/plain; charset=utf-8"
        if method == "GET" and path == "/v1/system":
            return 200, self._success(memory.snapshot(), request_id), "application/json"
        if method == "GET" and path == "/v1/memories":
            documents, cursor = self._list_memories(memory, query)
            return 200, self._success(documents, request_id, next_cursor=cursor), "application/json"
        if method == "POST" and path == "/v1/memories":
            payload = self._body(raw_body)
            text = str(payload.get("text", ""))
            if not text.strip():
                raise PlatformError(422, "text_required", "text is required")
            if len(text) > self.quota.max_write_characters:
                raise PlatformError(413, "memory_too_large", "memory exceeds the write orbit")
            result = memory.write(
                text, scope=payload.get("scope"), tags=payload.get("tags", ()),
                classification=payload.get("classification", "internal"),
                explicit_importance=float(payload.get("importance", 0.0)),
                entities=payload.get("entities"), fact_key=payload.get("fact_key"),
                pinned=bool(payload.get("pinned", False)),
            )
            return 201, self._success({
                "memory_id": result.memory_id, "state": result.state,
                "reason": result.action, "revision": memory.durable.revision,
            }, request_id), "application/json"
        match = re.fullmatch(r"/v1/memories/([A-Za-z0-9_-]+)", path)
        if method == "GET" and match:
            try:
                node = memory.durable.get_memory(match.group(1))
            except KeyError as error:
                raise PlatformError(404, "memory_not_found", "memory does not exist") from error
            if not memory._allowed_node(node):
                raise PlatformError(404, "memory_not_found", "memory does not exist")
            return 200, self._success(memory_document(node), request_id), "application/json"
        if method == "POST" and path == "/v1/search":
            payload = self._body(raw_body)
            question = str(payload.get("query", ""))
            if not question.strip():
                raise PlatformError(422, "query_required", "query is required")
            top_k = max(1, min(self.quota.max_top_k, int(payload.get("top_k", 5))))
            capsule_search = bool(payload.get("capsules", False))
            if capsule_search and bool(payload.get("chronological", False)):
                raise PlatformError(
                    422, "incompatible_search_modes",
                    "capsules and chronological cannot be enabled together",
                )
            if capsule_search:
                search = getattr(memory, "retrieve_capsules", None)
                if not callable(search):
                    raise PlatformError(
                        422, "capsules_unavailable", "this memory engine does not support capsules"
                    )
                hits = search(
                    question, top_k=top_k, scope=payload.get("scope"),
                    budget_chars=payload.get("budget_chars"),
                )
                data = [{
                    "capsule_id": hit.capsule_id, "kind": hit.kind,
                    "source_ids": list(hit.source_ids), "text": hit.text,
                    "force": hit.force, "components": hit.components,
                    "explanation": hit.explanation,
                } for hit in hits]
                return 200, self._success(data, request_id), "application/json"
            search = (
                getattr(memory, "chronological_context", None)
                if bool(payload.get("chronological", False)) else None
            )
            search = search if callable(search) else memory.retrieve
            hits = search(question, top_k=top_k, scope=payload.get("scope"),
                          budget_chars=payload.get("budget_chars"))
            data = [{
                "memory_id": hit.memory_id, "text": hit.text, "force": hit.force,
                "orbit_radius": hit.orbit_radius, "potential_energy": hit.potential_energy,
                "components": hit.components, "explanation": hit.explanation,
            } for hit in hits]
            return 200, self._success(data, request_id), "application/json"
        if method == "POST" and path == "/v1/retrieval-feedback":
            payload = self._body(raw_body)
            weights = memory.retrieval_feedback(
                str(payload["query"]), str(payload["useful_memory_id"]),
                [str(item) for item in payload.get("rejected_memory_ids", [])],
            )
            return 200, self._success(weights, request_id), "application/json"
        if method == "GET" and path == "/v1/entities":
            data = [_jsonable(item) for item in memory.entity_system.entities.values()]
            return 200, self._success(data, request_id), "application/json"
        if method == "GET" and path == "/v1/ambiguities":
            return 200, self._success(
                list(getattr(memory.entity_system, "ambiguities", [])), request_id
            ), "application/json"
        if method == "GET" and path == "/v1/graph":
            limit = max(50, min(2000, int(query.get("limit", [500])[0])))
            return 200, self._success(self._graph(memory, limit), request_id), "application/json"
        if method == "GET" and path == "/v1/communities":
            return 200, self._success(memory.entity_communities(), request_id), "application/json"
        if method == "GET" and path == "/v1/audit":
            limit = max(1, min(500, int(query.get("limit", [100])[0])))
            rows = memory.durable.database.connection.execute(
                "SELECT revision,event_type,aggregate_id,created_at,event_hash FROM event_log "
                "WHERE tenant_id=? ORDER BY revision DESC LIMIT ?",
                (principal.tenant_id, limit),
            ).fetchall()
            return 200, self._success([dict(row) for row in rows], request_id), "application/json"
        raise PlatformError(404, "route_not_found", "API route does not exist")

    def handle(self, method: str, target: str, headers: dict[str, str], body: bytes = b"") -> tuple[int, object, str]:
        started, request_id = time.perf_counter(), self._request_id(headers)
        parsed = urlparse(target)
        if method == "GET" and parsed.path == "/health":
            return 200, self._success({"status": "ok", "version": SERVER_VERSION}, request_id), "application/json"
        if method == "GET" and parsed.path == "/openapi.json":
            return 200, self.openapi(), "application/json"
        principal = None
        try:
            principal = self._authenticate(headers)
            operation = "write" if method in {"POST", "PUT", "PATCH", "DELETE"} else "read"
            self.limiter.check(principal, operation)
            memory = self.memory_lookup(principal.tenant_id, principal.principal_id)
            idem_key = self._header(headers, "idempotency-key") if method == "POST" else ""
            digest = hashlib.sha256(method.encode() + parsed.path.encode() + body).hexdigest()
            if idem_key:
                if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", idem_key):
                    raise PlatformError(400, "invalid_idempotency_key", "idempotency key must be 8-128 safe characters")
                replay = self.idempotency.replay(principal.tenant_id, idem_key, digest)
                if replay:
                    return replay[0], replay[1], "application/json"
            status, payload, content_type = self._route(
                method, parsed.path, parse_qs(parsed.query), headers, body,
                principal, memory, request_id,
            )
            if idem_key and content_type.startswith("application/json"):
                self.idempotency.store(principal.tenant_id, idem_key, digest, status, payload)
            self._record_usage(principal, parsed.path, status, started, len(body), payload)
            return status, payload, content_type
        except PlatformError as error:
            payload = self._error(error, request_id)
            if principal:
                self._record_usage(principal, parsed.path, error.status, started, len(body), payload)
            return error.status, payload, "application/json"
        except AuthorizationError:
            error = PlatformError(403, "permission_denied", "principal is not allowed to perform this operation")
            return 403, self._error(error, request_id), "application/json"
        except (KeyError, ValueError, TypeError) as error:
            wrapped = PlatformError(422, "invalid_request", str(error))
            return 422, self._error(wrapped, request_id), "application/json"

    def _record_usage(
        self, principal: Principal, operation: str, status: int, started: float,
        request_bytes: int, payload: object,
    ) -> None:
        response_bytes = len(payload.encode()) if isinstance(payload, str) else len(json.dumps(payload, default=str).encode())
        try:
            with self.authority.database._lock, self.authority.database.connection:
                self.authority.database.connection.execute(
                    "INSERT INTO umd_usage(tenant_id,principal_id,operation,status,latency_ms,request_bytes,response_bytes,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        principal.tenant_id, principal.principal_id, operation, status,
                        (time.perf_counter() - started) * 1000.0, request_bytes, response_bytes,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except sqlite3.Error:
            pass

    def _static(self, path: str) -> tuple[int, bytes, str] | None:
        relative = "index.html" if path in {"/", "/console"} else path.removeprefix("/console/")
        if path not in {"/", "/console"} and not path.startswith("/console/"):
            return None
        target = (self.console_directory / relative).resolve()
        if self.console_directory not in target.parents and target != self.console_directory:
            return 404, b"not found", "text/plain"
        if not target.is_file():
            return 404, b"not found", "text/plain"
        return 200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream"

    def make_server(
        self, host: str = "127.0.0.1", port: int = 0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> ThreadingHTTPServer:
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def _dispatch(self):
                static = platform._static(urlparse(self.path).path) if self.command == "GET" else None
                if static is not None:
                    status, raw, content_type = static
                else:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length) if length else b""
                    status, payload, content_type = platform.handle(
                        self.command, self.path, dict(self.headers), body
                    )
                    raw = payload.encode() if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store" if self.path.startswith("/v1/") else "public, max-age=60")
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _dispatch
            do_POST = _dispatch

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer((host, port), Handler)
        if ssl_context:
            server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        return server
