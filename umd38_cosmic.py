"""UMD 3.8 cosmic memory runtime.

UMD 3.8 keeps the planetary hierarchy of 3.7, but closes several engineering
gaps with original, dependency-light constructions:

* an incremental orbital navigable index instead of revision-wide rebuilds;
* a small trainable tidal interaction network for pairwise reranking;
* margin-based entity identity gravity with an ambiguity quarantine;
* one gravity-capsule commit for memory, graph state, and the audit event;
* signed idempotent constellation packets and read-repair planning;
* model-runtime activation, multimodal adapters, telemetry, REST and MCP faces.

The network and runtime interfaces are real adapters, not claims that a local
test process is a cross-region production deployment.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import sqlite3
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol

import numpy as np

try:
    from .umd35_core import MemoryNode, cosine, normalized_text, normalized_tokens, utcnow
    from .umd36_persistent import AuthorizationError
    from .umd37_planetary import (
        CLASSIFICATION_LEVEL,
        PlanetaryConfig,
        PlanetaryIndex,
        TemporalEntitySystem,
        UMD37PlanetaryMemory,
    )
except ImportError:
    from umd35_core import MemoryNode, cosine, normalized_text, normalized_tokens, utcnow
    from umd36_persistent import AuthorizationError
    from umd37_planetary import (
        CLASSIFICATION_LEVEL,
        PlanetaryConfig,
        PlanetaryIndex,
        TemporalEntitySystem,
        UMD37PlanetaryMemory,
    )


@dataclass
class UMD38Config(PlanetaryConfig):
    orbital_degree: int = 12
    orbital_search_width: int = 48
    identity_threshold: float = 0.86
    identity_margin: float = 0.08
    telemetry_span_limit: int = 512
    multimodal_max_bytes: int = 16 * 1024 * 1024


class IncrementalOrbitalIndex(PlanetaryIndex):
    """Incremental LSH/BM25 index plus a bounded navigable neighbor orbit.

    Insertions touch the new item, its LSH candidates, and at most ``degree``
    neighbors. A revision change with no memory changes only advances the index
    clock; it does not rebuild postings or vectors.
    """

    def __init__(
        self, dimensions: int, tables: int = 5, bits: int = 12,
        degree: int = 12, search_width: int = 48,
    ) -> None:
        super().__init__(dimensions, tables, bits)
        self.degree = max(4, min(64, degree))
        self.search_width = max(self.degree, search_width)
        self.vectors: dict[str, np.ndarray] = {}
        self.neighbors: dict[str, dict[str, float]] = defaultdict(dict)
        self.fingerprints: dict[str, str] = {}
        self.total_terms = 0
        self.full_rebuilds = 0
        self.incremental_upserts = 0
        self.incremental_removals = 0

    @staticmethod
    def _fingerprint(node: MemoryNode) -> str:
        digest = hashlib.sha256()
        digest.update(node.id.encode("utf-8"))
        digest.update(node.text.encode("utf-8"))
        digest.update(node.vector.astype(np.float32, copy=False).tobytes())
        digest.update(node.state.encode("ascii"))
        return digest.hexdigest()

    def _unlink(self, memory_id: str) -> None:
        vector = self.vectors.get(memory_id)
        if vector is not None:
            for table in range(self.tables):
                bucket = self.buckets[table].get(self._signature(vector, table))
                if bucket is not None:
                    bucket.discard(memory_id)
        terms = self.tokens.pop(memory_id, [])
        counts = self.term_frequencies.pop(memory_id, Counter())
        self.total_terms -= len(terms)
        for term in counts:
            self.document_frequency[term] -= 1
            if self.document_frequency[term] <= 0:
                self.document_frequency.pop(term, None)
        for other in list(self.neighbors.pop(memory_id, {})):
            self.neighbors[other].pop(memory_id, None)
        self.vectors.pop(memory_id, None)
        self.fingerprints.pop(memory_id, None)

    def remove(self, memory_id: str) -> None:
        if memory_id in self.fingerprints:
            self._unlink(memory_id)
            self.incremental_removals += 1
            self.average_length = self.total_terms / max(1, len(self.tokens))

    def upsert(self, node: MemoryNode) -> bool:
        if node.vector.shape != (self.dimensions,) or node.state not in {"stable", "provisional"}:
            self.remove(node.id)
            return False
        fingerprint = self._fingerprint(node)
        if self.fingerprints.get(node.id) == fingerprint:
            return False
        if node.id in self.fingerprints:
            self._unlink(node.id)
        vector = node.vector.astype(np.float32, copy=True)
        candidates: set[str] = set()
        for table in range(self.tables):
            signature = self._signature(vector, table)
            candidates.update(self.buckets[table].get(signature, ()))
            if len(candidates) < self.search_width:
                for bit in range(self.bits):
                    candidates.update(self.buckets[table].get(signature ^ (1 << bit), ()))
                    if len(candidates) >= self.search_width:
                        break
        if not candidates and self.vectors:
            # A few deterministic entry suns prevent isolated new components.
            candidates.update(sorted(self.vectors)[: min(self.degree, len(self.vectors))])
        ranked = sorted(
            ((max(0.0, cosine(vector, self.vectors[item])), item) for item in candidates),
            reverse=True,
        )[: self.degree]
        self.vectors[node.id] = vector
        self.neighbors[node.id] = {item: score for score, item in ranked}
        for score, other in ranked:
            edges = self.neighbors[other]
            edges[node.id] = score
            if len(edges) > self.degree:
                weakest = min(edges, key=edges.get)
                edges.pop(weakest, None)
                self.neighbors[weakest].pop(other, None)
        terms = normalized_tokens(node.text)
        counts = Counter(terms)
        self.tokens[node.id] = terms
        self.term_frequencies[node.id] = counts
        self.document_frequency.update(counts.keys())
        self.total_terms += len(terms)
        for table in range(self.tables):
            self.buckets[table][self._signature(vector, table)].add(node.id)
        self.fingerprints[node.id] = fingerprint
        self.incremental_upserts += 1
        self.average_length = self.total_terms / max(1, len(self.tokens))
        return True

    def rebuild(self, nodes: dict[str, MemoryNode], revision: int) -> None:
        self.buckets = [defaultdict(set) for _ in range(self.tables)]
        self.tokens.clear()
        self.term_frequencies.clear()
        self.document_frequency.clear()
        self.vectors.clear()
        self.neighbors.clear()
        self.fingerprints.clear()
        self.total_terms = 0
        for node in nodes.values():
            self.upsert(node)
        self.revision = revision
        self.full_rebuilds += 1

    def synchronize(self, nodes: dict[str, MemoryNode], revision: int) -> dict[str, int]:
        active = {
            key for key, node in nodes.items()
            if node.state in {"stable", "provisional"}
            and node.vector.shape == (self.dimensions,)
        }
        removed = 0
        for memory_id in set(self.fingerprints) - active:
            self.remove(memory_id)
            removed += 1
        changed = sum(bool(self.upsert(nodes[memory_id])) for memory_id in active)
        self.revision = revision
        return {"upserts": changed, "removals": removed, "indexed": len(self.tokens)}

    def ann_candidates(self, vector: np.ndarray) -> set[str]:
        seeds = super().ann_candidates(vector)
        if not seeds and self.vectors:
            seeds = set(sorted(self.vectors)[: min(4, len(self.vectors))])
        frontier = sorted(
            ((max(0.0, cosine(vector, self.vectors[item])), item) for item in seeds),
            reverse=True,
        )[: self.search_width]
        visited = {item for _, item in frontier}
        best = list(frontier)
        cursor = 0
        while cursor < len(frontier) and len(visited) < self.search_width * 3:
            _, current = frontier[cursor]
            cursor += 1
            additions = []
            for other in self.neighbors.get(current, {}):
                if other not in visited:
                    visited.add(other)
                    additions.append((max(0.0, cosine(vector, self.vectors[other])), other))
            if additions:
                best.extend(additions)
                frontier.extend(sorted(additions, reverse=True)[: self.search_width])
        return {item for _, item in sorted(best, reverse=True)[: self.search_width]}


class NeuralTidalReranker:
    """A bounded online MLP over query-memory interaction features."""

    def __init__(self, encoder, seed: int = 38017) -> None:
        self.encoder = encoder
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.18, (7, 10)).astype(np.float32)
        self.b1 = np.zeros(10, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.18, 10).astype(np.float32)
        self.b2 = np.float32(0.0)
        self.updates = 0

    def _features(self, query: str, text: str) -> np.ndarray:
        qtokens, ttokens = normalized_tokens(query), normalized_tokens(text)
        qs, ts = set(qtokens), set(ttokens)
        overlap = len(qs & ts) / max(1, len(qs | ts))
        containment = len(qs & ts) / max(1, len(qs))
        qids = set(__import__("re").findall(r"[A-Za-z]+[-_.:/]?\d+(?:\.\d+)*", query))
        tids = set(__import__("re").findall(r"[A-Za-z]+[-_.:/]?\d+(?:\.\d+)*", text))
        identifier = float(bool(qids and qids & tids))
        qv = np.asarray(self.encoder.encode(query), dtype=np.float32)
        tv = np.asarray(self.encoder.encode(text), dtype=np.float32)
        semantic = max(0.0, cosine(qv, tv))
        neg_q = any(x in normalized_text(query) for x in ("not", "never", "不", "没"))
        neg_t = any(x in normalized_text(text) for x in ("not", "never", "不", "没"))
        polarity = float(neg_q == neg_t)
        length_fit = math.exp(-abs(len(qtokens) - len(ttokens)) / max(4, len(qtokens) + len(ttokens)))
        bigrams_q = set(zip(qtokens, qtokens[1:]))
        bigrams_t = set(zip(ttokens, ttokens[1:]))
        bigram = len(bigrams_q & bigrams_t) / max(1, len(bigrams_q))
        return np.asarray(
            [semantic, overlap, containment, identifier, polarity, length_fit, bigram],
            dtype=np.float32,
        )

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        hidden = np.tanh(features @ self.w1 + self.b1)
        logit = float(hidden @ self.w2 + self.b2)
        return hidden, 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self._forward(self._features(query, item))[1] for item in texts]

    def learn_pairwise(
        self, query: str, useful_text: str, rejected_texts: Iterable[str],
        learning_rate: float = 0.025,
    ) -> float:
        losses = []
        positive = self._features(query, useful_text)
        for negative_text in list(rejected_texts)[:12]:
            negative = self._features(query, negative_text)
            hp, sp = self._forward(positive)
            hn, sn = self._forward(negative)
            margin = sp - sn
            loss = math.log1p(math.exp(-max(-20.0, min(20.0, margin))))
            gradient = -1.0 / (1.0 + math.exp(max(-20.0, min(20.0, margin))))
            old_w2 = self.w2.copy()
            self.w2 -= learning_rate * gradient * (hp - hn)
            dhp = gradient * old_w2 * (1.0 - hp * hp)
            dhn = -gradient * old_w2 * (1.0 - hn * hn)
            self.w1 -= learning_rate * (
                np.outer(positive, dhp) + np.outer(negative, dhn)
            )
            self.b1 -= learning_rate * (dhp + dhn)
            self.w1 = np.clip(self.w1, -2.0, 2.0)
            self.w2 = np.clip(self.w2, -2.0, 2.0)
            self.updates += 1
            losses.append(loss)
        return sum(losses) / max(1, len(losses))

    def state(self) -> dict[str, object]:
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": float(self.b2), "updates": self.updates,
        }

    def load_state(self, state: dict[str, object]) -> None:
        try:
            w1, b1 = np.asarray(state["w1"], dtype=np.float32), np.asarray(state["b1"], dtype=np.float32)
            w2 = np.asarray(state["w2"], dtype=np.float32)
            if w1.shape == (7, 10) and b1.shape == (10,) and w2.shape == (10,):
                self.w1, self.b1, self.w2 = w1, b1, w2
                self.b2 = np.float32(state.get("b2", 0.0))
                self.updates = int(state.get("updates", 0))
        except (KeyError, TypeError, ValueError):
            return


class IdentityGravitySystem(TemporalEntitySystem):
    """Entity resolution that refuses close calls instead of false-merging."""

    def __init__(
        self, encoder, merge_similarity: float = 0.92, promotion_count: int = 3,
        identity_threshold: float = 0.86, identity_margin: float = 0.08,
    ) -> None:
        super().__init__(encoder, merge_similarity, promotion_count)
        self.identity_threshold = identity_threshold
        self.identity_margin = identity_margin
        self.ambiguities: list[dict[str, object]] = []

    @staticmethod
    def _trigrams(value: str) -> set[str]:
        clean = f"  {normalized_text(value)}  "
        return {clean[index:index + 3] for index in range(max(1, len(clean) - 2))}

    def canonicalize(
        self, name: str, entity_type: str = "unknown", confidence: float = 0.5,
        observed_at: datetime | None = None,
    ):
        key = self._key(name)
        exact = self.alias_index.get(key)
        if exact is not None:
            return super().canonicalize(name, entity_type, confidence, observed_at)
        if not self.entities:
            return super().canonicalize(name, entity_type, confidence, observed_at)
        vector = np.asarray(self.encoder.encode(name), dtype=np.float32)
        source_tri = self._trigrams(name)
        scored = []
        for entity_id, entity in self.entities.items():
            semantic = max(0.0, cosine(vector, self.entity_vectors[entity_id]))
            target_tri = self._trigrams(entity.canonical_name)
            lexical = len(source_tri & target_tri) / max(1, len(source_tri | target_tri))
            compatible = float(
                entity_type in {"", "unknown"}
                or entity.entity_type in {"", "unknown", entity_type}
            )
            alias_bonus = max(
                (normalized_text(name) in normalized_text(alias) or normalized_text(alias) in normalized_text(name))
                for alias in entity.aliases
            )
            score = 0.58 * semantic + 0.22 * lexical + 0.14 * compatible + 0.06 * float(alias_bonus)
            scored.append((score, entity_id))
        scored.sort(reverse=True)
        best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best[0] >= self.identity_threshold and best[0] - second >= self.identity_margin:
            self.alias_index[key] = best[1]
            return super().canonicalize(name, entity_type, confidence, observed_at)
        if best[0] >= self.identity_threshold - 0.05:
            self.ambiguities.append({
                "name": name, "entity_type": entity_type,
                "candidates": [(item, round(score, 6)) for score, item in scored[:3]],
                "reason": "identity_margin_quarantine",
            })
        # Force a fresh entity by temporarily raising the base merge threshold.
        original = self.merge_similarity
        self.merge_similarity = 1.01
        try:
            return super().canonicalize(name, entity_type, confidence, observed_at)
        finally:
            self.merge_similarity = original


@dataclass
class TelemetrySpan:
    operation: str
    started_at: str
    duration_ms: float
    success: bool
    attributes: dict[str, object]


class CosmicTelemetry:
    """Bounded local telemetry with Prometheus and exporter-compatible output."""

    def __init__(self, span_limit: int = 512, exporter: Callable[[TelemetrySpan], None] | None = None) -> None:
        self.counters: Counter = Counter()
        self.durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1024))
        self.spans: deque[TelemetrySpan] = deque(maxlen=max(16, span_limit))
        self.exporter = exporter
        self._lock = threading.RLock()

    def record(self, operation: str, started: float, success: bool, **attributes) -> TelemetrySpan:
        span = TelemetrySpan(
            operation, utcnow().isoformat(), (time.perf_counter() - started) * 1000.0,
            success, attributes,
        )
        with self._lock:
            self.counters[f"{operation}_total"] += 1
            self.counters[f"{operation}_{'success' if success else 'failure'}"] += 1
            self.durations[operation].append(span.duration_ms)
            self.spans.append(span)
        if self.exporter:
            self.exporter(span)
        return span

    def prometheus(self) -> str:
        lines = []
        for key, value in sorted(self.counters.items()):
            lines.append(f"umd_{key} {int(value)}")
        for operation, values in sorted(self.durations.items()):
            lines.append(f'umd_operation_duration_ms_sum{{operation="{operation}"}} {sum(values):.6f}')
            lines.append(f'umd_operation_duration_ms_count{{operation="{operation}"}} {len(values)}')
        return "\n".join(lines) + "\n"


class ModelRuntimeAdapter(Protocol):
    def fingerprint(self) -> str: ...
    def apply_prompt(self, payload: bytes) -> object: ...
    def apply_kv_cache(self, payload: bytes) -> object: ...
    def apply_adapter(self, payload: bytes) -> object: ...


class MultimodalEncoder(Protocol):
    def encode_payload(self, payload: bytes, metadata: dict[str, object]) -> tuple[np.ndarray, str]: ...


@dataclass
class MultimodalMeteor:
    modality: Literal["image", "audio", "video", "document"]
    payload: bytes
    metadata: dict[str, object] = field(default_factory=dict)
    caption: str | None = None


@dataclass
class GravityPacket:
    packet_id: str
    tenant_id: str
    operation: str
    expected_revision: int
    payload: dict[str, object]
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    signature: str = ""

    def signing_bytes(self) -> bytes:
        body = {
            "packet_id": self.packet_id, "tenant_id": self.tenant_id,
            "operation": self.operation, "expected_revision": self.expected_revision,
            "payload": self.payload, "created_at": self.created_at,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def sign(self, key: bytes) -> "GravityPacket":
        self.signature = hmac.new(key, self.signing_bytes(), hashlib.sha256).hexdigest()
        return self

    def verify(self, key: bytes) -> bool:
        expected = hmac.new(key, self.signing_bytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)


class ReplicaTransport(Protocol):
    def apply(self, endpoint: str, packet: GravityPacket) -> dict[str, object]: ...
    def status(self, endpoint: str, tenant_id: str) -> dict[str, object]: ...


class HTTPReplicaTransport:
    """Concrete JSON/HTTP transport; TLS policy belongs to its supplied URLs/context."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _packet(packet: GravityPacket) -> dict[str, object]:
        return {**json.loads(packet.signing_bytes()), "signature": packet.signature}

    def apply(self, endpoint: str, packet: GravityPacket) -> dict[str, object]:
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/replica/apply",
            json.dumps(self._packet(packet)).encode("utf-8"),
            {"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def status(self, endpoint: str, tenant_id: str) -> dict[str, object]:
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/v1/replica/status?tenant=" + tenant_id,
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class NetworkConstellation:
    """Signed, idempotent quorum replication over pluggable network transports."""

    def __init__(
        self, endpoints: dict[str, str], transport: ReplicaTransport,
        signing_key: bytes, replication_factor: int = 3,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one replica endpoint is required")
        self.endpoints = dict(endpoints)
        self.transport = transport
        self.signing_key = signing_key
        self.replication_factor = max(1, min(replication_factor, len(endpoints)))

    @staticmethod
    def _force(tenant_id: str, replica_id: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{tenant_id}:{replica_id}".encode()).digest()[:8], "big")

    def placement(self, tenant_id: str) -> list[str]:
        return sorted(self.endpoints, key=lambda item: self._force(tenant_id, item), reverse=True)[
            : self.replication_factor
        ]

    @property
    def quorum(self) -> int:
        return self.replication_factor // 2 + 1

    def replicate(self, packet: GravityPacket) -> dict[str, object]:
        packet.sign(self.signing_key)
        acknowledgements, failures = {}, {}
        placed = self.placement(packet.tenant_id)
        # Parallel rays avoid adding every regional round-trip to write latency.
        with ThreadPoolExecutor(max_workers=len(placed), thread_name_prefix="umd-replica") as pool:
            futures = {
                pool.submit(self.transport.apply, self.endpoints[replica_id], packet): replica_id
                for replica_id in placed
            }
            for future in as_completed(futures):
                replica_id = futures[future]
                try:
                    acknowledgements[replica_id] = future.result()
                except Exception as error:
                    failures[replica_id] = type(error).__name__
        if len(acknowledgements) < self.quorum:
            raise RuntimeError(f"constellation quorum failed: {len(acknowledgements)}/{self.quorum}")
        return {"acknowledgements": acknowledgements, "failures": failures, "quorum": self.quorum}

    def read_repair_plan(self, tenant_id: str) -> dict[str, object]:
        states = {
            replica_id: self.transport.status(self.endpoints[replica_id], tenant_id)
            for replica_id in self.placement(tenant_id)
        }
        source = max(states, key=lambda item: int(states[item].get("revision", 0)))
        revision = int(states[source].get("revision", 0))
        return {
            "source": source, "revision": revision,
            "repair": [item for item, state in states.items() if int(state.get("revision", 0)) < revision],
        }


class CosmicCheckpointManager:
    """Online SQLite checkpoints and non-overwriting recovery drills.

    A checkpoint is consistent with WAL writers because SQLite's backup API is
    used. Recovery always targets a new path, making drills recoverable and
    preventing an accidental overwrite of the live universe.
    """

    @staticmethod
    def create(database, destination: str | Path) -> dict[str, object]:
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        replica = sqlite3.connect(target)
        try:
            with replica:
                database.connection.backup(replica)
        finally:
            replica.close()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows = database.connection.execute(
            "SELECT tenant_id, MAX(revision) revision FROM event_log GROUP BY tenant_id"
        ).fetchall()
        return {
            "path": str(target), "sha256": digest,
            "tenant_revisions": {row["tenant_id"]: int(row["revision"]) for row in rows},
            "created_at": utcnow().isoformat(),
        }

    @staticmethod
    def verify(path: str | Path, expected_sha256: str | None = None) -> dict[str, object]:
        source = Path(path).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        connection = sqlite3.connect(source)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
        finally:
            connection.close()
        return {
            "integrity": integrity, "events": int(events), "sha256": digest,
            "digest_matches": expected_sha256 is None or hmac.compare_digest(digest, expected_sha256),
        }

    @staticmethod
    def recover_to(checkpoint: str | Path, destination: str | Path) -> dict[str, object]:
        source, target = Path(checkpoint).resolve(), Path(destination).resolve()
        if target.exists():
            raise FileExistsError(f"recovery target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        origin = sqlite3.connect(source)
        restored = sqlite3.connect(target)
        try:
            with restored:
                origin.backup(restored)
        finally:
            origin.close()
            restored.close()
        return CosmicCheckpointManager.verify(target)


class CapabilityToken:
    """Short-lived HMAC capability tokens for the built-in gateway.

    Production OIDC can be supplied through ``claims_verifier`` on the gateway;
    this compact token format intentionally does not pretend to be an IdP.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("capability key must contain at least 32 bytes")
        self.key = key

    def issue(self, tenant_id: str, principal_id: str, ttl_seconds: int = 900) -> str:
        payload = json.dumps({
            "tenant": tenant_id, "principal": principal_id,
            "exp": int(time.time()) + max(1, ttl_seconds),
        }, sort_keys=True, separators=(",", ":")).encode()
        body = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = base64.urlsafe_b64encode(hmac.new(self.key, body, hashlib.sha256).digest()).rstrip(b"=")
        return (body + b"." + signature).decode("ascii")

    def verify(self, token: str) -> dict[str, object]:
        try:
            body, supplied = token.encode("ascii").split(b".", 1)
            expected = base64.urlsafe_b64encode(hmac.new(self.key, body, hashlib.sha256).digest()).rstrip(b"=")
            if not hmac.compare_digest(expected, supplied):
                raise AuthorizationError("invalid capability signature")
            payload = json.loads(base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4)))
            if int(payload["exp"]) < int(time.time()):
                raise AuthorizationError("expired capability")
            return payload
        except AuthorizationError:
            raise
        except Exception as error:
            raise AuthorizationError("invalid capability token") from error


class UMD38CosmicMemory(UMD37PlanetaryMemory):
    """UMD 3.8 compatibility layer with atomic write capsules and adapters."""

    def __init__(self, durable, *, config: UMD38Config | None = None, reranker=None,
                 reflection_proposer=None, telemetry: CosmicTelemetry | None = None) -> None:
        self.cosmic_config = config or UMD38Config()
        self.neural_tide = NeuralTidalReranker(durable.engine.encoder)
        self.telemetry = telemetry or CosmicTelemetry(self.cosmic_config.telemetry_span_limit)
        self.multimodal_encoders: dict[str, MultimodalEncoder] = {}
        self._external_reranker = reranker
        super().__init__(
            durable, config=self.cosmic_config,
            reranker=reranker or self.neural_tide.score,
            reflection_proposer=reflection_proposer,
        )
        old = self.entity_system
        identity = IdentityGravitySystem(
            durable.engine.encoder, self.cosmic_config.entity_merge_similarity,
            self.cosmic_config.ontology_promotion_count,
            self.cosmic_config.identity_threshold, self.cosmic_config.identity_margin,
        )
        identity.entities, identity.alias_index = old.entities, old.alias_index
        identity.entity_vectors, identity.orbits = old.entity_vectors, old.orbits
        identity.outgoing, identity.ontology = old.outgoing, old.ontology
        self.entity_system = identity
        self.index = IncrementalOrbitalIndex(
            durable.engine.config.embedding_dimensions,
            self.cosmic_config.ann_tables, self.cosmic_config.ann_bits,
            self.cosmic_config.orbital_degree, self.cosmic_config.orbital_search_width,
        )
        self.index.rebuild(durable.engine.nodes, durable.revision)
        self._load_cosmic_extensions()

    def _state_dict(self) -> dict[str, object]:
        state = super()._state_dict()
        neural = getattr(self, "neural_tide", None)
        state["umd38"] = {
            "neural_tide": neural.state() if neural else {},
            "identity_ambiguities": list(getattr(self.entity_system, "ambiguities", []))[-256:],
        }
        return state

    def _load_cosmic_extensions(self) -> None:
        row = self.durable.database.connection.execute(
            "SELECT envelope FROM planetary_state WHERE tenant_id=?", (self.durable.tenant_id,)
        ).fetchone()
        if row is None:
            return
        raw = self.durable.database.cipher(self.durable.tenant_id).decrypt(
            bytes(row["envelope"]), "planetary-state"
        )
        extension = json.loads(raw.decode("utf-8")).get("umd38", {})
        if isinstance(extension.get("neural_tide"), dict):
            self.neural_tide.load_state(extension["neural_tide"])
        if isinstance(self.entity_system, IdentityGravitySystem):
            self.entity_system.ambiguities = list(extension.get("identity_ambiguities", []))

    def _restore_after_failure(self) -> None:
        revision, state = self.durable.database.load_state(self.durable.tenant_id)
        self.durable.revision = revision
        self.durable.engine = self.durable._restore(state, self.durable.engine.config)
        self.entity_system = IdentityGravitySystem(
            self.durable.engine.encoder, self.cosmic_config.entity_merge_similarity,
            self.cosmic_config.ontology_promotion_count,
            self.cosmic_config.identity_threshold, self.cosmic_config.identity_margin,
        )
        if not self._load_planetary_state():
            self._rebuild_planetary_state()
        self.index.rebuild(self.durable.engine.nodes, self.durable.revision)
        self._load_cosmic_extensions()

    def write(self, text: str, *, tags: Iterable[str] = (), classification: str = "internal",
              expires_at: datetime | None = None, **kwargs):
        """Commit memory, entity orbits, planetary state, and audit once."""
        started = time.perf_counter()
        self.durable.database.require(self.durable.tenant_id, self.principal_id, "writer")
        scope = kwargs.get("scope")
        tags = tuple(tags)
        if classification not in CLASSIFICATION_LEVEL:
            raise ValueError("unknown classification")
        if not self.policy.allows(
            self.principal_id, "write", scope=scope, tags=tags, classification=classification
        ):
            raise AuthorizationError("ABAC policy denied cosmic write")
        metadata = dict(kwargs.pop("extraction_metadata", {}) or {})
        metadata["umd37_policy"] = {
            "tags": sorted(set(tags)), "classification": classification,
            "expires_at": self._dt(expires_at), "owner": self.principal_id,
        }
        vector_override = kwargs.pop("_vector_override", None)
        try:
            with self.durable._lock:
                result = self.durable.engine.write(text, extraction_metadata=metadata, **kwargs)
                node = self.durable.engine.nodes[result.memory_id]
                if vector_override is not None and node.state in {"stable", "provisional"}:
                    vector = np.asarray(vector_override, dtype=np.float32)
                    expected = (self.durable.engine.config.embedding_dimensions,)
                    if vector.shape != expected:
                        raise ValueError(f"vector override returned {vector.shape}, expected {expected}")
                    norm = float(np.linalg.norm(vector))
                    node.vector = vector / norm if norm else vector
                    # The text vector was used during initial attachment. Recast
                    # the local star and relation field before the capsule seals.
                    if node.star_id:
                        self.durable.engine._rebuild_star(node.star_id)
                    for other_id in list(self.durable.engine.relations.get(node.id, {})):
                        self.durable.engine.relations[other_id].pop(node.id, None)
                    self.durable.engine.relations.pop(node.id, None)
                    for entity in node.entities:
                        self.durable.engine.entity_index[entity].discard(node.id)
                    self.durable.engine._add_relations(node)
                self.entity_system.ingest(node)
                self._writes_since_reflection += 1
                self.durable._commit(
                    "cosmic.write", result.memory_id,
                    {"memory_id": result.memory_id, "scope": scope, "classification": classification},
                    planetary_state_payload=self._planetary_state_bytes(),
                )
        except Exception as error:
            self._restore_after_failure()
            self.telemetry.record("write", started, False, error=type(error).__name__)
            raise
        # Derived index and diagnostics must never turn a committed memory into
        # an apparent failed write. A stale index self-heals on the next read.
        try:
            self.index.synchronize(self.durable.engine.nodes, self.durable.revision)
        except Exception:
            self.index.revision = -1
        try:
            self.durable.database.record_validation(
                self.durable.tenant_id, self.principal_id, "cosmic.write",
                (time.perf_counter() - started) * 1000.0, True,
                source=self.durable.validation_source,
            )
        except Exception:
            pass
        self.telemetry.record("write", started, True, state=result.state)
        if self.config.reflection_interval and self._writes_since_reflection >= self.config.reflection_interval:
            self.reflect()
        return result

    def retrieve(self, query: str, **kwargs):
        started = time.perf_counter()
        try:
            self.index.synchronize(self.durable.engine.nodes, self.durable.revision)
            result = super().retrieve(query, **kwargs)
            self.telemetry.record("retrieve", started, True, hits=len(result))
            return result
        except Exception as error:
            self.telemetry.record("retrieve", started, False, error=type(error).__name__)
            raise

    def retrieval_feedback(self, query: str, useful_memory_id: str,
                           rejected_memory_ids: Iterable[str] = (), **kwargs):
        rejected_ids = list(rejected_memory_ids)
        useful = self.durable.get_memory(useful_memory_id)
        rejected = [self.durable.get_memory(item).text for item in rejected_ids]
        loss = self.neural_tide.learn_pairwise(query, useful.text, rejected)
        weights = super().retrieval_feedback(
            query, useful_memory_id, rejected_ids, **kwargs
        )
        # Persist the updated tidal network in the same feedback event lineage.
        self._commit_planetary(
            "tidal.feedback", useful_memory_id,
            {"useful_memory_id": useful_memory_id, "loss": loss, "pairs": len(rejected)},
        )
        return {**weights, "tidal_pairwise_loss": loss}

    def register_multimodal_encoder(self, modality: str, encoder: MultimodalEncoder) -> None:
        if modality not in {"image", "audio", "video", "document"}:
            raise ValueError("unsupported modality")
        self.multimodal_encoders[modality] = encoder

    def ingest_multimodal(self, meteor: MultimodalMeteor, **write_kwargs):
        if len(meteor.payload) > self.cosmic_config.multimodal_max_bytes:
            raise ValueError("multimodal meteor exceeds bounded payload size")
        encoder = self.multimodal_encoders.get(meteor.modality)
        if encoder is None:
            raise ValueError(f"no semantic encoder registered for {meteor.modality}")
        vector, generated_caption = encoder.encode_payload(meteor.payload, meteor.metadata)
        vector = np.asarray(vector, dtype=np.float32)
        expected = (self.durable.engine.config.embedding_dimensions,)
        if vector.shape != expected:
            raise ValueError(f"multimodal encoder returned {vector.shape}, expected {expected}")
        digest = hashlib.sha256(meteor.payload).hexdigest()
        caption = meteor.caption or generated_caption
        result = self.write(
            f"[{meteor.modality}] {caption}",
            extraction_metadata={
                "multimodal": {"modality": meteor.modality, "sha256": digest, "metadata": meteor.metadata}
            },
            _vector_override=vector,
            **write_kwargs,
        )
        return result

    def activate_model_memory(self, item_id: str, runtime: ModelRuntimeAdapter) -> object:
        item = self.model_memories[item_id]
        actual = runtime.fingerprint()
        if not hmac.compare_digest(actual, item.model_fingerprint):
            raise ValueError("model runtime fingerprint does not match artifact orbit")
        handlers = {
            "prompt": runtime.apply_prompt,
            "kv_cache": runtime.apply_kv_cache,
            "adapter": runtime.apply_adapter,
        }
        result = handlers[item.kind](item.payload)
        item.uses += 1
        self._commit_planetary(
            "model_memory.activate", item.id,
            {"id": item.id, "kind": item.kind, "model_fingerprint": actual},
        )
        return result

    def entity_communities(self, iterations: int = 12) -> list[dict[str, object]]:
        """Discover bounded relation communities by deterministic gravity voting."""
        labels = {entity_id: entity_id for entity_id in self.entity_system.entities}
        edges: dict[str, dict[str, float]] = defaultdict(dict)
        for orbit in self.entity_system.orbits.values():
            if not orbit.object_entity_id or orbit.invalidated_at is not None:
                continue
            edges[orbit.subject_id][orbit.object_entity_id] = max(
                edges[orbit.subject_id].get(orbit.object_entity_id, 0.0), orbit.confidence
            )
            edges[orbit.object_entity_id][orbit.subject_id] = max(
                edges[orbit.object_entity_id].get(orbit.subject_id, 0.0), orbit.confidence
            )
        for _ in range(max(1, min(iterations, 32))):
            changed = False
            for entity_id in sorted(labels):
                votes: Counter = Counter()
                for neighbor, weight in edges.get(entity_id, {}).items():
                    votes[labels[neighbor]] += weight
                if votes:
                    winner = min(
                        (label for label, score in votes.items() if score == max(votes.values())),
                        default=labels[entity_id],
                    )
                    if winner != labels[entity_id]:
                        labels[entity_id] = winner
                        changed = True
            if not changed:
                break
        groups: dict[str, list[str]] = defaultdict(list)
        for entity_id, label in labels.items():
            groups[label].append(entity_id)
        output = []
        for label, entity_ids in groups.items():
            evidence = sorted({
                orbit.evidence_memory_id for orbit in self.entity_system.orbits.values()
                if orbit.subject_id in entity_ids and orbit.object_entity_id in entity_ids
            })
            output.append({
                "community_id": "community_" + hashlib.sha256("|".join(sorted(entity_ids)).encode()).hexdigest()[:12],
                "entity_ids": sorted(entity_ids), "evidence_ids": evidence,
                "mass": len(entity_ids) + 0.2 * len(evidence),
            })
        return sorted(output, key=lambda item: (-item["mass"], item["community_id"]))

    def assemble_context(
        self, query: str, *, agent_id: str | None = None,
        budget_chars: int = 6000, top_k: int = 12,
    ) -> dict[str, object]:
        """Solve a bounded context packing problem across memory species."""
        candidates: list[tuple[float, str, str, str]] = []
        for hit in self.retrieve(query, top_k=top_k, budget_chars=budget_chars * 2):
            candidates.append((hit.force, "declarative", hit.memory_id, hit.text))
        if agent_id:
            for label, block in self.blocks.items():
                if agent_id == block.owner_agent or agent_id in block.readers:
                    candidates.append((1.15, "block", label, f"{label}: {block.value}"))
        for score, procedure in self.select_procedure(query, top_k=3):
            text = f"Procedure {procedure.name}: " + " -> ".join(procedure.steps)
            candidates.append((score, "procedure", procedure.id, text))
        # Density approximates the Lagrangian value of each character. Blocks
        # receive a fixed structural premium but still consume the same budget.
        ranked = sorted(
            candidates, key=lambda item: (item[0] / max(32, len(item[3])), item[0]), reverse=True
        )
        selected, used = [], 0
        for score, kind, item_id, text in ranked:
            if selected and used + len(text) > budget_chars:
                continue
            selected.append({"kind": kind, "id": item_id, "score": score, "text": text})
            used += len(text)
        return {
            "query": query, "items": selected, "used_chars": used,
            "budget_chars": budget_chars, "utilization": used / max(1, budget_chars),
        }

    def snapshot(self) -> dict[str, object]:
        base = super().snapshot()
        base.update({
            "version": "UMD 3.8",
            "formula": "Omega=Capsule[planet+orbit+audit]; F*=F+T_theta(q,m)+I_margin(e); ANN=incremental-orbits",
            "index_full_rebuilds": self.index.full_rebuilds,
            "index_incremental_upserts": self.index.incremental_upserts,
            "identity_ambiguities": len(getattr(self.entity_system, "ambiguities", [])),
            "tidal_training_updates": self.neural_tide.updates,
            "telemetry_spans": len(self.telemetry.spans),
        })
        return base


class UMDGateway:
    """Small REST/MCP boundary; callers provide tenant/principal memory lookup."""

    def __init__(
        self, memory_lookup: Callable[[str, str], UMD38CosmicMemory],
        token_authority: CapabilityToken | None = None,
        claims_verifier: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self.memory_lookup = memory_lookup
        self.token_authority = token_authority
        self.claims_verifier = claims_verifier

    def _claims(self, headers: dict[str, str]) -> dict[str, object]:
        raw = headers.get("authorization", headers.get("Authorization", ""))
        if not raw.startswith("Bearer "):
            raise AuthorizationError("bearer token required")
        token = raw[7:]
        if self.claims_verifier:
            return self.claims_verifier(token)
        if self.token_authority:
            return self.token_authority.verify(token)
        raise AuthorizationError("no token verifier configured")

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes = b"") -> tuple[int, object, str]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok", "version": "UMD 3.8"}, "application/json"
        claims = self._claims(headers)
        memory = self.memory_lookup(str(claims["tenant"]), str(claims["principal"]))
        if method == "GET" and path == "/metrics":
            return 200, memory.telemetry.prometheus(), "text/plain"
        payload = json.loads(body.decode("utf-8") or "{}")
        if method == "POST" and path == "/v1/memories":
            result = memory.write(payload["text"], scope=payload.get("scope"),
                                  tags=payload.get("tags", ()), classification=payload.get("classification", "internal"),
                                  explicit_importance=float(payload.get("importance", 0.0)))
            return 201, {"memory_id": result.memory_id, "state": result.state}, "application/json"
        if method == "POST" and path == "/v1/retrieve":
            hits = memory.retrieve(payload["query"], top_k=int(payload.get("top_k", 5)), scope=payload.get("scope"))
            return 200, [{"memory_id": item.memory_id, "text": item.text, "force": item.force,
                          "explanation": item.explanation} for item in hits], "application/json"
        return 404, {"error": "route_not_found"}, "application/json"

    def mcp(self, request: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        method = request.get("method")
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"tools": [
                {"name": "umd_write", "description": "Write a memory into a UMD cosmic orbit"},
                {"name": "umd_retrieve", "description": "Retrieve memories by planetary force"},
            ]}}
        if method != "tools/call":
            return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32601, "message": "method not found"}}
        params = request.get("params", {})
        name, arguments = params.get("name"), params.get("arguments", {})
        route = "/v1/memories" if name == "umd_write" else "/v1/retrieve" if name == "umd_retrieve" else None
        if route is None:
            return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32602, "message": "unknown tool"}}
        status, result, _ = self.handle("POST", route, headers, json.dumps(arguments).encode())
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"status": status, "content": result}}

    def make_server(
        self, host: str = "127.0.0.1", port: int = 0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> ThreadingHTTPServer:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def _dispatch(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                try:
                    status, payload, content_type = gateway.handle(
                        self.command, self.path, dict(self.headers), body
                    )
                except AuthorizationError as error:
                    status, payload, content_type = 401, {"error": str(error)}, "application/json"
                except (ValueError, KeyError) as error:
                    status, payload, content_type = 400, {"error": str(error)}, "application/json"
                raw = payload.encode() if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _dispatch
            do_POST = _dispatch

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer((host, port), Handler)
        if ssl_context is not None:
            server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        return server
