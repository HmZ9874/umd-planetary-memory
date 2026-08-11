"""UMD 3.7 planetary interaction memory algorithm.

Queries are modeled as comets crossing a bounded planetary system. Candidate
memories feel a normalized force composed of semantic gravity, BM25 resonance,
entity resonance, temporal phase alignment, graph flux, authority, learned
utility, and contradiction repulsion. The same layer adds temporal entity
orbits, evidence-conserving reflection, ABAC, shared agent blocks, and
procedural memory.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Literal

import numpy as np

try:
    from .umd35_core import (
        MemoryNode,
        RetrievedMemory,
        cosine,
        normalized_text,
        normalized_tokens,
        utcnow,
    )
    from .umd36_persistent import (
        AuthorizationError,
        UMD36TenantMemory,
        _from_json_bytes,
        _json_bytes,
    )
except ImportError:
    from umd35_core import (
        MemoryNode,
        RetrievedMemory,
        cosine,
        normalized_text,
        normalized_tokens,
        utcnow,
    )
    from umd36_persistent import (
        AuthorizationError,
        UMD36TenantMemory,
        _from_json_bytes,
        _json_bytes,
    )


CLASSIFICATION_LEVEL = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
TIME_CUES = {
    "when", "before", "after", "current", "latest", "previous", "history",
    "何时", "之前", "之后", "当前", "现在", "最新", "曾经", "历史",
}
GRAPH_CUES = {
    "why", "how", "related", "relationship", "depends", "through",
    "为什么", "如何", "关系", "关联", "依赖", "通过",
}
NEGATIONS = {"not", "never", "no", "不是", "没有", "不再", "从未"}


@dataclass
class PlanetaryWeights:
    semantic_gravity: float = 0.28
    bm25_resonance: float = 0.17
    entity_resonance: float = 0.13
    temporal_phase: float = 0.10
    graph_flux: float = 0.12
    authority: float = 0.08
    learned_utility: float = 0.07
    neural_rerank: float = 0.05
    contradiction_repulsion: float = 0.18

    def normalized_attraction(self) -> dict[str, float]:
        values = {
            "semantic_gravity": max(0.0, self.semantic_gravity),
            "bm25_resonance": max(0.0, self.bm25_resonance),
            "entity_resonance": max(0.0, self.entity_resonance),
            "temporal_phase": max(0.0, self.temporal_phase),
            "graph_flux": max(0.0, self.graph_flux),
            "authority": max(0.0, self.authority),
            "learned_utility": max(0.0, self.learned_utility),
            "neural_rerank": max(0.0, self.neural_rerank),
        }
        total = sum(values.values()) or 1.0
        return {key: value / total for key, value in values.items()}


@dataclass
class PlanetaryConfig:
    weights: PlanetaryWeights = field(default_factory=PlanetaryWeights)
    ann_tables: int = 5
    ann_bits: int = 12
    candidate_limit: int = 160
    graph_hops: int = 5
    graph_beam_width: int = 48
    rerank_top_n: int = 32
    reflection_interval: int = 0
    reflection_min_grounding: float = 0.72
    entity_merge_similarity: float = 0.92
    ontology_promotion_count: int = 3


@dataclass
class PlanetaryHit:
    memory_id: str
    text: str
    force: float
    orbit_radius: float
    potential_energy: float
    components: dict[str, float]
    explanation: dict[str, object]


@dataclass
class CanonicalEntity:
    id: str
    canonical_name: str
    entity_type: str = "unknown"
    aliases: set[str] = field(default_factory=set)
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=utcnow)
    last_seen: datetime = field(default_factory=utcnow)


@dataclass
class TemporalOrbit:
    id: str
    subject_id: str
    predicate: str
    object_value: str
    object_entity_id: str | None
    valid_from: datetime
    valid_to: datetime | None
    learned_at: datetime
    invalidated_at: datetime | None
    confidence: float
    evidence_memory_id: str


@dataclass
class OntologySignal:
    predicate: str
    observations: int = 0
    subject_types: Counter = field(default_factory=Counter)
    object_types: Counter = field(default_factory=Counter)
    promoted: bool = False


@dataclass
class AccessPolicy:
    principal_id: str
    actions: set[str] = field(default_factory=lambda: {"read", "write", "reflect", "procedure"})
    allowed_scopes: set[str] = field(default_factory=set)
    allowed_tags: set[str] = field(default_factory=set)
    max_classification: str = "restricted"


@dataclass
class MemoryBlock:
    label: str
    description: str
    value: str
    owner_agent: str
    readers: set[str] = field(default_factory=set)
    writers: set[str] = field(default_factory=set)
    read_only: bool = False
    max_characters: int = 5000
    version: int = 1


@dataclass
class ProcedureOrbit:
    id: str
    name: str
    trigger: str
    steps: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    successes: int = 0
    failures: int = 0
    last_used: datetime | None = None

    @property
    def confidence(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


@dataclass
class ModelMemoryOrbit:
    """Encrypted descriptor/payload for prompt, KV-cache, or adapter memory."""

    id: str
    kind: Literal["prompt", "kv_cache", "adapter"]
    model_fingerprint: str
    task_signature: str
    payload: bytes
    quality: float = 0.5
    uses: int = 0
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class ReflectionClaim:
    text: str
    evidence_ids: tuple[str, ...]
    confidence: float = 0.5


@dataclass
class ReflectionReport:
    accepted: tuple[ReflectionClaim, ...]
    rejected: tuple[dict[str, object], ...]
    memory_id: str | None
    grounding: float


class PlanetaryIndex:
    """Bounded BM25 + random-hyperplane ANN index over active memories."""

    def __init__(self, dimensions: int, tables: int = 5, bits: int = 12) -> None:
        self.dimensions = dimensions
        self.tables = max(1, tables)
        self.bits = max(4, min(bits, 20))
        rng = np.random.default_rng(370031)
        self.planes = rng.standard_normal(
            (self.tables, self.bits, dimensions), dtype=np.float32
        )
        self.buckets: list[dict[int, set[str]]] = [defaultdict(set) for _ in range(self.tables)]
        self.tokens: dict[str, list[str]] = {}
        self.term_frequencies: dict[str, Counter] = {}
        self.document_frequency: Counter = Counter()
        self.average_length = 1.0
        self.revision = -1

    def _signature(self, vector: np.ndarray, table: int) -> int:
        projections = self.planes[table] @ vector
        signature = 0
        for bit, positive in enumerate(projections >= 0):
            signature |= int(bool(positive)) << bit
        return signature

    def rebuild(self, nodes: dict[str, MemoryNode], revision: int) -> None:
        self.buckets = [defaultdict(set) for _ in range(self.tables)]
        self.tokens = {}
        self.term_frequencies = {}
        self.document_frequency = Counter()
        lengths = []
        for memory_id, node in nodes.items():
            if node.vector.shape != (self.dimensions,):
                continue
            terms = normalized_tokens(node.text)
            self.tokens[memory_id] = terms
            counts = Counter(terms)
            self.term_frequencies[memory_id] = counts
            self.document_frequency.update(counts.keys())
            lengths.append(len(terms))
            for table in range(self.tables):
                self.buckets[table][self._signature(node.vector, table)].add(memory_id)
        self.average_length = sum(lengths) / len(lengths) if lengths else 1.0
        self.revision = revision

    def ann_candidates(self, vector: np.ndarray) -> set[str]:
        result: set[str] = set()
        for table in range(self.tables):
            signature = self._signature(vector, table)
            result.update(self.buckets[table].get(signature, ()))
            # Probe one-bit neighboring orbits when the exact bucket is sparse.
            if len(result) < 32:
                for bit in range(self.bits):
                    result.update(self.buckets[table].get(signature ^ (1 << bit), ()))
        return result

    def bm25(self, query: str) -> dict[str, float]:
        terms = normalized_tokens(query)
        if not terms or not self.tokens:
            return {}
        count = len(self.tokens)
        k1, b = 1.45, 0.72
        scores: dict[str, float] = {}
        for memory_id, frequencies in self.term_frequencies.items():
            length = max(1, len(self.tokens[memory_id]))
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1.0) / (
                    tf + k1 * (1.0 - b + b * length / self.average_length)
                )
            if score:
                scores[memory_id] = score
        maximum = max(scores.values(), default=1.0)
        return {key: value / maximum for key, value in scores.items()}


class PlanetaryPolicyEngine:
    def __init__(self) -> None:
        self.policies: dict[str, AccessPolicy] = {}

    def set_policy(self, policy: AccessPolicy) -> None:
        if policy.max_classification not in CLASSIFICATION_LEVEL:
            raise ValueError("unknown classification")
        self.policies[policy.principal_id] = policy

    def allows(
        self,
        principal_id: str,
        action: str,
        *,
        scope: str | None,
        tags: Iterable[str] = (),
        classification: str = "internal",
    ) -> bool:
        policy = self.policies.get(principal_id)
        if policy is None:
            return True  # UMD36 RBAC remains the mandatory base layer.
        if action not in policy.actions:
            return False
        if policy.allowed_scopes and (scope or "") not in policy.allowed_scopes:
            return False
        tag_set = set(tags)
        if policy.allowed_tags and not tag_set.intersection(policy.allowed_tags):
            return False
        return CLASSIFICATION_LEVEL.get(classification, 3) <= CLASSIFICATION_LEVEL[policy.max_classification]


@dataclass
class ReplicaOrbit:
    id: str
    region: str
    healthy: bool = True
    revision: int = 0
    replication_lag_ms: float = 0.0
    latency_ms: float = 1.0


class RendezvousConstellation:
    """Deterministic sharding, quorum, failover, and revision reconciliation."""

    def __init__(self, replicas: Iterable[ReplicaOrbit], replication_factor: int = 3) -> None:
        self.replicas = {item.id: item for item in replicas}
        self.replication_factor = max(1, min(replication_factor, len(self.replicas)))
        if not self.replicas:
            raise ValueError("constellation needs at least one replica")

    @staticmethod
    def _hash_force(key: str, replica_id: str) -> float:
        raw = hashlib.sha256(f"{key}:{replica_id}".encode("utf-8")).digest()[:8]
        return (int.from_bytes(raw, "big") + 1) / (2**64 + 1)

    def placement(self, tenant_key: str) -> list[ReplicaOrbit]:
        return sorted(
            self.replicas.values(),
            key=lambda item: self._hash_force(tenant_key, item.id),
            reverse=True,
        )[: self.replication_factor]

    @property
    def quorum(self) -> int:
        return self.replication_factor // 2 + 1

    def read_orbit(self, tenant_key: str) -> ReplicaOrbit:
        candidates = [item for item in self.placement(tenant_key) if item.healthy]
        if not candidates:
            raise RuntimeError("no healthy replica orbit")
        # Stable read force favors newest revision, low lag, and low latency.
        return max(
            candidates,
            key=lambda item: (
                item.revision,
                math.exp(-item.replication_lag_ms / 500.0) / (1.0 + item.latency_ms / 100.0),
            ),
        )

    def write_quorum(
        self, tenant_key: str, operation: Callable[[ReplicaOrbit], int]
    ) -> dict[str, object]:
        acknowledgements = []
        failures = []
        for replica in self.placement(tenant_key):
            if not replica.healthy:
                failures.append(replica.id)
                continue
            try:
                replica.revision = int(operation(replica))
                acknowledgements.append(replica.id)
            except Exception:
                failures.append(replica.id)
        if len(acknowledgements) < self.quorum:
            raise RuntimeError(
                f"write quorum failed: {len(acknowledgements)}/{self.quorum} acknowledgements"
            )
        return {
            "acknowledgements": acknowledgements,
            "failures": failures,
            "quorum": self.quorum,
        }

    def reconciliation_plan(self, tenant_key: str) -> dict[str, object]:
        placed = self.placement(tenant_key)
        source = max(placed, key=lambda item: item.revision)
        return {
            "source": source.id,
            "revision": source.revision,
            "stale_replicas": [item.id for item in placed if item.revision < source.revision],
        }


class TemporalEntitySystem:
    """Canonical entities and bitemporal relation orbits with invalidation."""

    FUNCTIONAL_PREDICATES = {
        "database_version", "managed_by", "reports_to", "located_in", "status",
        "owner", "current_role", "uses", "采用", "负责人", "状态",
    }

    def __init__(self, encoder, merge_similarity: float = 0.92, promotion_count: int = 3) -> None:
        self.encoder = encoder
        self.merge_similarity = merge_similarity
        self.promotion_count = promotion_count
        self.entities: dict[str, CanonicalEntity] = {}
        self.alias_index: dict[str, str] = {}
        self.entity_vectors: dict[str, np.ndarray] = {}
        self.orbits: dict[str, TemporalOrbit] = {}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.ontology: dict[str, OntologySignal] = {}

    @staticmethod
    def _key(name: str) -> str:
        return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", name.casefold())

    @staticmethod
    def _id(prefix: str, value: str) -> str:
        return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"

    def canonicalize(
        self, name: str, entity_type: str = "unknown", confidence: float = 0.5,
        observed_at: datetime | None = None,
    ) -> CanonicalEntity:
        key = self._key(name)
        if not key:
            raise ValueError("entity name cannot be empty")
        entity_id = self.alias_index.get(key)
        vector = None
        if entity_id is None and self.entities:
            vector = self.encoder.encode(name)
            best = max(
                ((cosine(vector, item), candidate) for candidate, item in self.entity_vectors.items()),
                default=(0.0, None), key=lambda item: item[0],
            )
            if best[1] is not None and best[0] >= self.merge_similarity:
                entity_id = best[1]
        now = observed_at or utcnow()
        if entity_id is None:
            entity_id = self._id("entity", key)
            entity = CanonicalEntity(
                entity_id, name.strip(), entity_type or "unknown", {name.strip()},
                max(0.0, min(1.0, confidence)), now, now,
            )
            self.entities[entity_id] = entity
            self.entity_vectors[entity_id] = vector if vector is not None else self.encoder.encode(name)
        else:
            entity = self.entities[entity_id]
            entity.aliases.add(name.strip())
            entity.last_seen = max(entity.last_seen, now)
            entity.confidence = max(entity.confidence, max(0.0, min(1.0, confidence)))
            if entity.entity_type == "unknown" and entity_type:
                entity.entity_type = entity_type
        self.alias_index[key] = entity_id
        return entity

    def ingest(self, node: MemoryNode) -> None:
        for name in node.entities:
            self.canonicalize(
                name, node.entity_types.get(name, "unknown"), node.confidence, node.created_at
            )
        relations = list(node.entity_relations)
        for fact in node.facts:
            if fact.subject and fact.predicate and fact.object_value:
                relations.append(type("FactRelation", (), {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object_value": fact.object_value,
                    "object_is_entity": fact.object_value.casefold() in {e.casefold() for e in node.entities},
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "confidence": fact.confidence,
                })())
        for relation in relations:
            subject = self.canonicalize(
                relation.subject,
                node.entity_types.get(relation.subject.casefold(), "unknown"),
                relation.confidence,
                node.created_at,
            )
            object_entity = None
            if relation.object_is_entity:
                object_entity = self.canonicalize(
                    relation.object_value,
                    node.entity_types.get(relation.object_value.casefold(), "unknown"),
                    relation.confidence,
                    node.created_at,
                )
            start = relation.valid_from or node.valid_from or node.created_at
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if relation.predicate in self.FUNCTIONAL_PREDICATES:
                for orbit_id in list(self.outgoing.get(subject.id, ())):
                    old = self.orbits[orbit_id]
                    if (
                        old.predicate == relation.predicate
                        and old.object_value != relation.object_value
                        and old.valid_to is None
                    ):
                        old.valid_to = start
                        old.invalidated_at = node.created_at
            raw_id = (
                f"{subject.id}|{relation.predicate}|{relation.object_value}|"
                f"{start.isoformat()}|{node.id}"
            )
            orbit_id = self._id("orbit", raw_id)
            orbit = TemporalOrbit(
                orbit_id, subject.id, relation.predicate, relation.object_value,
                object_entity.id if object_entity else None, start, relation.valid_to,
                node.created_at, None, relation.confidence, node.id,
            )
            self.orbits[orbit_id] = orbit
            self.outgoing[subject.id].add(orbit_id)
            signal = self.ontology.setdefault(relation.predicate, OntologySignal(relation.predicate))
            signal.observations += 1
            signal.subject_types[subject.entity_type] += 1
            signal.object_types[object_entity.entity_type if object_entity else "literal"] += 1
            signal.promoted = signal.observations >= self.promotion_count

    def paths(
        self,
        start_name: str,
        target_name: str,
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        max_hops: int = 6,
        beam_width: int = 48,
    ) -> list[dict[str, object]]:
        start_id = self.alias_index.get(self._key(start_name))
        target_id = self.alias_index.get(self._key(target_name))
        if not start_id or not target_id:
            return []
        valid = valid_at or utcnow()
        known = known_at or utcnow()
        frontier = [(1.0, start_id, [], {start_id})]
        found = []
        for _ in range(max(1, min(max_hops, 8))):
            expanded = []
            for strength, entity_id, orbit_path, seen in frontier:
                for orbit_id in self.outgoing.get(entity_id, ()):
                    orbit = self.orbits[orbit_id]
                    if orbit.learned_at > known or orbit.valid_from > valid:
                        continue
                    if orbit.valid_to is not None and valid >= orbit.valid_to:
                        continue
                    if not orbit.object_entity_id or orbit.object_entity_id in seen:
                        continue
                    next_strength = strength * orbit.confidence * 0.78
                    next_path = orbit_path + [orbit_id]
                    if orbit.object_entity_id == target_id:
                        found.append((next_strength, next_path))
                    expanded.append(
                        (next_strength, orbit.object_entity_id, next_path, seen | {orbit.object_entity_id})
                    )
            frontier = sorted(expanded, key=lambda item: item[0], reverse=True)[:beam_width]
            if not frontier:
                break
        return [
            {
                "strength": strength,
                "hops": len(path),
                "orbits": [self.orbits[item] for item in path],
                "evidence_ids": [self.orbits[item].evidence_memory_id for item in path],
            }
            for strength, path in sorted(found, key=lambda item: item[0], reverse=True)
        ]


class UMD37PlanetaryMemory:
    """Original planetary interaction layer over durable UMD 3.6 storage."""

    def __init__(
        self,
        durable: UMD36TenantMemory,
        *,
        config: PlanetaryConfig | None = None,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        reflection_proposer: Callable[[list[MemoryNode]], list[ReflectionClaim]] | None = None,
    ) -> None:
        self.durable = durable
        self.config = config or PlanetaryConfig()
        self.reranker = reranker
        self.reflection_proposer = reflection_proposer
        self.index = PlanetaryIndex(
            durable.engine.config.embedding_dimensions,
            self.config.ann_tables,
            self.config.ann_bits,
        )
        self.entity_system = TemporalEntitySystem(
            durable.engine.encoder,
            self.config.entity_merge_similarity,
            self.config.ontology_promotion_count,
        )
        self.policy = PlanetaryPolicyEngine()
        self.blocks: dict[str, MemoryBlock] = {}
        self.procedures: dict[str, ProcedureOrbit] = {}
        self.model_memories: dict[str, ModelMemoryOrbit] = {}
        self._retrieval_traces: dict[str, dict[str, dict[str, float]]] = {}
        self._writes_since_reflection = 0
        self._create_tables()
        if not self._load_planetary_state():
            self._rebuild_planetary_state()

    @property
    def principal_id(self) -> str:
        return self.durable.principal_id

    def _create_tables(self) -> None:
        with self.durable.database.connection:
            self.durable.database.connection.execute(
                "CREATE TABLE IF NOT EXISTS planetary_state ("
                "tenant_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, envelope BLOB NOT NULL)"
            )

    @staticmethod
    def _dt(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    def _state_dict(self) -> dict[str, object]:
        return {
            "weights": vars(self.config.weights),
            "entities": [
                {
                    **{key: value for key, value in vars(item).items() if key not in {"aliases", "first_seen", "last_seen"}},
                    "aliases": sorted(item.aliases),
                    "first_seen": self._dt(item.first_seen),
                    "last_seen": self._dt(item.last_seen),
                }
                for item in self.entity_system.entities.values()
            ],
            "orbits": [
                {
                    **{key: value for key, value in vars(item).items() if key not in {"valid_from", "valid_to", "learned_at", "invalidated_at"}},
                    "valid_from": self._dt(item.valid_from),
                    "valid_to": self._dt(item.valid_to),
                    "learned_at": self._dt(item.learned_at),
                    "invalidated_at": self._dt(item.invalidated_at),
                }
                for item in self.entity_system.orbits.values()
            ],
            "ontology": [
                {
                    "predicate": item.predicate,
                    "observations": item.observations,
                    "subject_types": dict(item.subject_types),
                    "object_types": dict(item.object_types),
                    "promoted": item.promoted,
                }
                for item in self.entity_system.ontology.values()
            ],
            "policies": [
                {
                    "principal_id": item.principal_id,
                    "actions": sorted(item.actions),
                    "allowed_scopes": sorted(item.allowed_scopes),
                    "allowed_tags": sorted(item.allowed_tags),
                    "max_classification": item.max_classification,
                }
                for item in self.policy.policies.values()
            ],
            "blocks": [
                {
                    **{key: value for key, value in vars(item).items() if key not in {"readers", "writers"}},
                    "readers": sorted(item.readers),
                    "writers": sorted(item.writers),
                }
                for item in self.blocks.values()
            ],
            "procedures": [
                {
                    **{key: value for key, value in vars(item).items() if key != "last_used"},
                    "last_used": self._dt(item.last_used),
                }
                for item in self.procedures.values()
            ],
            "model_memories": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "model_fingerprint": item.model_fingerprint,
                    "task_signature": item.task_signature,
                    "payload": base64.b64encode(item.payload).decode("ascii"),
                    "quality": item.quality,
                    "uses": item.uses,
                    "created_at": self._dt(item.created_at),
                }
                for item in self.model_memories.values()
            ],
        }

    def _persist_planetary_state(self) -> None:
        raw = self._planetary_state_bytes()
        envelope = self.durable.database.cipher(self.durable.tenant_id).encrypt(raw, "planetary-state")
        with self.durable.database.connection:
            self.durable.database.connection.execute(
                "INSERT INTO planetary_state(tenant_id, revision, envelope) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET revision=excluded.revision, envelope=excluded.envelope",
                (self.durable.tenant_id, self.durable.revision, sqlite3.Binary(envelope)),
            )

    def _planetary_state_bytes(self) -> bytes:
        return json.dumps(
            self._state_dict(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

    def _commit_planetary(
        self, event_type: str, aggregate_id: str | None, payload: dict[str, object]
    ) -> None:
        self.durable._commit(
            event_type,
            aggregate_id,
            payload,
            planetary_state_payload=self._planetary_state_bytes(),
        )

    def _load_planetary_state(self) -> bool:
        row = self.durable.database.connection.execute(
            "SELECT revision, envelope FROM planetary_state WHERE tenant_id=?",
            (self.durable.tenant_id,),
        ).fetchone()
        if row is None:
            return False
        is_current = int(row["revision"]) == self.durable.revision
        raw = self.durable.database.cipher(self.durable.tenant_id).decrypt(
            bytes(row["envelope"]), "planetary-state"
        )
        state = json.loads(raw.decode("utf-8"))
        if isinstance(state.get("weights"), dict):
            for key, value in state["weights"].items():
                if hasattr(self.config.weights, key):
                    setattr(self.config.weights, key, float(value))
        for value in state.get("entities", []):
            entity = CanonicalEntity(
                value["id"], value["canonical_name"], value["entity_type"], set(value["aliases"]),
                value["confidence"], self._parse_dt(value["first_seen"]), self._parse_dt(value["last_seen"]),
            )
            self.entity_system.entities[entity.id] = entity
            self.entity_system.entity_vectors[entity.id] = self.durable.engine.encoder.encode(entity.canonical_name)
            for alias in entity.aliases:
                self.entity_system.alias_index[self.entity_system._key(alias)] = entity.id
        for value in state.get("orbits", []):
            orbit = TemporalOrbit(
                value["id"], value["subject_id"], value["predicate"], value["object_value"],
                value["object_entity_id"], self._parse_dt(value["valid_from"]),
                self._parse_dt(value["valid_to"]), self._parse_dt(value["learned_at"]),
                self._parse_dt(value["invalidated_at"]), value["confidence"], value["evidence_memory_id"],
            )
            self.entity_system.orbits[orbit.id] = orbit
            self.entity_system.outgoing[orbit.subject_id].add(orbit.id)
        for value in state.get("ontology", []):
            self.entity_system.ontology[value["predicate"]] = OntologySignal(
                value["predicate"], value["observations"], Counter(value["subject_types"]),
                Counter(value["object_types"]), value["promoted"],
            )
        for value in state.get("policies", []):
            self.policy.set_policy(AccessPolicy(
                value["principal_id"], set(value["actions"]), set(value["allowed_scopes"]),
                set(value["allowed_tags"]), value["max_classification"],
            ))
        for value in state.get("blocks", []):
            block = MemoryBlock(
                value["label"], value["description"], value["value"], value["owner_agent"],
                set(value["readers"]), set(value["writers"]), value["read_only"],
                value["max_characters"], value["version"],
            )
            self.blocks[block.label] = block
        for value in state.get("procedures", []):
            procedure = ProcedureOrbit(
                value["id"], value["name"], value["trigger"], tuple(value["steps"]),
                tuple(value["preconditions"]), value["successes"], value["failures"],
                self._parse_dt(value["last_used"]),
            )
            self.procedures[procedure.id] = procedure
        for value in state.get("model_memories", []):
            item = ModelMemoryOrbit(
                value["id"], value["kind"], value["model_fingerprint"],
                value["task_signature"], base64.b64decode(value["payload"]),
                value["quality"], value["uses"], self._parse_dt(value["created_at"]),
            )
            self.model_memories[item.id] = item
        return is_current

    def _rebuild_planetary_state(self) -> None:
        # Stream rows one at a time so historical node objects do not become a
        # resident RAM cache during recovery.
        self.entity_system = TemporalEntitySystem(
            self.durable.engine.encoder,
            self.config.entity_merge_similarity,
            self.config.ontology_promotion_count,
        )
        cursor = self.durable.database.connection.execute(
            "SELECT memory_id, envelope FROM memory_records WHERE tenant_id=? ORDER BY updated_at",
            (self.durable.tenant_id,),
        )
        cipher = self.durable.database.cipher(self.durable.tenant_id)
        for row in cursor:
            raw = cipher.decrypt(bytes(row["envelope"]), f"memory:{row['memory_id']}")
            node = _from_json_bytes(raw)
            if isinstance(node, MemoryNode):
                self.entity_system.ingest(node)
        self._persist_planetary_state()

    def rotate_master_key(self, new_master_key: bytes) -> dict[str, int]:
        """Rotate durable and planetary ciphertext in one database transaction."""
        return self.durable.database.rotate_master_key(new_master_key)

    def set_policy(self, policy: AccessPolicy) -> None:
        self.durable.database.require(self.durable.tenant_id, self.principal_id, "admin")
        self.policy.set_policy(policy)
        self._commit_planetary(
            "policy.set", policy.principal_id,
            {"principal_id": policy.principal_id, "actions": sorted(policy.actions)},
        )

    def _attributes(self, node: MemoryNode) -> dict[str, object]:
        value = node.extraction_metadata.get("umd37_policy", {})
        return value if isinstance(value, dict) else {}

    def _allowed_node(self, node: MemoryNode, action: str = "read") -> bool:
        attributes = self._attributes(node)
        return self.policy.allows(
            self.principal_id,
            action,
            scope=node.scope,
            tags=attributes.get("tags", ()),
            classification=str(attributes.get("classification", "internal")),
        )

    def write(
        self,
        text: str,
        *,
        tags: Iterable[str] = (),
        classification: str = "internal",
        expires_at: datetime | None = None,
        **kwargs,
    ):
        scope = kwargs.get("scope")
        if classification not in CLASSIFICATION_LEVEL:
            raise ValueError("unknown classification")
        if not self.policy.allows(
            self.principal_id, "write", scope=scope, tags=tags, classification=classification
        ):
            raise AuthorizationError("ABAC policy denied planetary write")
        metadata = dict(kwargs.pop("extraction_metadata", {}) or {})
        metadata["umd37_policy"] = {
            "tags": sorted(set(tags)),
            "classification": classification,
            "expires_at": self._dt(expires_at),
            "owner": self.principal_id,
        }
        result = self.durable.write(text, extraction_metadata=metadata, **kwargs)
        node = self.durable.engine.nodes.get(result.memory_id)
        if node is None:
            node = self.durable.get_memory(result.memory_id)
        self.entity_system.ingest(node)
        self._writes_since_reflection += 1
        self._commit_planetary(
            "planetary.ingest", result.memory_id,
            {"memory_id": result.memory_id},
        )
        if (
            self.config.reflection_interval
            and self._writes_since_reflection >= self.config.reflection_interval
        ):
            self.reflect()
        return result

    def _adaptive_weights(self, query: str) -> dict[str, float]:
        weights = vars(self.config.weights).copy()
        tokens = set(normalized_tokens(query))
        if tokens & TIME_CUES or re.search(r"\b(?:19|20)\d{2}\b", query):
            weights["temporal_phase"] *= 1.9
        if tokens & GRAPH_CUES:
            weights["graph_flux"] *= 1.8
            weights["entity_resonance"] *= 1.35
        if re.search(r"[A-Za-z]+[-_.:/]\d|\b\d+(?:\.\d+)+\b", query):
            weights["bm25_resonance"] *= 1.7
        positive = {key: max(0.0, value) for key, value in weights.items() if key != "contradiction_repulsion"}
        total = sum(positive.values()) or 1.0
        positive = {key: value / total for key, value in positive.items()}
        positive["contradiction_repulsion"] = max(0.0, weights["contradiction_repulsion"])
        return positive

    @staticmethod
    def _temporal_phase(node: MemoryNode, now: datetime, time_query: bool) -> float:
        if node.valid_from and node.valid_from > now:
            return 0.75 if time_query else 0.15
        if node.valid_to and now >= node.valid_to:
            return 0.70 if time_query else 0.05
        age_days = max(0.0, (now - node.updated_at).total_seconds() / 86400.0)
        return math.exp(-age_days / (365.0 if time_query else 120.0))

    @staticmethod
    def _contradiction(query: str, node: MemoryNode) -> float:
        query_negated = any(cue in normalized_text(query) for cue in NEGATIONS)
        node_negated = any(cue in normalized_text(node.text) for cue in NEGATIONS)
        polarity_mismatch = query_negated != node_negated
        query_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))
        node_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", node.text))
        value_mismatch = bool(query_values and node_values and query_values.isdisjoint(node_values))
        return min(1.0, 0.65 * polarity_mismatch + 0.55 * value_mismatch)

    def retrieve(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int = 5,
        now: datetime | None = None,
        budget_chars: int | None = None,
    ) -> list[PlanetaryHit]:
        self.durable.database.require(self.durable.tenant_id, self.principal_id, "reader")
        if self.index.revision != self.durable.revision:
            self.index.rebuild(self.durable.engine.nodes, self.durable.revision)
        vector = self.durable.engine.encoder.encode(query)
        bm25 = self.index.bm25(query)
        query_entities = self.durable.engine._entities(query, None)
        candidates = self.index.ann_candidates(vector)
        candidates.update(
            key for key, _ in sorted(bm25.items(), key=lambda item: item[1], reverse=True)[
                : self.config.candidate_limit
            ]
        )
        for entity in query_entities:
            candidates.update(self.durable.engine.entity_index.get(entity, ()))
        if len(self.durable.engine.nodes) <= self.config.candidate_limit:
            candidates.update(self.durable.engine.nodes)
        nodes = []
        for memory_id in candidates:
            node = self.durable.engine.nodes.get(memory_id)
            if (
                node is not None
                and node.state in {"stable", "provisional"}
                and (not scope or not node.scope or node.scope == scope)
                and self._allowed_node(node)
            ):
                nodes.append(node)
        reranks: dict[str, float] = {}
        semantic_order = sorted(nodes, key=lambda item: cosine(vector, item.vector), reverse=True)
        if self.reranker and semantic_order:
            selected = semantic_order[: self.config.rerank_top_n]
            values = self.reranker(query, [item.text for item in selected])
            if len(values) != len(selected):
                raise ValueError("reranker returned the wrong number of scores")
            reranks = {
                node.id: max(0.0, min(1.0, float(score)))
                for node, score in zip(selected, values)
            }
        weights = self._adaptive_weights(query)
        current = now or utcnow()
        time_query = bool(set(normalized_tokens(query)) & TIME_CUES)
        scored: list[PlanetaryHit] = []
        for node in nodes:
            components = {
                "semantic_gravity": max(0.0, cosine(vector, node.vector)),
                "bm25_resonance": bm25.get(node.id, 0.0),
                "entity_resonance": len(node.entities & query_entities) / max(1, len(node.entities | query_entities)),
                "temporal_phase": self._temporal_phase(node, current, time_query),
                "graph_flux": self.durable.engine._relation_signal(node, query_entities),
                "authority": node.source_trust,
                "learned_utility": node.utility_ema,
                "neural_rerank": reranks.get(node.id, 0.0),
                "contradiction_repulsion": self._contradiction(query, node),
            }
            attraction = sum(
                weights[key] * components[key]
                for key in weights if key != "contradiction_repulsion"
            )
            repulsion = weights["contradiction_repulsion"] * components["contradiction_repulsion"]
            state_factor = 1.0 if node.state == "stable" else 0.78
            force = max(0.0, state_factor * (attraction - repulsion))
            radius = 1.0 / (0.05 + force)
            energy = -node.mass * force
            dominant = max(
                (key for key in components if key != "contradiction_repulsion"),
                key=lambda key: weights.get(key, 0.0) * components[key],
            )
            scored.append(PlanetaryHit(
                node.id, node.text, force, radius, energy, components,
                {
                    "dominant_force": dominant,
                    "adaptive_weights": weights,
                    "state_factor": state_factor,
                    "formula": "F=S*(Σ(w_k*x_k)-w_c*C); r=1/(0.05+F); E=-mass*F",
                },
            ))
        scored.sort(key=lambda item: item.force, reverse=True)
        trace_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self._retrieval_traces[trace_key] = {
            item.memory_id: dict(item.components) for item in scored[: self.config.candidate_limit]
        }
        if len(self._retrieval_traces) > 128:
            self._retrieval_traces.pop(next(iter(self._retrieval_traces)))
        budget = budget_chars or self.durable.engine.config.default_budget_chars
        output, used = [], 0
        for hit in scored:
            if len(output) >= top_k:
                break
            if output and used + len(hit.text) > budget:
                continue
            output.append(hit)
            used += len(hit.text)
        return output

    def retrieval_feedback(
        self,
        query: str,
        useful_memory_id: str,
        rejected_memory_ids: Iterable[str] = (),
        *,
        learning_rate: float = 0.035,
    ) -> dict[str, float]:
        """Learn bounded planetary force weights from an explicit ranking choice."""
        trace_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        trace = self._retrieval_traces.get(trace_key)
        if not trace or useful_memory_id not in trace:
            raise ValueError("retrieve the query before providing ranking feedback")
        rejected_ids = list(rejected_memory_ids)
        rejected = [trace[item] for item in rejected_ids if item in trace]
        if not rejected:
            rejected = [value for key, value in trace.items() if key != useful_memory_id][:5]
        if not rejected:
            return vars(self.config.weights).copy()
        useful = trace[useful_memory_id]
        for field_name in (
            "semantic_gravity", "bm25_resonance", "entity_resonance", "temporal_phase",
            "graph_flux", "authority", "learned_utility", "neural_rerank",
        ):
            baseline = sum(item[field_name] for item in rejected) / len(rejected)
            current = getattr(self.config.weights, field_name)
            setattr(
                self.config.weights,
                field_name,
                max(0.01, min(0.60, current + learning_rate * (useful[field_name] - baseline))),
            )
        baseline_repulsion = sum(item["contradiction_repulsion"] for item in rejected) / len(rejected)
        repulsion = self.config.weights.contradiction_repulsion
        self.config.weights.contradiction_repulsion = max(
            0.01,
            min(
                0.60,
                repulsion + learning_rate * (
                    baseline_repulsion - useful["contradiction_repulsion"]
                ),
            ),
        )
        self._commit_planetary(
            "retrieval.feedback", useful_memory_id,
            {
                "query_sha256": trace_key,
                "useful_memory_id": useful_memory_id,
                "rejected_memory_ids": rejected_ids,
            },
        )
        return vars(self.config.weights).copy()

    def entity_paths(self, start: str, target: str, **kwargs) -> list[dict[str, object]]:
        return self.entity_system.paths(
            start, target,
            max_hops=kwargs.pop("max_hops", self.config.graph_hops),
            beam_width=kwargs.pop("beam_width", self.config.graph_beam_width),
            **kwargs,
        )

    def reflect(self, top_k: int = 16) -> ReflectionReport:
        if not self.policy.allows(
            self.principal_id, "reflect", scope=None, classification="internal"
        ):
            raise AuthorizationError("ABAC policy denied reflection")
        nodes = sorted(
            (node for node in self.durable.engine.nodes.values() if self._allowed_node(node)),
            key=lambda node: node.mass * node.utility_ema * node.confidence,
            reverse=True,
        )[: max(1, min(top_k, 32))]
        if self.reflection_proposer:
            proposals = self.reflection_proposer(nodes)
        else:
            proposals = []
            for node in nodes:
                fact = node.fact
                if fact.subject and fact.predicate and fact.object_value:
                    proposals.append(ReflectionClaim(
                        f"{fact.subject} {fact.predicate} {fact.object_value}",
                        (node.id,), fact.confidence,
                    ))
        accepted: list[ReflectionClaim] = []
        rejected: list[dict[str, object]] = []
        for claim in proposals[:24]:
            evidence = [self.durable.get_memory(item) for item in claim.evidence_ids]
            claim_tokens = set(normalized_tokens(claim.text))
            evidence_tokens = set(
                token for node in evidence for token in normalized_tokens(node.text)
            )
            grounding = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
            conflicts = 0
            for left in evidence:
                for right in evidence:
                    if left.id >= right.id or left.fact_key != right.fact_key:
                        continue
                    if left.fact.object_value != right.fact.object_value:
                        conflicts += 1
            if grounding < self.config.reflection_min_grounding or conflicts:
                rejected.append({
                    "claim": claim.text,
                    "grounding": grounding,
                    "conflicts": conflicts,
                })
            else:
                accepted.append(claim)
        memory_id = None
        if accepted:
            summary = "Reflection summary: " + "；".join(claim.text for claim in accepted)
            result = self.durable.write(
                summary[:3000],
                source="assistant_inference",
                kind="observation",
                explicit_importance=0.45,
                auto_extract=False,
                extraction_metadata={
                    "reflection": {
                        "evidence_ids": sorted({item for claim in accepted for item in claim.evidence_ids}),
                        "verified": True,
                    }
                },
            )
            memory_id = result.memory_id
        self._writes_since_reflection = 0
        self._commit_planetary(
            "reflection.verify", memory_id,
            {
                "memory_id": memory_id,
                "accepted": len(accepted),
                "rejected": len(rejected),
            },
        )
        total = len(accepted) + len(rejected)
        return ReflectionReport(
            tuple(accepted), tuple(rejected), memory_id,
            len(accepted) / total if total else 0.0,
        )

    def put_block(self, block: MemoryBlock, *, actor_agent: str) -> None:
        existing = self.blocks.get(block.label)
        if existing:
            if existing.read_only or (
                actor_agent != existing.owner_agent and actor_agent not in existing.writers
            ):
                raise AuthorizationError("agent cannot update this memory block")
            block.version = existing.version + 1
        elif actor_agent != block.owner_agent:
            raise AuthorizationError("only the owner can create a memory block")
        if len(block.value) > block.max_characters:
            raise ValueError("memory block exceeds its bounded orbit")
        self.blocks[block.label] = block
        self._commit_planetary(
            "block.put", block.label,
            {"label": block.label, "version": block.version},
        )

    def context_blocks(self, agent_id: str) -> list[MemoryBlock]:
        return [
            block for block in self.blocks.values()
            if agent_id == block.owner_agent or agent_id in block.readers or agent_id in block.writers
        ]

    def add_procedure(self, procedure: ProcedureOrbit) -> None:
        if not self.policy.allows(
            self.principal_id, "procedure", scope=None, classification="internal"
        ):
            raise AuthorizationError("ABAC policy denied procedure memory")
        if not procedure.steps:
            raise ValueError("procedure requires at least one step")
        self.procedures[procedure.id] = procedure
        self._commit_planetary(
            "procedure.put", procedure.id,
            {"procedure_id": procedure.id, "name": procedure.name},
        )

    def select_procedure(self, task: str, top_k: int = 3) -> list[tuple[float, ProcedureOrbit]]:
        task_tokens = set(normalized_tokens(task))
        total_trials = sum(item.successes + item.failures for item in self.procedures.values()) + 1
        scored = []
        for procedure in self.procedures.values():
            trigger_tokens = set(normalized_tokens(procedure.trigger + " " + procedure.name))
            resonance = len(task_tokens & trigger_tokens) / max(1, len(task_tokens | trigger_tokens))
            trials = procedure.successes + procedure.failures
            exploration = math.sqrt(2.0 * math.log(total_trials + 1.0) / (trials + 1.0))
            score = 0.65 * resonance + 0.25 * procedure.confidence + 0.10 * min(1.0, exploration)
            scored.append((score, procedure))
        return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]

    def procedure_feedback(self, procedure_id: str, success: bool) -> None:
        procedure = self.procedures[procedure_id]
        if success:
            procedure.successes += 1
        else:
            procedure.failures += 1
        procedure.last_used = utcnow()
        self._commit_planetary(
            "procedure.feedback", procedure.id,
            {"procedure_id": procedure.id, "success": success},
        )

    def register_model_memory(self, item: ModelMemoryOrbit) -> None:
        """Register model-specific memory without silently applying incompatible state."""
        if item.kind not in {"prompt", "kv_cache", "adapter"}:
            raise ValueError("unsupported model-memory kind")
        if not item.model_fingerprint or not item.payload:
            raise ValueError("model memory requires a fingerprint and payload")
        if len(item.payload) > 64 * 1024 * 1024:
            raise ValueError("model-memory payload exceeds the 64 MiB orbit bound")
        item.quality = max(0.0, min(1.0, item.quality))
        self.model_memories[item.id] = item
        self._commit_planetary(
            "model_memory.put", item.id,
            {"id": item.id, "kind": item.kind, "model_fingerprint": item.model_fingerprint},
        )

    def select_model_memory(
        self, model_fingerprint: str, task: str, *, kind: str | None = None, top_k: int = 3
    ) -> list[tuple[float, ModelMemoryOrbit]]:
        task_tokens = set(normalized_tokens(task))
        scored = []
        for item in self.model_memories.values():
            if item.model_fingerprint != model_fingerprint or (kind and item.kind != kind):
                continue
            signature_tokens = set(normalized_tokens(item.task_signature))
            resonance = len(task_tokens & signature_tokens) / max(1, len(task_tokens | signature_tokens))
            score = 0.72 * resonance + 0.28 * item.quality
            scored.append((score, item))
        return sorted(scored, key=lambda value: value[0], reverse=True)[:top_k]

    def model_memory_feedback(self, item_id: str, useful: bool) -> None:
        item = self.model_memories[item_id]
        item.uses += 1
        target = 1.0 if useful else 0.0
        item.quality = 0.85 * item.quality + 0.15 * target
        self._commit_planetary(
            "model_memory.feedback", item.id,
            {"id": item.id, "useful": useful},
        )

    def purge_expired(
        self, now: datetime | None = None, *, hard_delete: bool = False
    ) -> dict[str, int]:
        """Archive expired memories; optionally redact and physically remove them."""
        self.durable.database.require(self.durable.tenant_id, self.principal_id, "admin")
        current = now or utcnow()
        expired = []
        for node in list(self.durable.engine.nodes.values()):
            value = self._attributes(node).get("expires_at")
            expiry = self._parse_dt(value) if isinstance(value, str) else None
            if expiry and current >= expiry:
                expired.append(node)
        for node in expired:
            self.durable.engine._archive_node(node)
        if expired:
            self.durable._commit(
                "retention.expire", None,
                {"memory_ids": [node.id for node in expired], "expired_at": current},
                planetary_state_payload=self._planetary_state_bytes(),
            )
        redacted = 0
        if hard_delete:
            for node in expired:
                result = self.durable.database.hard_delete_memory(
                    self.durable.tenant_id, self.principal_id, node.id
                )
                redacted += result["events_redacted"]
            if expired:
                self._rebuild_planetary_state()
        return {
            "expired": len(expired),
            "hard_deleted": len(expired) if hard_delete else 0,
            "events_redacted": redacted,
        }

    def snapshot(self) -> dict[str, object]:
        promoted = sum(item.promoted for item in self.entity_system.ontology.values())
        return {
            "version": "UMD 3.7",
            "formula": "F=S*(Σ(w_k*x_k)-w_c*C); r=1/(ε+F); E=-mF",
            "active_entities": len(self.entity_system.entities),
            "temporal_orbits": len(self.entity_system.orbits),
            "ontology_predicates": len(self.entity_system.ontology),
            "promoted_ontology_predicates": promoted,
            "memory_blocks": len(self.blocks),
            "procedures": len(self.procedures),
            "model_memory_orbits": len(self.model_memories),
            "ann_indexed_memories": len(self.index.tokens),
            "durable": self.durable.snapshot(),
        }
