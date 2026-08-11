"""UMD 3.12 query-independent evidence constellations.

Capsules group at most four source memories while retaining immutable source
IDs.  They are reconstructed from durable memory after restart and keep no
second plaintext cache.  Retrieval remains label-blind: query-time work can
select existing capsules but can never change their membership.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

try:
    from .umd35_core import MemoryNode, cosine, normalized_tokens, utcnow
    from .umd310_cognitive import UMD310CognitiveMemory, UMD310Config
except ImportError:
    from umd35_core import MemoryNode, cosine, normalized_tokens, utcnow
    from umd310_cognitive import UMD310CognitiveMemory, UMD310Config


CapsuleKind = Literal["atomic", "dialogue_pair", "episode", "version_chain", "directive_execution"]
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b|\b[A-Za-z]+[-_.:/]\d+(?:\.\d+)*\b")
SUBGOAL_SPLIT_RE = re.compile(
    r"\s*(?:,|;|\band\b|\bthen\b|\bincluding\b|\bas well as\b|以及|并且|然后|包括|、)\s*",
    re.IGNORECASE,
)


@dataclass
class UMD312Config(UMD310Config):
    capsule_max_sources: int = 4
    capsule_candidate_limit: int = 160
    capsules_per_source_limit: int = 8
    capsule_absolute_gate: float = 0.18
    capsule_relative_gate: float = 0.72
    capsule_subgoal_bonus: float = 0.18
    capsule_entity_bonus: float = 0.14
    capsule_time_bonus: float = 0.12
    capsule_contradiction_bonus: float = 0.10
    capsule_source_bonus: float = 0.08
    capsule_redundancy_penalty: float = 0.16
    capsule_uncertainty_penalty: float = 0.12
    capsule_cold_rebuild_limit: int = 512


@dataclass(frozen=True)
class EvidenceCapsule:
    capsule_id: str
    kind: CapsuleKind
    source_ids: tuple[str, ...]


@dataclass
class CapsuleHit:
    capsule_id: str
    kind: CapsuleKind
    source_ids: tuple[str, ...]
    text: str
    force: float
    components: dict[str, float]
    explanation: dict[str, object]


class UMD312ConstellationMemory(UMD310CognitiveMemory):
    """UMD 3.10 plus bounded, provenance-preserving evidence capsules."""

    def __init__(self, durable, *, config: UMD312Config | None = None, **kwargs) -> None:
        self.capsule_config = config or UMD312Config()
        self.capsules: dict[str, EvidenceCapsule] = {}
        self.capsules_by_source: dict[str, set[str]] = defaultdict(set)
        self._capsules_dirty = True
        super().__init__(durable, config=self.capsule_config, **kwargs)

    def _index_node(self, node: MemoryNode) -> None:
        super()._index_node(node)
        self._capsules_dirty = True

    def _remove_node(self, memory_id: str, *, preserve_directive_history: bool = False) -> None:
        super()._remove_node(memory_id, preserve_directive_history=preserve_directive_history)
        self._capsules_dirty = True

    @staticmethod
    def _capsule_id(kind: CapsuleKind, source_ids: Iterable[str]) -> str:
        payload = kind + "\0" + "\0".join(source_ids)
        return "cap_" + hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _eligible(self, node: MemoryNode, *, history: bool = False) -> bool:
        states = {"stable", "provisional"}
        if history:
            states |= {"invalidated", "archived"}
        return (
            node.state in states
            and self._allowed_node(node)
            and bool(node.text or self.durable.get_text(node.id))
        )

    def _register_capsule(self, kind: CapsuleKind, source_ids: Iterable[str]) -> None:
        unique = tuple(dict.fromkeys(source_ids))[: self.capsule_config.capsule_max_sources]
        if not unique:
            return
        if any(len(self.capsules_by_source[source]) >= self.capsule_config.capsules_per_source_limit
               for source in unique):
            return
        capsule_id = self._capsule_id(kind, unique)
        if capsule_id in self.capsules:
            return
        capsule = EvidenceCapsule(capsule_id, kind, unique)
        self.capsules[capsule_id] = capsule
        for source in unique:
            self.capsules_by_source[source].add(capsule_id)

    def _rebuild_capsules(self) -> None:
        self.capsules = {}
        self.capsules_by_source = defaultdict(set)
        # Atomic capsules make the capsule API lossless relative to the coarse
        # retriever; multi-source capsules add coverage rather than replacing it.
        for node in self.durable.engine.nodes.values():
            if self._eligible(node):
                self._register_capsule("atomic", (node.id,))

        for _, sequence in sorted(self.episodes_by_planet.items()):
            nodes = [self._node(memory_id) for _, memory_id in sequence]
            nodes = [node for node in nodes if node is not None and self._eligible(node)]
            ids = [node.id for node in nodes]
            for index in range(len(ids) - 1):
                self._register_capsule("dialogue_pair", ids[index:index + 2])
            for index in range(0, len(ids), 3):
                chunk = ids[index:index + self.capsule_config.capsule_max_sources]
                if len(chunk) >= 2:
                    self._register_capsule("episode", chunk)
            for index, node in enumerate(nodes):
                if self._directive_role(node.text, node.kind):
                    self._register_capsule("directive_execution", ids[index:index + 2])

        by_fact: dict[tuple[str, str], list[MemoryNode]] = defaultdict(list)
        for node in self.durable.engine.nodes.values():
            if node.fact_key and self._eligible(node, history=True):
                by_fact[node.scope or "*", node.fact_key].append(node)
        # Superseded facts are intentionally cold in UMD's durable layer.
        # Stream only a bounded window and retain IDs, never plaintext objects.
        hot_ids = set(self.durable.engine.nodes)
        for node in self.durable.cold_history(self.capsule_config.capsule_cold_rebuild_limit):
            if node.id not in hot_ids and node.fact_key and self._eligible(node, history=True):
                by_fact[node.scope or "*", node.fact_key].append(node)
        for nodes in by_fact.values():
            nodes.sort(key=lambda node: (node.updated_at, node.id))
            if len(nodes) >= 2:
                self._register_capsule(
                    "version_chain",
                    (node.id for node in nodes[-self.capsule_config.capsule_max_sources:]),
                )
        self._capsules_dirty = False

    def _ensure_capsules(self) -> None:
        if self._capsules_dirty:
            self._rebuild_capsules()

    @staticmethod
    def _subgoals(query: str) -> list[str]:
        values = [query.strip()]
        values.extend(part.strip() for part in SUBGOAL_SPLIT_RE.split(query) if len(part.strip()) >= 4)
        return list(dict.fromkeys(values))[:6]

    def _capsule_nodes(self, capsule: EvidenceCapsule, scope: str | None) -> list[MemoryNode]:
        nodes = []
        for source_id in capsule.source_ids:
            node = self._node(source_id)
            if node is None or not self._eligible(node, history=capsule.kind == "version_chain"):
                return []
            if scope and node.scope and node.scope != scope:
                return []
            nodes.append(node)
        return nodes

    def _score_capsule(
        self, capsule: EvidenceCapsule, nodes: list[MemoryNode], query: str,
        subgoals: list[str], query_vectors: list, now: datetime,
    ) -> tuple[float, dict[str, float], set[int], set[str], set[str]]:
        texts = [node.text or self.durable.get_text(node.id) for node in nodes]
        node_vectors = [
            node.vector if node.vector.size else self.durable.engine.encoder.encode(text)
            for node, text in zip(nodes, texts)
        ]
        subgoal_scores: list[float] = []
        for subgoal, query_vector in zip(subgoals, query_vectors):
            semantic_part = max(max(0.0, cosine(query_vector, vector)) for vector in node_vectors)
            lexical_part = max(self._lexical_overlap(subgoal, text) for text in texts)
            subgoal_scores.append(0.72 * semantic_part + 0.28 * lexical_part)
        semantic = max(
            max(0.0, cosine(query_vectors[0], vector)) for vector in node_vectors
        )
        lexical = max(self._lexical_overlap(query, text) for text in texts)
        query_entities = set(ENTITY_RE.findall(query))
        entities = set().union(*(node.entities for node in nodes))
        entities.update(item for text in texts for item in ENTITY_RE.findall(text))
        entity = len(query_entities & entities) / max(1, len(query_entities)) if query_entities else 0.0
        mode = self._query_mode(query)
        time_buckets = {node.created_at.strftime("%Y-%m") for node in nodes}
        if mode.episodic:
            temporal = min(1.0, 0.45 + 0.20 * len(time_buckets))
        else:
            ages = [max(0.0, (now - node.updated_at).total_seconds() / 86400.0) for node in nodes]
            temporal = max(math.exp(-age / 365.0) for age in ages)
        contradiction = float(mode.contradiction and capsule.kind == "version_chain")
        directive = float(
            (mode.instructions or mode.preferences) and capsule.kind == "directive_execution"
        )
        provenance = sum(node.source_trust for node in nodes) / len(nodes)
        relevance = (
            0.26 * semantic + 0.18 * lexical + 0.14 * entity + 0.12 * temporal
            + 0.10 * contradiction + 0.10 * directive + 0.10 * provenance
        )
        best_subgoal = max(subgoal_scores, default=0.0)
        covered_subgoals = {
            index for index, score in enumerate(subgoal_scores)
            if score >= max(0.24, 0.68 * best_subgoal)
        }
        components = {
            "semantic": semantic, "lexical": lexical, "entity": entity,
            "temporal": temporal, "contradiction_pair": contradiction,
            "directive_execution": directive, "provenance_confidence": provenance,
        }
        return relevance, components, covered_subgoals, entities, time_buckets

    def retrieve_capsules(
        self, query: str, *, scope: str | None = None, top_k: int = 10,
        budget_chars: int | None = None, now: datetime | None = None,
    ) -> list[CapsuleHit]:
        if top_k < 1:
            return []
        self._ensure_capsules()
        current = now or utcnow()
        coarse = super().retrieve(
            query, scope=scope,
            top_k=min(self.capsule_config.capsule_candidate_limit, max(64, top_k * 8)),
            budget_chars=max(
                budget_chars or 0,
                self.durable.engine.config.default_budget_chars * 16,
            ),
            now=current,
        )
        candidate_ids: list[str] = []
        for hit in coarse:
            candidate_ids.extend(sorted(self.capsules_by_source.get(hit.memory_id, ())))
        candidate_ids = list(dict.fromkeys(candidate_ids))[: self.capsule_config.capsule_candidate_limit]
        subgoals = self._subgoals(query)
        query_vectors = [self.durable.engine.encoder.encode(subgoal) for subgoal in subgoals]
        scored = []
        for capsule_id in candidate_ids:
            capsule = self.capsules[capsule_id]
            nodes = self._capsule_nodes(capsule, scope)
            if not nodes:
                continue
            relevance, components, goals, entities, times = self._score_capsule(
                capsule, nodes, query, subgoals, query_vectors, current,
            )
            scored.append((capsule, nodes, relevance, components, goals, entities, times))
        if not scored:
            return []
        best_relevance = max(item[2] for item in scored)
        gate = max(
            self.capsule_config.capsule_absolute_gate,
            self.capsule_config.capsule_relative_gate * best_relevance,
        )
        remaining = [item for item in scored if item[2] >= gate]
        selected = []
        used_goals: set[int] = set()
        used_entities: set[str] = set()
        used_times: set[str] = set()
        used_sources: set[str] = set()
        mode = self._query_mode(query)
        while remaining and len(selected) < top_k:
            best_index, best_value = 0, float("-inf")
            for index, item in enumerate(remaining):
                capsule, nodes, relevance, _, goals, entities, times = item
                new_goal = len(goals - used_goals) / max(1, len(goals))
                new_entity = len(entities - used_entities) / max(1, len(entities))
                new_time = float(bool(times - used_times))
                contradiction = float(mode.contradiction and capsule.kind == "version_chain")
                new_source = (
                    len(set(capsule.source_ids) - used_sources)
                    / self.capsule_config.capsule_max_sources
                )
                redundancy = len(set(capsule.source_ids) & used_sources) / len(capsule.source_ids)
                uncertainty = 1.0 - sum(node.source_trust for node in nodes) / len(nodes)
                value = (
                    relevance
                    + self.capsule_config.capsule_subgoal_bonus * new_goal
                    + self.capsule_config.capsule_entity_bonus * new_entity
                    + self.capsule_config.capsule_time_bonus * new_time
                    + self.capsule_config.capsule_contradiction_bonus * contradiction
                    + self.capsule_config.capsule_source_bonus * new_source
                    - self.capsule_config.capsule_redundancy_penalty * redundancy
                    - self.capsule_config.capsule_uncertainty_penalty * uncertainty
                )
                if value > best_value or (
                    value == best_value and capsule.capsule_id < remaining[best_index][0].capsule_id
                ):
                    best_index, best_value = index, value
            item = remaining.pop(best_index)
            selected.append((item, best_value))
            capsule, _, _, _, goals, entities, times = item
            used_goals.update(goals)
            used_entities.update(entities)
            used_times.update(times)
            used_sources.update(capsule.source_ids)

        budget = budget_chars or self.durable.engine.config.default_budget_chars
        output: list[CapsuleHit] = []
        used_chars = 0
        emitted_sources: set[str] = set()
        for item, adjusted in selected:
            capsule, nodes, relevance, components, goals, _, _ = item
            text = "\n".join(
                f"[source:{node.id}] {node.text or self.durable.get_text(node.id)}" for node in nodes
            )
            output_capsule = capsule
            output_nodes = nodes
            budget_fallback_from = None
            if used_chars + len(text) > budget:
                remaining_budget = budget - used_chars
                fitting = []
                for node in nodes:
                    if node.id in emitted_sources:
                        continue
                    node_text = f"[source:{node.id}] {node.text or self.durable.get_text(node.id)}"
                    if len(node_text) <= remaining_budget:
                        fitting.append((self._lexical_overlap(query, node_text), node, node_text))
                if fitting:
                    _, node, text = max(fitting, key=lambda value: (value[0], value[1].id))
                    output_capsule = EvidenceCapsule(
                        self._capsule_id("atomic", (node.id,)), "atomic", (node.id,)
                    )
                    output_nodes = [node]
                    budget_fallback_from = capsule.capsule_id
                elif not output and remaining_budget > 0:
                    # A single source may itself exceed the answer budget. Its
                    # atomic membership/provenance stays intact; only the
                    # rendered excerpt is truncated.
                    node = max(
                        nodes,
                        key=lambda value: self._lexical_overlap(
                            query, value.text or self.durable.get_text(value.id)
                        ),
                    )
                    raw = f"[source:{node.id}] {node.text or self.durable.get_text(node.id)}"
                    text = raw[:remaining_budget]
                    output_capsule = EvidenceCapsule(
                        self._capsule_id("atomic", (node.id,)), "atomic", (node.id,)
                    )
                    output_nodes = [node]
                    budget_fallback_from = capsule.capsule_id
                else:
                    continue
            output.append(CapsuleHit(
                output_capsule.capsule_id, output_capsule.kind, output_capsule.source_ids,
                text, relevance,
                components,
                {
                    "formula": "R=.26S+.18L+.14E+.12T+.10C+.10D+.10P",
                    "selection": "R+.18subgoal+.14entity+.12time+.10contradiction+.08source-.16redundancy-.12uncertainty",
                    "coverage_adjusted_force": adjusted,
                    "covered_subgoals": sorted(goals),
                    "relevance_gate": gate,
                    "query_independent_membership": True,
                    "max_sources": self.capsule_config.capsule_max_sources,
                    "budget_fallback_from": budget_fallback_from,
                    "rendered_source_count": len(output_nodes),
                },
            ))
            used_chars += len(text)
            emitted_sources.update(output_capsule.source_ids)
        return output

    def capsule_index_container_bytes(self) -> int:
        # Excludes shared source-ID strings and all plaintext/node/vector data.
        return (
            sys.getsizeof(self.capsules)
            + sum(
                sys.getsizeof(value) + sys.getsizeof(value.source_ids)
                for value in self.capsules.values()
            )
            + sys.getsizeof(self.capsules_by_source)
            + sum(sys.getsizeof(value) for value in self.capsules_by_source.values())
        )

    def snapshot(self) -> dict[str, object]:
        self._ensure_capsules()
        base = super().snapshot()
        source_links = sum(len(capsule.source_ids) for capsule in self.capsules.values())
        active = max(1, len(self.durable.engine.active_ids))
        container_bytes = self.capsule_index_container_bytes()
        base.update({
            "version": "UMD 3.12",
            "capsules": len(self.capsules),
            "capsule_source_links": source_links,
            "capsule_max_sources": self.capsule_config.capsule_max_sources,
            "capsule_index_container_bytes": container_bytes,
            "capsule_index_container_bytes_per_active_memory": container_bytes / active,
            "capsule_membership_query_independent": True,
            "capsule_plaintext_cache": False,
        })
        return base
