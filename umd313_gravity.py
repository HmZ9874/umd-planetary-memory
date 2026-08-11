"""UMD 3.13 relative-gravity retrieval.

The durable hierarchy remains immutable.  A query creates transient attractors
for its subgoals and lets memories form provenance-preserving tidal bridges
when the new field is stronger than home-orbit inertia.  Nothing is re-parented
or copied, so restart recovery, tenant isolation and source provenance retain
their UMD 3.12 semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

try:
    from .umd35_core import MemoryNode, cosine, normalized_tokens, utcnow
    from .umd312_capsules import UMD312Config, UMD312ConstellationMemory
except ImportError:
    from umd35_core import MemoryNode, cosine, normalized_tokens, utcnow
    from umd312_capsules import UMD312Config, UMD312ConstellationMemory


GRAVITY_SPLIT_RE = re.compile(
    r"\s*(?:,|;|、|，|；|\band\b|\bthen\b|\bwhile\b|\bas well as\b|以及|并且|然后|同时|还有)\s*",
    re.IGNORECASE,
)
GRAVITY_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_.-]{1,}\b|[\u4e00-\u9fff]{2,8}"
)
TIME_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b|"
    r"\b(?:today|yesterday|tomorrow|before|after|first|last|next)\b|"
    r"今天|昨天|明天|之前|之后|首次|最后|下次",
    re.IGNORECASE,
)


@dataclass
class UMD313Config(UMD312Config):
    gravity_candidate_limit: int = 256
    gravity_subgoal_limit: int = 6
    gravity_sources_per_bridge: int = 4
    gravity_candidates_per_goal: int = 32
    gravity_capture_temperature: float = 0.16
    gravity_capture_threshold: float = 0.54
    gravity_relative_margin: float = 0.025
    gravity_home_inertia: float = 0.10
    gravity_migration_cost: float = 0.035
    gravity_uncertainty_cost: float = 0.07
    gravity_new_goal_bonus: float = 0.20
    gravity_cross_planet_bonus: float = 0.12
    gravity_relation_bridge_bonus: float = 0.10
    gravity_time_phase_bonus: float = 0.07
    gravity_source_bonus: float = 0.05
    gravity_redundancy_penalty: float = 0.18


@dataclass(frozen=True)
class GravityElement:
    semantic_resonance: float
    lexical_resonance: float
    entity_lock: float
    temporal_phase: float
    relation_bridge: float
    provenance: float
    vitality: float
    mass: float
    novelty: float
    home_inertia: float
    migration_cost: float
    uncertainty: float
    raw_attraction: float
    net_potential: float
    capture_probability: float


@dataclass
class GravityHit:
    bridge_id: str
    source_ids: tuple[str, ...]
    text: str
    force: float
    elements: dict[str, float]
    explanation: dict[str, object]


@dataclass
class _GravityCandidate:
    node: MemoryNode
    text: str
    goal: int
    elements: GravityElement
    goal_potentials: tuple[float, ...]
    entities: set[str]
    time_keys: set[str]


class UMD313RelativeGravityMemory(UMD312ConstellationMemory):
    """Dynamic query-time bridges over immutable UMD memory orbits."""

    FORMULA = (
        "A=.30Q+.14X+.15Xi+.08Tau+.12Lambda+.08Pi+.05V+.03M+.05N; "
        "Delta=A-.10I-.035K-.07U; Pcapture=sigmoid(Delta/temperature)"
    )
    OPTIMIZER = (
        "Omega=mean(Delta)+.20new_goal+.12cross_planet+.10relation+"
        ".07new_time+.05new_source-.18redundancy"
    )

    def __init__(self, durable, *, config: UMD313Config | None = None, **kwargs) -> None:
        self.gravity_config = config or UMD313Config()
        super().__init__(durable, config=self.gravity_config, **kwargs)

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-20.0, min(20.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    def _gravity_subgoals(self, query: str) -> list[str]:
        parts = [part.strip() for part in GRAVITY_SPLIT_RE.split(query) if len(part.strip()) >= 3]
        if len(parts) <= 1:
            return [query.strip()]
        return list(dict.fromkeys(parts))[: self.gravity_config.gravity_subgoal_limit]

    @staticmethod
    def _entities(text: str) -> set[str]:
        return {item.casefold() for item in GRAVITY_ENTITY_RE.findall(text)}

    @staticmethod
    def _time_keys(text: str, node: MemoryNode) -> set[str]:
        values = {item.casefold() for item in TIME_RE.findall(text)}
        values.add(node.created_at.strftime("%Y-%m"))
        return values

    @staticmethod
    def _set_overlap(left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left))

    def _home_strength(self, node: MemoryNode) -> float:
        if not node.planet_id or not node.vector.size:
            return 0.0
        planet = self.durable.engine.planets.get(node.planet_id)
        if planet is None or not planet.centroid.size:
            return 0.0
        return max(0.0, cosine(node.vector, planet.centroid))

    @staticmethod
    def _relation_terms(node: MemoryNode) -> set[str]:
        terms: set[str] = set()
        for relation in node.entity_relations:
            terms.update(normalized_tokens(relation.subject))
            terms.update(normalized_tokens(relation.predicate))
            terms.update(normalized_tokens(relation.object_value))
        return terms

    def _score_node_goal(
        self, node: MemoryNode, text: str, goal: str, goal_vector, now: datetime,
        query_entities: set[str], query_time: set[str], all_candidate_entities: set[str],
    ) -> GravityElement:
        vector = node.vector if node.vector.size else self.durable.engine.encoder.encode(text)
        semantic = max(0.0, cosine(goal_vector, vector))
        lexical = self._lexical_overlap(goal, text)
        node_entities = {item.casefold() for item in node.entities} | self._entities(text)
        entity = self._set_overlap(query_entities, node_entities) if query_entities else 0.0
        node_time = self._time_keys(text, node)
        temporal = self._set_overlap(query_time, node_time) if query_time else 0.0
        relation_terms = self._relation_terms(node)
        goal_terms = set(normalized_tokens(goal))
        relation = len(relation_terms & goal_terms) / max(1, len(goal_terms))
        provenance = max(0.0, min(1.0, node.source_trust))
        vitality = max(0.0, min(1.0, 0.6 * node.utility_ema + 0.4 * node.confidence))
        mass = math.tanh(math.log1p(max(0.0, node.mass)) / 3.0)
        novelty = 1.0 - self._set_overlap(all_candidate_entities, node_entities)
        home = self._home_strength(node)
        migration = float(bool(node.planet_id))
        uncertainty = 1.0 - provenance
        attraction = (
            0.30 * semantic + 0.14 * lexical + 0.15 * entity + 0.08 * temporal
            + 0.12 * relation + 0.08 * provenance + 0.05 * vitality
            + 0.03 * mass + 0.05 * novelty
        )
        net = (
            attraction
            - self.gravity_config.gravity_home_inertia * home
            - self.gravity_config.gravity_migration_cost * migration
            - self.gravity_config.gravity_uncertainty_cost * uncertainty
        )
        probability = self._sigmoid(net / self.gravity_config.gravity_capture_temperature)
        return GravityElement(
            semantic, lexical, entity, temporal, relation, provenance, vitality, mass,
            novelty, home, migration, uncertainty, attraction, net, probability,
        )

    def _gravity_candidates(
        self, query: str, scope: str | None, now: datetime,
    ) -> tuple[list[str], list[_GravityCandidate]]:
        goals = self._gravity_subgoals(query)
        vectors = [self.durable.engine.encoder.encode(goal) for goal in goals]
        query_entities = self._entities(query)
        query_time = {item.casefold() for item in TIME_RE.findall(query)}
        nodes = [
            node for node in self.durable.engine.nodes.values()
            if self._eligible(node) and (not scope or not node.scope or node.scope == scope)
        ]
        population_entities = set().union(*(node.entities for node in nodes)) if nodes else set()
        candidates: list[_GravityCandidate] = []
        for node in nodes:
            text = node.text or self.durable.get_text(node.id)
            elements = [
                self._score_node_goal(
                    node, text, goal, vector, now, query_entities, query_time,
                    {item.casefold() for item in population_entities},
                )
                for goal, vector in zip(goals, vectors)
            ]
            potentials = tuple(item.net_potential for item in elements)
            order = sorted(range(len(elements)), key=lambda index: (-potentials[index], index))
            best = order[0]
            runner_up = potentials[order[1]] if len(order) > 1 else float("-inf")
            relative_margin = potentials[best] - runner_up
            captured = (
                elements[best].capture_probability >= self.gravity_config.gravity_capture_threshold
                and (len(order) == 1 or relative_margin >= self.gravity_config.gravity_relative_margin)
            )
            if captured:
                candidates.append(_GravityCandidate(
                    node, text, best, elements[best], potentials,
                    {item.casefold() for item in node.entities} | self._entities(text),
                    self._time_keys(text, node),
                ))

        # Each attractor reserves candidates before the global truncation.  A
        # dominant easy subgoal therefore cannot consume the full candidate set.
        by_goal: list[_GravityCandidate] = []
        for goal in range(len(goals)):
            values = [item for item in candidates if item.goal == goal]
            values.sort(key=lambda item: (-item.elements.net_potential, item.node.id))
            by_goal.extend(values[: self.gravity_config.gravity_candidates_per_goal])
        unique = {item.node.id: item for item in by_goal}
        ordered = sorted(
            unique.values(),
            key=lambda item: (-item.elements.net_potential, item.node.id),
        )[: self.gravity_config.gravity_candidate_limit]
        return goals, ordered

    @staticmethod
    def _candidate_link(left: _GravityCandidate, right: _GravityCandidate) -> float:
        shared_entities = len(left.entities & right.entities) / max(
            1, min(len(left.entities), len(right.entities))
        )
        left_relations = UMD313RelativeGravityMemory._relation_terms(left.node)
        right_relations = UMD313RelativeGravityMemory._relation_terms(right.node)
        relation_overlap = len(left_relations & right_relations) / max(
            1, min(len(left_relations), len(right_relations))
        )
        return max(shared_entities, relation_overlap)

    def _bridge_value(
        self, group: list[_GravityCandidate], candidate: _GravityCandidate,
        used_goals: set[int], used_sources: set[str], used_times: set[str],
    ) -> float:
        new_goal = float(candidate.goal not in used_goals and all(candidate.goal != item.goal for item in group))
        group_planets = {item.node.planet_id for item in group if item.node.planet_id}
        cross_planet = float(bool(group_planets) and candidate.node.planet_id not in group_planets)
        relation = max((self._candidate_link(item, candidate) for item in group), default=0.0)
        group_times = set().union(*(item.time_keys for item in group)) if group else set()
        new_time = float(bool(candidate.time_keys - used_times - group_times))
        new_source = float(candidate.node.id not in used_sources)
        redundant = max(
            (max(0.0, cosine(item.node.vector, candidate.node.vector)) for item in group
             if item.node.vector.size and candidate.node.vector.size),
            default=0.0,
        )
        return (
            candidate.elements.net_potential
            + self.gravity_config.gravity_new_goal_bonus * new_goal
            + self.gravity_config.gravity_cross_planet_bonus * cross_planet
            + self.gravity_config.gravity_relation_bridge_bonus * relation
            + self.gravity_config.gravity_time_phase_bonus * new_time
            + self.gravity_config.gravity_source_bonus * new_source
            - self.gravity_config.gravity_redundancy_penalty * redundant
        )

    def retrieve_gravity(
        self, query: str, *, scope: str | None = None, top_k: int = 10,
        budget_chars: int | None = None, now: datetime | None = None,
    ) -> list[GravityHit]:
        if top_k < 1:
            return []
        current = now or utcnow()
        goals, candidates = self._gravity_candidates(query, scope, current)
        if not candidates:
            return []
        remaining = list(candidates)
        used_goals: set[int] = set()
        used_sources: set[str] = set()
        used_times: set[str] = set()
        groups: list[tuple[list[_GravityCandidate], float]] = []
        max_sources = min(
            self.gravity_config.capsule_max_sources,
            self.gravity_config.gravity_sources_per_bridge,
        )
        while remaining and len(groups) < top_k:
            group: list[_GravityCandidate] = []
            values: list[float] = []
            while remaining and len(group) < max_sources:
                scored = [
                    (self._bridge_value(group, item, used_goals, used_sources, used_times), item)
                    for item in remaining
                ]
                value, chosen = max(scored, key=lambda pair: (pair[0], pair[1].node.id))
                if group and value < self.gravity_config.gravity_capture_threshold:
                    break
                group.append(chosen)
                values.append(value)
                remaining.remove(chosen)
                if len({item.goal for item in group}) >= len(goals):
                    break
            if not group:
                break
            groups.append((group, sum(values) / len(values)))
            used_goals.update(item.goal for item in group)
            used_sources.update(item.node.id for item in group)
            used_times.update(*(item.time_keys for item in group))

        budget = budget_chars or self.durable.engine.config.default_budget_chars
        output: list[GravityHit] = []
        used_chars = 0
        for group, value in groups:
            rendered = [f"[source:{item.node.id}] {item.text}" for item in group]
            text = "\n".join(rendered)
            if used_chars + len(text) > budget:
                fitting = [item for item in group if used_chars + len(f"[source:{item.node.id}] {item.text}") <= budget]
                if not fitting:
                    continue
                group = fitting
                text = "\n".join(f"[source:{item.node.id}] {item.text}" for item in group)
            source_ids = tuple(item.node.id for item in group)
            bridge_id = self._capsule_id("tidal_bridge", source_ids)  # type: ignore[arg-type]
            means = {
                "semantic_resonance": sum(item.elements.semantic_resonance for item in group) / len(group),
                "lexical_resonance": sum(item.elements.lexical_resonance for item in group) / len(group),
                "entity_lock": sum(item.elements.entity_lock for item in group) / len(group),
                "temporal_phase": sum(item.elements.temporal_phase for item in group) / len(group),
                "relation_bridge": sum(item.elements.relation_bridge for item in group) / len(group),
                "net_potential": sum(item.elements.net_potential for item in group) / len(group),
                "capture_probability": sum(item.elements.capture_probability for item in group) / len(group),
            }
            output.append(GravityHit(
                bridge_id, source_ids, text, value, means,
                {
                    "formula": self.FORMULA,
                    "optimizer": self.OPTIMIZER,
                    "ai_language": {
                        "Q": "semantic_resonance", "X": "lexical_resonance",
                        "Xi": "entity_lock", "Tau": "temporal_phase",
                        "Lambda": "relation_bridge", "Pi": "provenance",
                        "I": "home_orbit_inertia", "K": "migration_cost",
                        "U": "uncertainty", "Omega": "bridge_objective",
                    },
                    "query_attractors": goals,
                    "captured_goal_indices": sorted({item.goal for item in group}),
                    "home_planets": sorted({item.node.planet_id for item in group if item.node.planet_id}),
                    "cross_planet": len({item.node.planet_id for item in group if item.node.planet_id}) > 1,
                    "transient_bridge": True,
                    "persistent_reparenting": False,
                    "immutable_provenance": True,
                },
            ))
            used_chars += len(text)
        return output

    def snapshot(self) -> dict[str, object]:
        base = super().snapshot()
        base.update({
            "version": "UMD 3.13",
            "relative_gravity": True,
            "gravity_candidate_limit": self.gravity_config.gravity_candidate_limit,
            "tidal_bridges_persisted": False,
            "persistent_orbits_mutated_by_query": False,
            "gravity_formula": self.FORMULA,
        })
        return base
