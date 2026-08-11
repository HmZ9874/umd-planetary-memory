"""UMD 3.14 adaptive tidal propagation and Lagrange bridge search.

UMD 3.13 introduced transient relative-gravity capture.  This layer removes
three remaining local optima: a source may serve several query attractors,
indirect evidence can receive a bounded two-stage tidal field from direct
evidence, and bridge membership is chosen with beam search instead of a single
greedy step.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime

try:
    from .umd35_core import MemoryNode, cosine, utcnow
    from .umd313_gravity import (
        GravityElement, GravityHit, TIME_RE, UMD313Config, UMD313RelativeGravityMemory,
    )
except ImportError:
    from umd35_core import MemoryNode, cosine, utcnow
    from umd313_gravity import (
        GravityElement, GravityHit, TIME_RE, UMD313Config, UMD313RelativeGravityMemory,
    )


CURRENT_INTENT_RE = re.compile(
    r"\b(?:current|currently|latest|present|presently|now|today|as of now)\b|"
    r"当前|现在|最新|如今|目前",
    re.IGNORECASE,
)
HISTORICAL_INTENT_RE = re.compile(
    r"\b(?:before|previous|previously|prior|formerly|used to|historical|history|"
    r"old (?:value|fact|version|setting)|what was)\b|"
    r"之前|以前|原来|曾经|历史|旧的",
    re.IGNORECASE,
)
NEGATED_QUERY_RE = re.compile(
    r"\b(?:not|never|no|without|didn['’]?t|doesn['’]?t|isn['’]?t|wasn['’]?t)\b|"
    r"不是|没有|未曾|从未|不再|并非",
    re.IGNORECASE,
)
DEEP_GRAPH_QUERY_RE = re.compile(
    r"\b(?:ultimately|associated|through|chain|linked|trace|derive|indirect|"
    r"eventually|relationship path)\b|"
    r"最终|关联|链路|追溯|间接|关系路径",
    re.IGNORECASE,
)
PRONOUN_START_RE = re.compile(
    r"^\s*(?:she|he|they|it|her|him|them|她|他|他们|她们|它)\b",
    re.IGNORECASE,
)
EXPLICIT_SUBJECT_RE = re.compile(r"^\s*([A-Z][a-z]{2,})\b")
DIRECTIVE_FAMILY_QUERY_CUES: dict[str, tuple[str, ...]] = {
    "format": ("code", "snippet", "implement", "syntax", "format", "代码", "实现", "格式"),
    "dependencies": ("library", "libraries", "dependency", "dependencies", "framework", "tool", "库", "依赖", "框架", "工具"),
    "response_style": ("answer", "reply", "response", "explain", "回答", "回复", "解释"),
    "language": ("language", "english", "chinese", "语言", "英文", "中文"),
    "privacy": ("security", "private", "privacy", "secret", "安全", "隐私", "秘密"),
    "workflow": ("how", "implement", "build", "create", "setup", "如何", "实现", "创建", "设置"),
}


@dataclass
class UMD314Config(UMD313Config):
    tidal_seed_per_goal: int = 16
    tidal_decay: float = 0.34
    tidal_second_hop_decay: float = 0.58
    tidal_max_hops: int = 4
    tidal_deep_max_hops: int = 8
    tidal_min_wave: float = 0.006
    tidal_frontier_width: int = 6
    tidal_structural_floor: float = 0.012
    tidal_hub_entity_fraction: float = 0.25
    tidal_hub_min_degree: int = 8
    cold_candidate_scan_limit: int = 512
    cold_candidate_preselect: int = 64
    adaptive_mad_scale: float = 0.35
    adaptive_min_potential: float = 0.015
    adaptive_capture_floor: float = 0.46
    adaptive_reserved_per_goal: int = 4
    multi_attractor_window: float = 0.075
    lagrange_beam_width: int = 24
    lagrange_branch_width: int = 36
    lagrange_budget_penalty: float = 0.14
    lagrange_ambiguity_penalty: float = 0.08
    lagrange_cohesion_bonus: float = 0.18
    bridge_absolute_gate: float = 0.05
    bridge_relative_gate: float = 0.42
    bridge_min_results: int = 3
    directive_reserve: int = 3
    directive_field_bonus: float = 0.24
    directive_family_bonus: float = 0.14
    episodic_planet_reserve: int = 12
    episodic_planet_field_bonus: float = 0.06


@dataclass(frozen=True)
class _AdaptiveCandidate:
    node: MemoryNode
    text: str
    goals: frozenset[int]
    direct_potentials: tuple[float, ...]
    field_potentials: tuple[float, ...]
    capture_probabilities: tuple[float, ...]
    elements: tuple[GravityElement, ...]
    entities: frozenset[str]
    entity_weights: tuple[tuple[str, float], ...]
    time_keys: frozenset[str]
    structural_goals: frozenset[int]

    @property
    def best_potential(self) -> float:
        return max(self.field_potentials[index] for index in self.goals)

    @property
    def confidence(self) -> float:
        return max(self.capture_probabilities[index] for index in self.goals)


@dataclass(frozen=True)
class _BeamState:
    members: tuple[_AdaptiveCandidate, ...]
    score: float


class UMD314LagrangeMemory(UMD313RelativeGravityMemory):
    """Relative gravity with multi-attractor propagation and bounded search."""

    FIELD_FORMULA = (
        "F_g=Delta_g+sum_h(.34*.58^(h-1)*max_j(link(i,j)*wave_h(j))), h<=H(query)"
    )
    LAGRANGE_FORMULA = (
        "L=field+.20adaptive_goal+.18cohesion+.12cross_planet+.07time+"
        ".05source+.24directive+.06episode_coverage-.18redundancy-.14budget-.08ambiguity"
    )

    def __init__(self, durable, *, config: UMD314Config | None = None, **kwargs) -> None:
        self.lagrange_config = config or UMD314Config()
        super().__init__(durable, config=self.lagrange_config, **kwargs)

    def _eligible(self, node: MemoryNode, *, history: bool = False) -> bool:
        if not super()._eligible(node, history=history):
            return False
        if history or node.state != "provisional" or not node.version_of:
            return True
        predecessor = self.durable.engine.nodes.get(node.version_of)
        if predecessor is None:
            try:
                predecessor = self.durable.get_memory(node.version_of)
            except (KeyError, PermissionError):
                predecessor = None
        # A lower-authority contradiction remains auditable but cannot enter a
        # current-state field until explicitly confirmed or independently won.
        return not (
            predecessor is not None
            and predecessor.state == "stable"
            and predecessor.fact_key == node.fact_key
            and predecessor.source_trust > node.source_trust
        )

    def _effective_tidal_hops(self, query: str) -> int:
        base = max(1, self.lagrange_config.tidal_max_hops)
        if DEEP_GRAPH_QUERY_RE.search(query):
            return max(base, self.lagrange_config.tidal_deep_max_hops)
        return base

    @staticmethod
    def _query_negated(query: str) -> bool:
        return bool(NEGATED_QUERY_RE.search(query))

    @staticmethod
    def _text_negated(text: str) -> bool:
        return bool(NEGATED_QUERY_RE.search(text))

    @staticmethod
    def _ambiguous_coreferences(
        nodes: list[MemoryNode], texts: dict[str, str],
    ) -> set[str]:
        """Find unresolved pronouns with multiple recent explicit subjects."""
        ordered = sorted(nodes, key=lambda node: (node.created_at, node.id))
        ambiguous: set[str] = set()
        for index, node in enumerate(ordered):
            if not PRONOUN_START_RE.search(texts[node.id]):
                continue
            antecedents: set[str] = set()
            for previous in ordered[max(0, index - 24):index]:
                if node.scope and previous.scope and node.scope != previous.scope:
                    continue
                match = EXPLICIT_SUBJECT_RE.search(texts[previous.id])
                if match and match.group(1).casefold() not in {"project", "correction", "the"}:
                    antecedents.add(match.group(1).casefold())
            if len(antecedents) >= 2:
                ambiguous.add(node.id)
        return ambiguous

    def _directive_applicability(self, query: str, node: MemoryNode) -> float:
        metadata = node.extraction_metadata.get("umd310_cognitive", {})
        family = metadata.get("family") if isinstance(metadata, dict) else None
        family = str(family or self._directive_family(node.text))
        lowered = query.casefold()
        family_match = any(
            cue in lowered for cue in DIRECTIVE_FAMILY_QUERY_CUES.get(family, ())
        )
        lexical = self._lexical_overlap(query, node.text or self.durable.get_text(node.id))
        role_floor = 0.45 if node.kind == "instruction" else 0.30
        return min(1.0, max(role_floor, 0.58 * lexical + 0.42 * float(family_match)))

    @staticmethod
    def _rarity(entities: set[str], frequencies: Counter[str]) -> float:
        if not entities:
            return 0.0
        return sum(1.0 / math.log2(2.0 + frequencies[item]) for item in entities) / len(entities)

    def _direct_elements(
        self, node: MemoryNode, text: str, goal: str, vector, now: datetime,
        query_entities: set[str], query_time: set[str], frequencies: Counter[str],
    ) -> GravityElement:
        base = self._score_node_goal(
            node, text, goal, vector, now, query_entities, query_time, set(),
        )
        entities = {item.casefold() for item in node.entities} | self._entities(text)
        novelty = self._rarity(entities, frequencies)
        attraction = base.raw_attraction - 0.05 * base.novelty + 0.05 * novelty
        net = (
            attraction
            - self.lagrange_config.gravity_home_inertia * base.home_inertia
            - self.lagrange_config.gravity_migration_cost * base.migration_cost
            - self.lagrange_config.gravity_uncertainty_cost * base.uncertainty
        )
        probability = self._sigmoid(net / self.lagrange_config.gravity_capture_temperature)
        return replace(
            base, novelty=novelty, raw_attraction=attraction,
            net_potential=net, capture_probability=probability,
        )

    def _tidal_link(self, left, right) -> float:
        left_entities = set(left.entities) if hasattr(left, "entities") else (
            {item.casefold() for item in left.node.entities} | self._entities(left.text)
        )
        right_entities = set(right.entities) if hasattr(right, "entities") else (
            {item.casefold() for item in right.node.entities} | self._entities(right.text)
        )
        left_weights = dict(getattr(left, "entity_weights", ()))
        right_weights = dict(getattr(right, "entity_weights", ()))
        shared = left_entities & right_entities
        shared_weight = sum(
            min(left_weights.get(item, 1.0), right_weights.get(item, 1.0))
            for item in shared
        )
        left_mass = sum(left_weights.get(item, 1.0) for item in left_entities)
        right_mass = sum(right_weights.get(item, 1.0) for item in right_entities)
        entity = (
            shared_weight / max(1e-9, min(left_mass, right_mass))
            * min(1.0, shared_weight)
        )
        left_relations = self._relation_terms(left.node)
        right_relations = self._relation_terms(right.node)
        relation = len(left_relations & right_relations) / max(
            1, min(len(left_relations), len(right_relations))
        )
        semantic = 0.0
        if left.node.vector.size and right.node.vector.size:
            semantic = max(0.0, cosine(left.node.vector, right.node.vector))
        # Pure semantic similarity cannot create a strong bridge by itself;
        # shared entities or extracted relations must carry most of the field.
        structural = max(entity, relation)
        return min(1.0, 0.82 * structural + 0.18 * semantic * structural)

    @staticmethod
    def _adaptive_threshold(values: list[float], config: UMD314Config) -> float:
        if not values:
            return float("inf")
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        return max(config.adaptive_min_potential, median + config.adaptive_mad_scale * mad)

    @staticmethod
    def _historical_query(query: str) -> bool:
        # Explicit current-state intent has precedence. Bare "Old" is not a
        # temporal instruction because it is common inside proper names.
        if CURRENT_INTENT_RE.search(query):
            return False
        return bool(HISTORICAL_INTENT_RE.search(query))

    def _cold_candidates(
        self, query: str, scope: str | None, hot_ids: set[str],
    ) -> list[MemoryNode]:
        historical = self._historical_query(query)
        query_entities = self._entities(query)
        try:
            indexed = self.durable.search_cold(
                query, max(
                    self.lagrange_config.cold_candidate_preselect * 4,
                    self.lagrange_config.cold_candidate_preselect,
                ),
            )
            recent = self.durable.cold_history(
                min(
                    self.lagrange_config.cold_candidate_scan_limit,
                    self.lagrange_config.cold_candidate_preselect * 2,
                )
            )
            cold = list({node.id: node for node in (*indexed, *recent)}.values())
        except (AttributeError, PermissionError):
            try:
                cold = self.durable.cold_history(
                    self.lagrange_config.cold_candidate_scan_limit
                )
            except (AttributeError, PermissionError):
                return []
        scored: list[tuple[float, MemoryNode]] = []
        for node in cold:
            if node.id in hot_ids:
                continue
            if node.state == "invalidated" and not historical:
                continue
            if node.state not in {"archived", "invalidated"}:
                continue
            if scope and node.scope and node.scope != scope:
                continue
            if not self._eligible(node, history=True):
                continue
            text = node.text or self.durable.get_text(node.id)
            entities = {item.casefold() for item in node.entities} | self._entities(text)
            entity = len(query_entities & entities) / max(1, len(query_entities))
            lexical = self._lexical_overlap(query, text)
            history_bonus = float(historical and node.state == "invalidated")
            score = 0.62 * lexical + 0.28 * entity + 0.10 * history_bonus
            scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [node for _, node in scored[: self.lagrange_config.cold_candidate_preselect]]

    def _adaptive_candidates(
        self, query: str, scope: str | None, now: datetime,
    ) -> tuple[list[str], list[_AdaptiveCandidate], list[float], dict[str, object]]:
        goals = self._gravity_subgoals(query)
        vectors = [self.durable.engine.encoder.encode(goal) for goal in goals]
        query_entities = self._entities(query)
        query_time = {item.casefold() for item in self._time_keys_from_query(query)}
        mode = self._query_mode(query)
        effective_hops = self._effective_tidal_hops(query)
        structural_floor = self.lagrange_config.tidal_structural_floor
        if effective_hops > self.lagrange_config.tidal_max_hops:
            # Deep paths have multiplicative attenuation. Lower only the
            # structural continuation gate, while hub limits and beam budgets
            # continue to bound fan-out and memory use.
            structural_floor *= 0.30
        query_negated = self._query_negated(query)
        nodes = [
            node for node in self.durable.engine.nodes.values()
            if self._eligible(node) and (not scope or not node.scope or node.scope == scope)
        ]
        hot_ids = {node.id for node in nodes}
        nodes.extend(self._cold_candidates(query, scope, hot_ids))
        # Directive satellites are sparse durable control memories. They must
        # enter the relative-gravity field even when lexical cold preselection
        # would miss them.
        directive_pool = self._latest_directives(mode, scope)
        by_id = {node.id: node for node in nodes}
        for node in directive_pool:
            if node.id not in by_id and node.state != "invalidated":
                by_id[node.id] = node
        nodes = list(by_id.values())
        texts = {node.id: node.text or self.durable.get_text(node.id) for node in nodes}
        directive_ranked = sorted(
            (
                (self._directive_applicability(query, node), node)
                for node in directive_pool if node.id in texts
            ),
            key=lambda item: (-item[0], -item[1].updated_at.timestamp(), item[1].id),
        )
        directive_priority = {
            node.id: applicability
            for applicability, node in directive_ranked[: self.lagrange_config.directive_reserve]
        }
        ambiguous_ids = self._ambiguous_coreferences(nodes, texts)
        if query_negated:
            # A positive assertion cannot answer a negated proposition. Keep
            # only explicit negative evidence rather than returning a lexical
            # match that means the opposite of the question.
            nodes = [node for node in nodes if self._text_negated(texts[node.id])]
            texts = {node.id: texts[node.id] for node in nodes}
            ambiguous_ids.intersection_update(texts)
        policy = {
            "effective_tidal_hops": effective_hops,
            "query_negation_handled": query_negated,
            "coreference_ambiguous": bool(ambiguous_ids),
            "ambiguous_source_ids": tuple(sorted(ambiguous_ids)),
            "directive_satellites_reserved": tuple(sorted(directive_priority)),
            "episodic_query": mode.episodic,
        }
        if not nodes:
            return goals, [], [], policy
        entity_sets = {
            node.id: ({item.casefold() for item in node.entities} | self._entities(texts[node.id]))
            for node in nodes
        }
        frequencies: Counter[str] = Counter(
            entity for values in entity_sets.values() for entity in values
        )
        direct: dict[str, tuple[GravityElement, ...]] = {}
        for node in nodes:
            elements = tuple(
                self._direct_elements(
                    node, texts[node.id], goal, vector, now,
                    query_entities, query_time, frequencies,
                )
                for goal, vector in zip(goals, vectors)
            )
            applicability = directive_priority.get(node.id)
            if applicability is not None:
                bonus = (
                    self.lagrange_config.directive_field_bonus
                    + self.lagrange_config.directive_family_bonus * applicability
                )
                elements = tuple(replace(
                    item,
                    raw_attraction=item.raw_attraction + bonus,
                    net_potential=item.net_potential + bonus,
                    capture_probability=self._sigmoid(
                        (item.net_potential + bonus)
                        / self.lagrange_config.gravity_capture_temperature
                    ),
                ) for item in elements)
            direct[node.id] = elements

        lightweight = [
            _AdaptiveCandidate(
                node, texts[node.id], frozenset(),
                tuple(item.net_potential for item in direct[node.id]),
                tuple(item.net_potential for item in direct[node.id]),
                tuple(item.capture_probability for item in direct[node.id]),
                direct[node.id], frozenset(entity_sets[node.id]),
                tuple(sorted(
                    (entity, 1.0 / math.log2(2.0 + frequencies[entity]))
                    for entity in entity_sets[node.id]
                )),
                frozenset(self._time_keys(texts[node.id], node)),
                frozenset(),
            )
            for node in nodes
        ]
        entity_index: dict[str, set[int]] = {}
        relation_index: dict[str, set[int]] = {}
        relation_sets: list[set[str]] = []
        for index, candidate in enumerate(lightweight):
            for entity in candidate.entities:
                entity_index.setdefault(entity, set()).add(index)
            relations = self._relation_terms(candidate.node)
            relation_sets.append(relations)
            for relation in relations:
                relation_index.setdefault(relation, set()).add(index)
        hub_limit = max(
            self.lagrange_config.tidal_hub_min_degree,
            int(len(lightweight) * self.lagrange_config.tidal_hub_entity_fraction),
        )
        neighbor_cache: dict[int, tuple[int, ...]] = {}

        def structural_neighbors(parent: int) -> tuple[int, ...]:
            cached = neighbor_cache.get(parent)
            if cached is not None:
                return cached
            values: set[int] = set()
            for entity in lightweight[parent].entities:
                linked = entity_index.get(entity, set())
                if len(linked) <= hub_limit:
                    values.update(linked)
            for relation in relation_sets[parent]:
                linked = relation_index.get(relation, set())
                if len(linked) <= hub_limit:
                    values.update(linked)
            values.discard(parent)
            cached = tuple(sorted(values))
            neighbor_cache[parent] = cached
            return cached

        fields = [list(item.direct_potentials) for item in lightweight]
        structural_memberships: list[set[int]] = [set() for _ in goals]
        directive_indices = {
            index for index, item in enumerate(lightweight)
            if item.node.id in directive_priority
        }
        episodic_indices: set[int] = set()
        if mode.episodic:
            best_by_planet: dict[str, int] = {}
            for index, item in enumerate(lightweight):
                planet_id = item.node.planet_id
                if not planet_id:
                    continue
                current = best_by_planet.get(planet_id)
                if current is None or (
                    max(item.direct_potentials), item.node.id
                ) > (
                    max(lightweight[current].direct_potentials),
                    lightweight[current].node.id,
                ):
                    best_by_planet[planet_id] = index
            episodic_indices = set(sorted(
                best_by_planet.values(),
                key=lambda index: (
                    -max(lightweight[index].direct_potentials),
                    lightweight[index].node.id,
                ),
            )[: self.lagrange_config.episodic_planet_reserve])
            for index in episodic_indices:
                best_goal = max(
                    range(len(goals)), key=lambda goal: fields[index][goal]
                )
                fields[index][best_goal] += self.lagrange_config.episodic_planet_field_bonus
        policy["episodic_planets_reserved"] = len(episodic_indices)
        for goal in range(len(goals)):
            structural_memberships[goal].update(directive_indices)
            structural_memberships[goal].update(
                index for index in episodic_indices
                if goal == max(range(len(goals)), key=lambda item: fields[index][item])
            )
            wave = [max(0.0, item.direct_potentials[goal]) for item in lightweight]
            for hop in range(1, effective_hops + 1):
                seed_indices = sorted(
                    range(len(lightweight)),
                    key=lambda index: (-wave[index], -fields[index][goal], lightweight[index].node.id),
                )[: self.lagrange_config.tidal_seed_per_goal]
                next_wave = [0.0] * len(lightweight)
                for seed in seed_indices:
                    if wave[seed] <= 0.0:
                        continue
                    for index in structural_neighbors(seed):
                        propagated = (
                            self._tidal_link(lightweight[index], lightweight[seed]) * wave[seed]
                        )
                        if propagated > next_wave[index]:
                            next_wave[index] = propagated
                for index, propagated in enumerate(next_wave):
                    fields[index][goal] += (
                        self.lagrange_config.tidal_decay
                        * self.lagrange_config.tidal_second_hop_decay ** (hop - 1)
                        * propagated
                    )
                wave = next_wave
                if max(wave, default=0.0) < self.lagrange_config.tidal_min_wave:
                    break

            # A narrow structural frontier prevents a strong lexical crowd
            # from erasing a rare multi-hop chain at the final MAD threshold.
            anchors = sorted(
                range(len(lightweight)),
                key=lambda index: (-lightweight[index].direct_potentials[goal], lightweight[index].node.id),
            )[: self.lagrange_config.adaptive_reserved_per_goal]
            # Anchors are part of the structural path by definition. Without
            # this reservation, later propagated scores can displace a bridge
            # anchor from the final MAD ranking and create a broken path.
            structural_memberships[goal].update(anchors)
            frontier = list(anchors)
            visited = set(anchors)
            for hop in range(1, effective_hops + 1):
                options: dict[int, float] = {}
                reserved_children: dict[int, float] = {}
                for parent in frontier:
                    parent_options = []
                    for index in structural_neighbors(parent):
                        if index in visited:
                            continue
                        strength = (
                            self._tidal_link(lightweight[index], lightweight[parent])
                            * max(0.0, fields[parent][goal])
                        )
                        if strength >= structural_floor:
                            parent_options.append((strength, index))
                            options[index] = max(options.get(index, 0.0), strength)
                    parent_options.sort(
                        key=lambda item: (-item[0], lightweight[item[1]].node.id)
                    )
                    if parent_options:
                        strength, index = parent_options[0]
                        reserved_children[index] = max(reserved_children.get(index, 0.0), strength)
                chosen_map = dict(reserved_children)
                ranked_options = sorted(
                    options.items(), key=lambda item: (-item[1], lightweight[item[0]].node.id)
                )
                target_width = max(
                    self.lagrange_config.tidal_frontier_width, len(chosen_map)
                )
                for index, strength in ranked_options:
                    if len(chosen_map) >= target_width:
                        break
                    chosen_map[index] = strength
                chosen = sorted(
                    ((strength, index) for index, strength in chosen_map.items()),
                    key=lambda item: (-item[0], lightweight[item[1]].node.id),
                )
                if not chosen:
                    break
                frontier = [index for _, index in chosen]
                for strength, index in chosen:
                    visited.add(index)
                    structural_memberships[goal].add(index)
                    fields[index][goal] += (
                        self.lagrange_config.tidal_decay
                        * self.lagrange_config.tidal_second_hop_decay ** (hop - 1)
                        * strength
                    )

        thresholds = [
            self._adaptive_threshold([row[goal] for row in fields], self.lagrange_config)
            for goal in range(len(goals))
        ]
        memberships: dict[str, set[int]] = {node.id: set() for node in nodes}
        probabilities: list[list[float]] = [[0.0] * len(goals) for _ in nodes]
        for goal in range(len(goals)):
            order = sorted(
                range(len(nodes)), key=lambda index: (-fields[index][goal], nodes[index].id)
            )
            best = fields[order[0]][goal] if order else 0.0
            considered = list(order[: self.lagrange_config.gravity_candidates_per_goal])
            considered.extend(sorted(
                structural_memberships[goal] - set(considered),
                key=lambda index: (-fields[index][goal], nodes[index].id),
            ))
            rank_by_index = {index: rank for rank, index in enumerate(order)}
            for index in considered:
                rank = rank_by_index[index]
                probability = self._sigmoid(
                    (fields[index][goal] - thresholds[goal])
                    / self.lagrange_config.gravity_capture_temperature
                )
                probabilities[index][goal] = probability
                within_relative_field = fields[index][goal] >= best - self.lagrange_config.multi_attractor_window
                reserved = (
                    rank < self.lagrange_config.adaptive_reserved_per_goal
                    and fields[index][goal] >= self.lagrange_config.adaptive_min_potential
                )
                if index in structural_memberships[goal] or reserved or (
                    fields[index][goal] >= thresholds[goal]
                    and probability >= self.lagrange_config.adaptive_capture_floor
                ) or within_relative_field:
                    memberships[nodes[index].id].add(goal)

        candidates = []
        for index, item in enumerate(lightweight):
            goals_for_item = memberships[item.node.id]
            if not goals_for_item:
                continue
            candidates.append(replace(
                item, goals=frozenset(goals_for_item),
                field_potentials=tuple(fields[index]),
                capture_probabilities=tuple(probabilities[index]),
                structural_goals=frozenset(
                    goal for goal in goals_for_item
                    if index in structural_memberships[goal]
                ),
            ))
        candidates.sort(key=lambda item: (-item.best_potential, item.node.id))
        return goals, candidates[: self.lagrange_config.gravity_candidate_limit], thresholds, policy

    @staticmethod
    def _time_keys_from_query(query: str) -> set[str]:
        # Reuse the parent's matcher without inventing a synthetic node date.
        return {item.casefold() for item in TIME_RE.findall(query)}

    @staticmethod
    def _semantic_redundancy(group: tuple[_AdaptiveCandidate, ...]) -> float:
        values = []
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if left.node.vector.size and right.node.vector.size:
                    values.append(max(0.0, cosine(left.node.vector, right.node.vector)))
        return sum(values) / max(1, len(values))

    def _cohesion(self, group: tuple[_AdaptiveCandidate, ...]) -> float:
        if len(group) < 2:
            return 0.0
        # Maximum-link attachment is a small maximum-spanning-tree surrogate.
        connected = [group[0]]
        remaining = list(group[1:])
        links = []
        while remaining:
            value, chosen = max(
                (
                    (max(self._tidal_link(item, candidate) for item in connected), candidate)
                    for candidate in remaining
                ),
                key=lambda pair: (pair[0], pair[1].node.id),
            )
            connected.append(chosen)
            remaining.remove(chosen)
            links.append(value)
        return sum(links) / max(1, len(links))

    def _state_score(
        self, members: tuple[_AdaptiveCandidate, ...], goal_count: int,
        used_goals: set[int], used_times: set[str], used_sources: set[str], budget: int,
    ) -> float:
        potentials = [item.best_potential for item in members]
        covered = set().union(*(item.goals for item in members))
        new_goals = covered - used_goals
        uncovered_fraction = (goal_count - len(used_goals)) / max(1, goal_count)
        adaptive_goal = len(new_goals) / max(1, goal_count) * (1.0 + uncovered_fraction)
        planets = {item.node.planet_id for item in members if item.node.planet_id}
        cross_planet = (len(planets) - 1) / max(1, len(members) - 1)
        times = set().union(*(item.time_keys for item in members))
        new_time = float(bool(times - used_times))
        new_source = sum(item.node.id not in used_sources for item in members) / len(members)
        cohesion = self._cohesion(members)
        # Similarity is useful when two memories are structurally linked and
        # complementary. Penalize only the unexplained part of similarity.
        redundancy = self._semantic_redundancy(members) * (1.0 - cohesion)
        ambiguity = sum(1.0 - item.confidence for item in members) / len(members)
        chars = sum(len(item.text) + len(item.node.id) + 10 for item in members)
        budget_pressure = chars / max(1, budget)
        return (
            sum(potentials) / len(potentials)
            + self.lagrange_config.gravity_new_goal_bonus * adaptive_goal
            + self.lagrange_config.lagrange_cohesion_bonus * cohesion
            + self.lagrange_config.gravity_cross_planet_bonus * cross_planet
            + self.lagrange_config.gravity_time_phase_bonus * new_time
            + self.lagrange_config.gravity_source_bonus * new_source
            - self.lagrange_config.gravity_redundancy_penalty * redundancy
            - self.lagrange_config.lagrange_budget_penalty * budget_pressure
            - self.lagrange_config.lagrange_ambiguity_penalty * ambiguity
        )

    def _beam_bridge(
        self, candidates: list[_AdaptiveCandidate], goal_count: int,
        used_goals: set[int], used_times: set[str], used_sources: set[str], budget: int,
    ) -> _BeamState | None:
        if not candidates:
            return None
        branch_candidates = candidates[: self.lagrange_config.lagrange_branch_width]
        # Guarantee that every still-uncovered goal has representatives in the
        # expansion set even when one easy attractor dominates raw relevance.
        for goal in set(range(goal_count)) - used_goals:
            per_goal = [item for item in candidates if goal in item.goals]
            per_goal.sort(key=lambda item: (-item.field_potentials[goal], item.node.id))
            branch_candidates.extend(per_goal[:4])
        branch_candidates = list({item.node.id: item for item in branch_candidates}.values())
        beam = [_BeamState((), float("-inf"))]
        complete: list[_BeamState] = []
        max_sources = min(
            self.lagrange_config.capsule_max_sources,
            self.lagrange_config.gravity_sources_per_bridge,
        )
        for _ in range(max_sources):
            expanded: dict[tuple[str, ...], _BeamState] = {}
            for state in beam:
                member_ids = {item.node.id for item in state.members}
                for candidate in branch_candidates:
                    if candidate.node.id in member_ids:
                        continue
                    members = tuple(sorted((*state.members, candidate), key=lambda item: item.node.id))
                    score = self._state_score(
                        members, goal_count, used_goals, used_times, used_sources, budget,
                    )
                    key = tuple(item.node.id for item in members)
                    current = expanded.get(key)
                    if current is None or score > current.score:
                        expanded[key] = _BeamState(members, score)
            beam = sorted(
                expanded.values(),
                key=lambda state: (-state.score, tuple(item.node.id for item in state.members)),
            )[: self.lagrange_config.lagrange_beam_width]
            complete.extend(beam)
            if not beam:
                break
        return max(
            complete,
            key=lambda state: (
                state.score,
                len(set().union(*(item.goals for item in state.members))),
                -len(state.members),
            ),
            default=None,
        )

    def retrieve_gravity(
        self, query: str, *, scope: str | None = None, top_k: int = 10,
        budget_chars: int | None = None, now: datetime | None = None,
    ) -> list[GravityHit]:
        if top_k < 1:
            return []
        current = now or utcnow()
        goals, candidates, thresholds, query_policy = self._adaptive_candidates(
            query, scope, current,
        )
        if not candidates:
            return []
        budget = budget_chars or self.durable.engine.config.default_budget_chars
        remaining_budget = budget
        used_goals: set[int] = set()
        used_times: set[str] = set()
        used_sources: set[str] = set()
        output: list[GravityHit] = []
        remaining = list(candidates)
        best_bridge_score: float | None = None
        while remaining and len(output) < top_k and remaining_budget > 0:
            state = self._beam_bridge(
                remaining, len(goals), used_goals, used_times, used_sources, remaining_budget,
            )
            if state is None:
                break
            if best_bridge_score is None:
                best_bridge_score = state.score
            bridge_gate = max(
                self.lagrange_config.bridge_absolute_gate,
                self.lagrange_config.bridge_relative_gate * best_bridge_score,
            )
            deep_completion = (
                query_policy["effective_tidal_hops"] > self.lagrange_config.tidal_max_hops
                and any(item.structural_goals for item in state.members)
            )
            if (
                len(output) >= self.lagrange_config.bridge_min_results
                and state.score < bridge_gate
                and not deep_completion
            ):
                break
            original_members = tuple(state.members)
            fitting = list(state.members)
            while fitting:
                text = "\n".join(f"[source:{item.node.id}] {item.text}" for item in fitting)
                if len(text) <= remaining_budget:
                    break
                fitting.remove(min(fitting, key=lambda item: (item.best_potential, item.node.id)))
            budget_truncated = False
            if not fitting:
                best = max(
                    original_members,
                    key=lambda item: (item.best_potential, item.confidence, item.node.id),
                )
                fitting = [best]
                raw = f"[source:{best.node.id}] {best.text}"
                text = raw[:remaining_budget]
                budget_truncated = True
            members = tuple(fitting)
            source_ids = tuple(item.node.id for item in members)
            payload = "tidal_lagrange\0" + "\0".join(source_ids)
            bridge_id = "lag_" + hashlib.sha256(payload.encode()).hexdigest()[:20]
            if not budget_truncated:
                text = "\n".join(f"[source:{item.node.id}] {item.text}" for item in members)
            covered = set().union(*(item.goals for item in members))
            planets = {item.node.planet_id for item in members if item.node.planet_id}
            output.append(GravityHit(
                bridge_id, source_ids, text, state.score,
                {
                    "direct_potential": sum(max(item.direct_potentials) for item in members) / len(members),
                    "tidal_field": sum(item.best_potential for item in members) / len(members),
                    "capture_probability": sum(item.confidence for item in members) / len(members),
                    "cohesion": self._cohesion(members),
                    "budget_pressure": len(text) / max(1, remaining_budget),
                },
                {
                    "formula": self.FIELD_FORMULA,
                    "optimizer": self.LAGRANGE_FORMULA,
                    "query_attractors": goals,
                    "adaptive_thresholds": thresholds,
                    "captured_goal_indices": sorted(covered),
                    "deep_path_completion": deep_completion,
                    "multi_attractor_sources": sum(len(item.goals) > 1 for item in members),
                    "home_planets": sorted(planets),
                    "cross_planet": len(planets) > 1,
                    "tidal_propagation_hops": query_policy["effective_tidal_hops"],
                    "query_negation_handled": query_policy["query_negation_handled"],
                    "directive_satellites_reserved": query_policy["directive_satellites_reserved"],
                    "episodic_planets_reserved": query_policy["episodic_planets_reserved"],
                    "coreference_ambiguous": query_policy["coreference_ambiguous"],
                    "ambiguous_source_ids": query_policy["ambiguous_source_ids"],
                    "selection_algorithm": "bounded_beam_lagrange",
                    "budget_atomic_excerpt": budget_truncated,
                    "transient_bridge": True,
                    "persistent_reparenting": False,
                    "immutable_provenance": True,
                },
            ))
            remaining_budget -= len(text)
            member_ids = set(source_ids)
            remaining = [item for item in remaining if item.node.id not in member_ids]
            used_goals.update(covered)
            used_sources.update(member_ids)
            used_times.update(*(item.time_keys for item in members))
        return output

    def snapshot(self) -> dict[str, object]:
        base = super().snapshot()
        base.update({
            "version": "UMD 3.14",
            "tidal_propagation_hops": self.lagrange_config.tidal_max_hops,
            "multi_attractor_capture": True,
            "adaptive_capture_threshold": "median_plus_mad",
            "bridge_optimizer": "bounded_beam_lagrange",
            "lagrange_beam_width": self.lagrange_config.lagrange_beam_width,
            "persistent_orbits_mutated_by_query": False,
        })
        return base
