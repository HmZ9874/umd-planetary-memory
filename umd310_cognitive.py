"""UMD 3.10 cognitive orbits for directives, episodes, and evidence coverage.

This layer targets three structural failures of generic similarity search:

* durable user instructions and preferences can be semantically distant from
  the current task, so they live in sparse directive-satellite indexes;
* ordering and project-progress questions need a chronological episode chain,
  not isolated nearest neighbours;
* summaries and multi-session reasoning need evidence coverage across planets,
  not several near-duplicate turns from one session.

All indexes contain IDs only and are reconstructed from encrypted durable
memory after restart. Raw memory text is not copied into another RAM cache.
"""

from __future__ import annotations

import bisect
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

try:
    from .umd35_core import MemoryNode, cosine, normalized_text, normalized_tokens, utcnow
    from .umd37_planetary import PlanetaryHit
    from .umd38_cosmic import UMD38Config, UMD38CosmicMemory
except ImportError:
    from umd35_core import MemoryNode, cosine, normalized_text, normalized_tokens, utcnow
    from umd37_planetary import PlanetaryHit
    from umd38_cosmic import UMD38Config, UMD38CosmicMemory


DirectiveRole = Literal["instruction", "preference"]

INSTRUCTION_PATTERNS = (
    r"\balways\b", r"\bwhenever\b", r"\bevery time\b", r"\bmust\b",
    r"\bshould\b", r"\bdo not\b", r"\bdon't\b", r"\bplease (?:use|keep|format|answer|reply)",
    r"\bformat (?:all |the )?", r"\brespond (?:with|in)\b", r"\banswer (?:with|in)\b",
    r"总是", r"每次", r"必须", r"不要", r"请用", r"请保持", r"回答时", r"格式化",
)
PREFERENCE_PATTERNS = (
    r"\bi prefer\b", r"\bmy preference\b", r"\bi like\b", r"\bi dislike\b",
    r"\bi want\b", r"\bkeep .* (?:simple|minimal|lightweight|concise)\b",
    r"\bminimal dependenc", r"\blightweight\b", r"\beasy to maintain\b",
    r"我偏好", r"我喜欢", r"我不喜欢", r"我希望", r"尽量简", r"轻量", r"少依赖",
)
TASK_QUERY_PATTERNS = (
    r"\bhow (?:do|can|should|would)\b", r"\bshow me how\b", r"\bimplement\b",
    r"\bwrite (?:a |the )?(?:code|function|class|query)\b", r"\bcode\b",
    r"\brecommend\b", r"\bsuggest\b", r"\bwhat (?:library|libraries|tool|tools|framework)\b",
    r"怎么", r"如何", r"实现", r"代码", r"推荐", r"建议", r"用什么",
)
PREFERENCE_QUERY_PATTERNS = (
    r"\brecommend\b", r"\bsuggest\b", r"\bwhich (?:one|tool|library|framework)\b",
    r"\bwhat (?:should|library|libraries|tool|tools|framework)\b", r"\bchoose\b",
    r"推荐", r"建议", r"选择", r"用什么", r"哪个好",
)
EPISODIC_QUERY_PATTERNS = (
    r"\bin (?:what |which )?order\b", r"\bchronolog", r"\btimeline\b",
    r"\bfirst\b", r"\bthen\b", r"\bbefore\b", r"\bafter\b",
    r"\bthroughout\b", r"\bacross (?:our |the )?(?:conversation|session)",
    r"\bprogress", r"\bcomprehensive summary\b", r"\bsummar",
    r"\bhow many\b", r"\ball (?:the )?", r"\bdifferent aspects\b",
    r"顺序", r"时间线", r"先后", r"之前", r"之后", r"整个过程", r"总结", r"进展", r"多少",
)
CONTRADICTION_QUERY_PATTERNS = (
    r"\bcontradict", r"\bconflict", r"\bwhich (?:is|was) correct\b",
    r"矛盾", r"冲突", r"哪个正确",
)

FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("format", ("format", "syntax", "markdown", "code block", "格式", "高亮")),
    ("dependencies", ("dependenc", "library", "libraries", "framework", "lightweight", "minimal", "依赖", "轻量", "框架")),
    ("response_style", ("concise", "detailed", "bullet", "tone", "reply", "answer", "简洁", "详细", "语气", "回答")),
    ("language", ("english", "chinese", "language", "中文", "英文", "语言")),
    ("privacy", ("private", "privacy", "secret", "redact", "隐私", "秘密", "脱敏")),
    ("workflow", ("always", "whenever", "before", "after", "must", "总是", "每次", "之前", "之后", "必须")),
)


@dataclass
class UMD310Config(UMD38Config):
    directive_reserve: int = 3
    directive_scan_limit: int = 64
    episode_neighbor_radius: int = 2
    candidate_multiplier: int = 8
    candidate_floor: int = 48
    coverage_planet_bonus: float = 0.16
    coverage_time_bonus: float = 0.10
    coverage_entity_bonus: float = 0.08
    redundancy_penalty: float = 0.12
    directive_trusted_sources: tuple[str, ...] = ("user", "user_correction", "verified_tool")
    directive_history_per_family: int = 4
    cold_directive_rebuild_limit: int = 512


@dataclass(frozen=True)
class QueryMode:
    instructions: bool
    preferences: bool
    episodic: bool
    contradiction: bool


def _matches(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)


class UMD310CognitiveMemory(UMD38CosmicMemory):
    """UMD 3.8 runtime plus sparse cognitive routing and coverage selection."""

    def __init__(self, durable, *, config: UMD310Config | None = None, **kwargs) -> None:
        self.cognitive_config = config or UMD310Config()
        self.directive_ids: dict[DirectiveRole, set[str]] = {
            "instruction": set(), "preference": set(),
        }
        self.directive_families: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.episodes_by_star: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self.episodes_by_planet: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        super().__init__(durable, config=self.cognitive_config, **kwargs)
        self._rebuild_cognitive_orbits()

    @staticmethod
    def _directive_role(text: str, requested_kind: str | None = None) -> DirectiveRole | None:
        if requested_kind in {"instruction", "preference"}:
            return requested_kind
        if _matches(text, PREFERENCE_PATTERNS):
            return "preference"
        if _matches(text, INSTRUCTION_PATTERNS):
            return "instruction"
        return None

    @staticmethod
    def _directive_family(text: str) -> str:
        lowered = text.lower()
        for family, cues in FAMILY_PATTERNS:
            if any(cue in lowered for cue in cues):
                return family
        content = [
            token for token in normalized_tokens(text)
            if token not in {"always", "prefer", "please", "should", "must", "want"}
        ]
        return "topic:" + "_".join(content[:2]) if content else "general"

    @staticmethod
    def _query_mode(query: str) -> QueryMode:
        task = _matches(query, TASK_QUERY_PATTERNS)
        preference = _matches(query, PREFERENCE_QUERY_PATTERNS)
        contradiction = _matches(query, CONTRADICTION_QUERY_PATTERNS)
        return QueryMode(
            instructions=task or contradiction,
            preferences=preference or task or contradiction,
            episodic=_matches(query, EPISODIC_QUERY_PATTERNS),
            contradiction=contradiction,
        )

    @staticmethod
    def _active(node: MemoryNode) -> bool:
        return node.state in {"stable", "provisional"} and bool(node.text)

    def _node(self, memory_id: str) -> MemoryNode | None:
        node = self.durable.engine.nodes.get(memory_id)
        if node is not None:
            return node
        try:
            return self.durable.get_memory(memory_id)
        except KeyError:
            return None

    def _family_key(self, node: MemoryNode, role: DirectiveRole) -> tuple[str, str, str]:
        metadata = node.extraction_metadata.get("umd310_cognitive", {})
        family = metadata.get("family") if isinstance(metadata, dict) else None
        return role, node.scope or "*", str(family or self._directive_family(node.text))

    @staticmethod
    def _insert_episode(index: list[tuple[datetime, str]], node: MemoryNode) -> None:
        item = node.created_at, node.id
        position = bisect.bisect_right(index, item)
        if not any(memory_id == node.id for _, memory_id in index[max(0, position - 1): position + 1]):
            index.insert(position, item)

    def _index_node(self, node: MemoryNode) -> None:
        role = self._directive_role(node.text, node.kind)
        if role and node.source in self.cognitive_config.directive_trusted_sources:
            key = self._family_key(node, role)
            family = self.directive_families[key]
            if node.id not in family:
                family.append(node.id)
                family.sort(key=lambda memory_id: (
                    (self._node(memory_id) or node).updated_at, memory_id
                ))
                del family[:-self.cognitive_config.directive_history_per_family]
            if self._active(node):
                self.directive_ids[role].add(node.id)
        if not self._active(node):
            return
        if node.star_id:
            self._insert_episode(self.episodes_by_star[node.star_id], node)
        if node.planet_id:
            self._insert_episode(self.episodes_by_planet[node.planet_id], node)

    def _remove_node(self, memory_id: str, *, preserve_directive_history: bool = False) -> None:
        for values in self.directive_ids.values():
            values.discard(memory_id)
        if not preserve_directive_history:
            for key in list(self.directive_families):
                values = self.directive_families[key]
                self.directive_families[key] = [item for item in values if item != memory_id]
                if not self.directive_families[key]:
                    self.directive_families.pop(key, None)
        for index in (self.episodes_by_star, self.episodes_by_planet):
            for key in list(index):
                index[key] = [item for item in index[key] if item[1] != memory_id]
                if not index[key]:
                    index.pop(key, None)

    def _rebuild_cognitive_orbits(self) -> None:
        self.directive_ids = {"instruction": set(), "preference": set()}
        self.directive_families = defaultdict(list)
        self.episodes_by_star = defaultdict(list)
        self.episodes_by_planet = defaultdict(list)
        for node in self.durable.engine.nodes.values():
            self._index_node(node)
        # Cold records are streamed through a bounded window. Only directive
        # IDs survive this pass; cold node objects and plaintext do not.
        for node in self.durable.cold_history(self.cognitive_config.cold_directive_rebuild_limit):
            if node.id not in self.durable.engine.nodes:
                self._index_node(node)

    def _restore_after_failure(self) -> None:
        super()._restore_after_failure()
        self._rebuild_cognitive_orbits()

    def write(self, text: str, **kwargs):
        role = self._directive_role(text, kwargs.get("kind"))
        source = str(kwargs.get("source", "user"))
        if role and source in self.cognitive_config.directive_trusted_sources:
            kwargs["kind"] = role
            kwargs["explicit_importance"] = max(0.82, float(kwargs.get("explicit_importance", 0.0)))
            metadata = dict(kwargs.pop("extraction_metadata", {}) or {})
            metadata["umd310_cognitive"] = {
                "role": role,
                "family": self._directive_family(text),
                "router_version": 1,
            }
            kwargs["extraction_metadata"] = metadata
        result = super().write(text, **kwargs)
        if result.superseded:
            self._remove_node(result.superseded, preserve_directive_history=True)
        node = self.durable.engine.nodes.get(result.memory_id)
        if node is not None:
            self._index_node(node)
        return result

    def _latest_directives(
        self, mode: QueryMode, scope: str | None,
    ) -> list[MemoryNode]:
        roles: list[DirectiveRole] = []
        if mode.instructions:
            roles.append("instruction")
        if mode.preferences:
            roles.append("preference")
        include_history = mode.contradiction
        nodes: list[MemoryNode] = []
        for key, memory_ids in self.directive_families.items():
            role, family_scope, _ = key
            if role not in roles or (scope and family_scope not in {"*", scope}):
                continue
            usable = []
            for item in memory_ids:
                node = self._node(item)
                if node is None or not self._allowed_node(node):
                    continue
                has_text = bool(node.text or self.durable.get_text(node.id))
                if self._active(node) or (node.state == "archived" and has_text) or (
                    include_history and node.state == "invalidated" and has_text
                ):
                    usable.append(node)
            if usable:
                nodes.extend(usable if include_history else usable[-1:])
        nodes.sort(key=lambda node: (node.updated_at, node.id), reverse=True)
        return nodes[: self.cognitive_config.directive_scan_limit]

    @staticmethod
    def _lexical_overlap(query: str, text: str) -> float:
        query_tokens, text_tokens = set(normalized_tokens(query)), set(normalized_tokens(text))
        return len(query_tokens & text_tokens) / max(1, len(query_tokens))

    def _cognitive_hit(
        self, query: str, node: MemoryNode, *, role_resonance: float = 0.0,
        episode_resonance: float = 0.0, now: datetime | None = None,
    ) -> PlanetaryHit:
        current = now or utcnow()
        text = node.text or self.durable.get_text(node.id)
        query_vector = self.durable.engine.encoder.encode(query)
        node_vector = node.vector if node.vector.size else self.durable.engine.encoder.encode(text)
        semantic = max(0.0, cosine(query_vector, node_vector))
        lexical = self._lexical_overlap(query, text)
        age_days = max(0.0, (current - node.updated_at).total_seconds() / 86400.0)
        recency = math.exp(-age_days / (365.0 if role_resonance else 150.0))
        force = max(0.0, min(1.25,
            0.30 * semantic + 0.18 * lexical + 0.24 * role_resonance
            + 0.12 * episode_resonance + 0.08 * recency + 0.08 * node.source_trust
        ))
        components = {
            "semantic_gravity": semantic,
            "bm25_resonance": lexical,
            "directive_resonance": role_resonance,
            "episode_resonance": episode_resonance,
            "recency_phase": recency,
            "authority": node.source_trust,
        }
        return PlanetaryHit(
            node.id, text, force, 1.0 / (0.05 + force), -node.mass * force,
            components,
            {
                "dominant_force": max(components, key=components.get),
                "formula": "Fc=.30S+.18L+.24D+.12E+.08T+.08A",
                "cognitive_role": node.kind,
                "planet_id": node.planet_id,
                "star_id": node.star_id,
                "created_at": node.created_at.isoformat(),
            },
        )

    def _episode_candidates(
        self, base_hits: list[PlanetaryHit], query: str, scope: str | None,
    ) -> list[PlanetaryHit]:
        candidate_ids: set[str] = set()
        radius = self.cognitive_config.episode_neighbor_radius
        anchor_stars: list[str] = []
        for hit in base_hits[:6]:
            node = self.durable.engine.nodes.get(hit.memory_id)
            if node is None:
                continue
            if node.star_id and node.star_id not in anchor_stars:
                anchor_stars.append(node.star_id)
            if node.planet_id:
                sequence = self.episodes_by_planet.get(node.planet_id, [])
                positions = [index for index, (_, memory_id) in enumerate(sequence) if memory_id == node.id]
                for position in positions:
                    candidate_ids.update(
                        memory_id for _, memory_id in sequence[max(0, position - radius): position + radius + 1]
                    )

        # A broad temporal question receives one strong anchor from each planet
        # in the best matching stars, preventing a single session from filling
        # the entire context window.
        query_vector = self.durable.engine.encoder.encode(query)
        for star_id in anchor_stars[:3]:
            by_planet: dict[str, list[MemoryNode]] = defaultdict(list)
            for _, memory_id in self.episodes_by_star.get(star_id, []):
                node = self.durable.engine.nodes.get(memory_id)
                if node and node.planet_id and self._active(node):
                    by_planet[node.planet_id].append(node)
            for nodes in by_planet.values():
                best = max(nodes, key=lambda item: (
                    max(0.0, cosine(query_vector, item.vector))
                    + 0.35 * self._lexical_overlap(query, item.text),
                    item.updated_at,
                ))
                candidate_ids.add(best.id)

        output = []
        for memory_id in candidate_ids:
            node = self._node(memory_id)
            if (
                node is not None and self._active(node) and self._allowed_node(node)
                and (not scope or not node.scope or node.scope == scope)
            ):
                output.append(self._cognitive_hit(query, node, episode_resonance=1.0))
        return output

    def _contradiction_candidates(
        self, base_hits: list[PlanetaryHit], query: str, scope: str | None,
    ) -> list[PlanetaryHit]:
        candidate_ids: set[str] = set()
        for hit in base_hits[:12]:
            node = self.durable.engine.nodes.get(hit.memory_id)
            if node is None:
                continue
            if node.version_of:
                candidate_ids.add(node.version_of)
            if node.superseded_by:
                candidate_ids.add(node.superseded_by)
            if node.fact_key:
                candidate_ids.update(self.durable.engine.fact_index.get(node.fact_key, ()))
        output = []
        for memory_id in candidate_ids:
            node = self._node(memory_id)
            if (
                node is not None and self._allowed_node(node)
                and node.state in {"stable", "provisional", "invalidated", "archived"}
                and (not scope or not node.scope or node.scope == scope)
                and bool(node.text or self.durable.get_text(node.id))
            ):
                output.append(self._cognitive_hit(query, node, episode_resonance=0.85))
        return output

    def _coverage_select(
        self, hits: list[PlanetaryHit], top_k: int, *, episodic: bool,
        mandatory: list[str],
    ) -> list[PlanetaryHit]:
        by_id: dict[str, PlanetaryHit] = {}
        for hit in hits:
            previous = by_id.get(hit.memory_id)
            if previous is None or hit.force > previous.force:
                by_id[hit.memory_id] = hit
        selected: list[PlanetaryHit] = []
        selected_ids: set[str] = set()
        for memory_id in mandatory:
            hit = by_id.get(memory_id)
            if hit and memory_id not in selected_ids and len(selected) < top_k:
                selected.append(hit)
                selected_ids.add(memory_id)
        if not episodic:
            for hit in sorted(by_id.values(), key=lambda item: (-item.force, item.memory_id)):
                if hit.memory_id not in selected_ids and len(selected) < top_k:
                    selected.append(hit)
                    selected_ids.add(hit.memory_id)
            return selected

        used_planets: set[str] = set()
        used_time_buckets: set[str] = set()
        used_entities: set[str] = set()
        for hit in selected:
            node = self._node(hit.memory_id)
            if node is None:
                continue
            if node.planet_id:
                used_planets.add(node.planet_id)
            used_time_buckets.add(node.created_at.strftime("%Y-%m"))
            used_entities.update(node.entities)
        remaining = {key: value for key, value in by_id.items() if key not in selected_ids}
        while remaining and len(selected) < top_k:
            best_id = ""
            best_value = -1e9
            for memory_id, hit in remaining.items():
                node = self._node(memory_id)
                if node is None:
                    continue
                planet_bonus = float(bool(node.planet_id and node.planet_id not in used_planets))
                time_bucket = node.created_at.strftime("%Y-%m")
                time_bonus = float(time_bucket not in used_time_buckets)
                entity_bonus = len(node.entities - used_entities) / max(1, len(node.entities))
                redundancy = max(
                    (
                        max(0.0, cosine(node.vector, selected_node.vector))
                        for item in selected
                        for selected_node in [self._node(item.memory_id)]
                        if selected_node is not None
                    ),
                    default=0.0,
                )
                value = (
                    hit.force
                    + self.cognitive_config.coverage_planet_bonus * planet_bonus
                    + self.cognitive_config.coverage_time_bonus * time_bonus
                    + self.cognitive_config.coverage_entity_bonus * entity_bonus
                    - self.cognitive_config.redundancy_penalty * redundancy
                )
                if value > best_value or (value == best_value and memory_id < best_id):
                    best_id, best_value = memory_id, value
            if not best_id:
                break
            hit = remaining.pop(best_id)
            node = self._node(best_id)
            if node is None:
                continue
            hit.explanation["coverage_adjusted_force"] = best_value
            selected.append(hit)
            selected_ids.add(best_id)
            if node.planet_id:
                used_planets.add(node.planet_id)
            used_time_buckets.add(node.created_at.strftime("%Y-%m"))
            used_entities.update(node.entities)
        return selected

    def retrieve(
        self, query: str, *, scope: str | None = None, top_k: int = 5,
        now: datetime | None = None, budget_chars: int | None = None,
    ) -> list[PlanetaryHit]:
        if top_k < 1:
            return []
        mode = self._query_mode(query)
        candidate_k = max(
            self.cognitive_config.candidate_floor,
            min(self.config.candidate_limit, top_k * self.cognitive_config.candidate_multiplier),
        )
        base = super().retrieve(
            query, scope=scope, top_k=candidate_k, now=now,
            budget_chars=max(budget_chars or 0, self.durable.engine.config.default_budget_chars * 8),
        )
        extras: list[PlanetaryHit] = []
        directives = self._latest_directives(mode, scope)
        for node in directives:
            extras.append(self._cognitive_hit(query, node, role_resonance=1.0, now=now))
        if mode.episodic:
            extras.extend(self._episode_candidates(base, query, scope))
        if mode.contradiction:
            extras.extend(self._contradiction_candidates(base, query, scope))

        # Preserve one high-confidence factual anchor, then reserve sparse
        # satellite slots. This prevents directives from replacing the task's
        # subject while guaranteeing they reach the answer context.
        mandatory = [base[0].memory_id] if base else []
        mandatory.extend(
            node.id for node in directives[: self.cognitive_config.directive_reserve]
        )
        if mode.contradiction:
            mandatory.extend(hit.memory_id for hit in extras[:2])
        selected = self._coverage_select(
            base + extras, top_k, episodic=mode.episodic, mandatory=mandatory,
        )
        budget = budget_chars or self.durable.engine.config.default_budget_chars
        output: list[PlanetaryHit] = []
        used = 0
        for hit in selected:
            if output and used + len(hit.text) > budget:
                continue
            output.append(hit)
            used += len(hit.text)
        return output

    def chronological_context(self, query: str, *, top_k: int = 12, **kwargs) -> list[PlanetaryHit]:
        """Return the selected evidence in event time order for answer synthesis."""
        hits = self.retrieve(query, top_k=top_k, **kwargs)

        def event_time(hit: PlanetaryHit):
            node = self._node(hit.memory_id)
            return (node.valid_from or node.created_at) if node else utcnow()

        return sorted(
            hits,
            key=lambda hit: (event_time(hit), hit.memory_id),
        )

    def cognitive_index_container_bytes(self) -> int:
        """Approximate owned container overhead without counting shared IDs/text."""
        seen: set[int] = set()

        def owned(value) -> int:
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            if isinstance(value, dict):
                return sys.getsizeof(value) + sum(
                    owned(item) for pair in value.items() for item in pair
                )
            if isinstance(value, (list, set, tuple, defaultdict)):
                return sys.getsizeof(value) + sum(
                    owned(item) for item in value if isinstance(item, (dict, list, set, tuple, defaultdict))
                )
            return 0

        return sum(owned(value) for value in (
            self.directive_ids,
            self.directive_families,
            self.episodes_by_star,
            self.episodes_by_planet,
        ))

    def snapshot(self) -> dict[str, object]:
        base = super().snapshot()
        container_bytes = self.cognitive_index_container_bytes()
        active = max(1, len(self.durable.engine.active_ids))
        base.update({
            "version": "UMD 3.10",
            "formula": (
                "Fc=.30S+.18L+.24D+.12E+.08T+.08A; "
                "Select=F+planet_coverage+time_coverage+entity_coverage-redundancy"
            ),
            "directive_satellites": sum(len(values) for values in self.directive_ids.values()),
            "directive_families": len(self.directive_families),
            "episode_star_chains": len(self.episodes_by_star),
            "episode_planet_chains": len(self.episodes_by_planet),
            "cognitive_index_container_bytes": container_bytes,
            "cognitive_index_container_bytes_per_active_memory": container_bytes / active,
        })
        return base
