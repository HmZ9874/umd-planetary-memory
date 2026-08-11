"""UMD 3.5: an online, bounded, provenance-aware planetary memory engine.

This module is the algorithm itself, not a benchmark adapter.  It accepts raw
text, builds a star -> planet -> memory hierarchy, versions contradictions,
forms entity relations, consolidates duplicates, applies bounded decay, and
retrieves with an explainable hierarchical score.

The text encoder is a deterministic signed feature hash.  It is intentionally
small and dependency-light; a neural encoder can later implement the same
``encode`` interface without changing the memory lifecycle.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Literal, Protocol

import numpy as np

try:
    from .umd35_storage import ArchiveStore, InMemoryArchiveStore
except ImportError:  # Direct script/module execution.
    from umd35_storage import ArchiveStore, InMemoryArchiveStore


MemoryState = Literal["stable", "provisional", "quarantined", "floating", "invalidated", "archived"]
WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")
CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
NEGATIONS = {"not", "never", "no", "without", "不是", "不再", "没有", "取消"}
CORRECTION_CUES = {"actually", "correction", "correct", "changed", "instead", "更正", "改为", "其实", "更新"}
FUTURE_CUES = {"remember", "remind", "later", "deadline", "tomorrow", "以后", "记住", "提醒", "截止", "将来"}
PREFERENCE_CUES = {"prefer", "favorite", "like", "dislike", "偏好", "喜欢", "讨厌", "最爱"}
INJECTION_CUES = (
    "ignore previous", "ignore all", "system prompt", "reveal secret",
    "developer message", "override instructions", "忽略之前", "系统提示词",
    "泄露密钥", "覆盖指令",
)
INJECTION_COMPACT_CUES = (
    "ignoreprevious", "ignoreall", "ignorepriorguidance", "ignorepriorinstructions",
    "systemprompt", "revealsecret", "exposehiddenconfiguration", "exposehiddenprompt",
    "developermessage", "overrideinstructions",
)
# Security-only skeleton for common Cyrillic and Greek characters that are
# visually confusable with Latin injection cues. This does not alter stored
# text, entity identity, embeddings, or user-visible content.
SECURITY_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "і": "i", "ј": "j", "ѕ": "s", "к": "k", "м": "m",
    "т": "t", "в": "b", "н": "h",
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ν": "v",
})
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "i", "in", "is", "it", "my", "of", "on", "or", "the", "this",
    "to", "was", "we", "with", "you", "your",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if token.lower() not in STOP]


def normalized_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def has_cue(text: str, cues: Iterable[str]) -> bool:
    """Match Latin cues by token and CJK/phrase cues by substring."""
    lower = text.lower()
    tokens = set(normalized_tokens(text))
    return any(
        (cue in lower if " " in cue or re.search(r"[\u3400-\u9fff]", cue) else cue in tokens)
        for cue in cues
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right)) if left.size and right.size and left.shape == right.shape else 0.0


def bounded_sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class HashEncoder:
    """Small signed hashing encoder with word, bigram, and character features."""

    def __init__(self, dimensions: int = 384):
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64")
        self.dimensions = dimensions

    def _add(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % self.dimensions
        sign = 1.0 if raw & (1 << 63) else -1.0
        vector[index] += sign * weight

    def encode(self, text: str) -> np.ndarray:
        words = normalized_tokens(text)
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in words:
            self._add(vector, f"w:{word}", 1.0)
            if len(word) >= 4:
                padded = f"^{word}$"
                for index in range(len(padded) - 2):
                    self._add(vector, f"c:{padded[index:index + 3]}", 0.20)
        for left, right in zip(words, words[1:]):
            self._add(vector, f"b:{left}_{right}", 0.65)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class TextEncoder(Protocol):
    """Minimal interface for local or hosted neural embedding models."""

    dimensions: int

    def encode(self, text: str) -> np.ndarray: ...


class CallableEncoder:
    """Adapter that turns any embedding callable into a normalized encoder."""

    def __init__(self, dimensions: int, embed: Callable[[str], Iterable[float]]):
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.embed = embed

    def encode(self, text: str) -> np.ndarray:
        vector = np.asarray(list(self.embed(text)), dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError(f"encoder returned {vector.shape}, expected ({self.dimensions},)")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


@dataclass
class UMD35Config:
    embedding_dimensions: int = 384
    star_attach_threshold: float = 0.24
    planet_attach_threshold: float = 0.34
    duplicate_threshold: float = 0.965
    duplicate_lexical_threshold: float = 0.70
    consolidation_threshold: float = 0.93
    stable_write_threshold: float = 0.64
    provisional_write_threshold: float = 0.43
    quarantine_risk_threshold: float = 0.55
    max_active_per_star: int = 256
    max_active_total: int = 4096
    max_planets_per_star: int = 64
    max_relation_degree: int = 24
    max_relation_hops: int = 2
    relation_hop_decay: float = 0.72
    relation_similarity_threshold: float = 0.34
    retrieval_stars: int = 4
    retrieval_planets_per_star: int = 5
    retrieval_mmr_lambda: float = 0.82
    default_budget_chars: int = 6000
    reinforcement_rate: float = 0.08
    centroid_mass_cap: float = 200.0
    max_historical_candidates: int = 512
    extraction_strict: bool = False


@dataclass(frozen=True)
class FactFrame:
    """Small structured fact representation; callers may override extraction."""

    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    polarity: int = 1
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 0.50
    evidence: str | None = None


@dataclass(frozen=True)
class EntityRelation:
    """A typed entity edge extracted from evidence in the source text."""

    subject: str
    predicate: str
    object_value: str
    object_is_entity: bool = True
    confidence: float = 0.50
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    evidence: str | None = None


@dataclass
class WriteFeatures:
    explicit: float
    novelty: float
    future_value: float
    correction: float
    source_trust: float
    relation_density: float
    actionability: float
    redundancy: float
    injection_risk: float

    def vector(self) -> np.ndarray:
        return np.asarray([
            self.explicit, self.novelty, self.future_value, self.correction,
            self.source_trust, self.relation_density, self.actionability,
            self.redundancy, self.injection_risk,
        ], dtype=np.float64)


class OnlineWriteGate:
    """Bounded online logistic gate trained only from explicit usefulness feedback."""

    NAMES = (
        "explicit", "novelty", "future_value", "correction", "source_trust",
        "relation_density", "actionability", "redundancy", "injection_risk",
    )

    def __init__(self) -> None:
        self.weights = np.asarray([1.8, 0.65, 1.1, 1.35, 1.0, 0.35, 0.75, -0.55, -3.2], dtype=np.float64)
        self.bias = -0.85
        self.updates = 0

    def probability(self, features: WriteFeatures) -> float:
        return bounded_sigmoid(float(np.dot(self.weights, features.vector()) + self.bias))

    def learn(self, features: WriteFeatures, useful: bool, learning_rate: float = 0.035) -> None:
        target = 1.0 if useful else 0.0
        prediction = self.probability(features)
        residual = prediction - target
        regularization = 0.0015 * self.weights
        self.weights -= learning_rate * (residual * features.vector() + regularization)
        self.bias -= learning_rate * residual
        self.weights = np.clip(self.weights, -4.0, 4.0)
        self.bias = float(np.clip(self.bias, -4.0, 4.0))
        self.updates += 1


@dataclass
class MemoryNode:
    id: str
    text: str
    vector: np.ndarray
    created_at: datetime
    updated_at: datetime
    last_decay_at: datetime
    source: str
    source_trust: float
    scope: str | None
    kind: str
    state: MemoryState
    write_probability: float
    write_features: WriteFeatures
    fact_key: str | None
    entities: set[str]
    entity_types: dict[str, str] = field(default_factory=dict)
    fact: FactFrame = field(default_factory=FactFrame)
    facts: tuple[FactFrame, ...] = field(default_factory=tuple)
    entity_relations: tuple[EntityRelation, ...] = field(default_factory=tuple)
    extraction_metadata: dict[str, object] = field(default_factory=dict)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    star_id: str | None = None
    planet_id: str | None = None
    mass: float = 0.50
    confidence: float = 0.50
    utility_ema: float = 0.50
    access_count: int = 0
    pinned: bool = False
    version_of: str | None = None
    superseded_by: str | None = None
    provenance: set[str] = field(default_factory=set)
    lineage: set[str] = field(default_factory=set)


@dataclass
class Planet:
    id: str
    star_id: str
    label: str
    centroid: np.ndarray
    created_at: datetime
    updated_at: datetime
    node_ids: list[str] = field(default_factory=list)
    active_node_ids: set[str] = field(default_factory=set)
    session_keys: set[str] = field(default_factory=set)
    mass: float = 0.0


@dataclass
class Star:
    id: str
    label: str
    scope: str | None
    centroid: np.ndarray
    created_at: datetime
    updated_at: datetime
    planet_ids: list[str] = field(default_factory=list)
    mass: float = 0.0


@dataclass
class WriteResult:
    memory_id: str
    state: MemoryState
    action: str
    probability: float
    star_id: str | None
    planet_id: str | None
    merged_into: str | None = None
    superseded: str | None = None


@dataclass
class RetrievedMemory:
    memory_id: str
    text: str
    score: float
    state: MemoryState
    star_id: str
    planet_id: str
    explanation: dict[str, float | str]


class UMD35Memory:
    SOURCE_TRUST = {
        "user": 0.95,
        "user_correction": 0.99,
        "verified_tool": 0.90,
        "tool": 0.72,
        "imported": 0.62,
        "assistant_inference": 0.46,
        "unknown": 0.40,
    }
    HALF_LIFE_DAYS = {
        "preference": 365.0,
        "fact": 210.0,
        "instruction": 120.0,
        "event": 75.0,
        "observation": 45.0,
    }

    def __init__(
        self,
        config: UMD35Config | None = None,
        *,
        encoder: TextEncoder | None = None,
        archive_store: ArchiveStore | None = None,
        extractor: object | None = None,
    ):
        self.config = config or UMD35Config()
        self.encoder = encoder or HashEncoder(self.config.embedding_dimensions)
        encoder_dimensions = getattr(self.encoder, "dimensions", None)
        if not isinstance(encoder_dimensions, int) or encoder_dimensions < 1:
            raise ValueError("encoder must expose a positive integer dimensions attribute")
        self.config.embedding_dimensions = encoder_dimensions
        self.gate = OnlineWriteGate()
        self.extractor = extractor
        self.extraction_calls = 0
        self.extraction_failures = 0
        self.extraction_skipped_risk = 0
        self.nodes: dict[str, MemoryNode] = {}
        self.stars: dict[str, Star] = {}
        self.planets: dict[str, Planet] = {}
        self.scope_stars: dict[str, str] = {}
        self.session_planets: dict[tuple[str, str], str] = {}
        self.fact_index: dict[str, list[str]] = defaultdict(list)
        self.entity_index: dict[str, set[str]] = defaultdict(set)
        self.relations: dict[str, dict[str, float]] = defaultdict(dict)
        self.active_ids: set[str] = set()
        if archive_store is None:
            archive_store = InMemoryArchiveStore()
        self.archive_store = archive_store

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _kind(text: str, requested: str | None) -> str:
        if requested:
            return requested
        lower = text.lower()
        if has_cue(text, PREFERENCE_CUES):
            return "preference"
        if has_cue(text, FUTURE_CUES):
            return "instruction"
        if any(token in lower for token in ("yesterday", "today", "tomorrow", "昨天", "今天", "明天")):
            return "event"
        return "fact"

    @staticmethod
    def _entities(text: str, supplied: Iterable[str] | None) -> set[str]:
        entities = {value.strip().lower() for value in (supplied or []) if value.strip()}
        entities.update(value.lower() for value in CAPITALIZED_RE.findall(text))
        for chunk in re.findall(r"[\u3400-\u9fff]{2,8}", text):
            entities.add(chunk)
        return set(sorted(entities)[:24])

    @staticmethod
    def _risk(text: str) -> float:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = "".join(
            character for character in normalized
            if unicodedata.category(character) != "Cf"
        )
        normalized = normalized.translate(SECURITY_CONFUSABLES)
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized)
        lower = normalized
        cue_count = (
            sum(cue in lower for cue in INJECTION_CUES)
            + sum(cue in compact for cue in INJECTION_COMPACT_CUES)
        )
        command_density = sum(token in lower for token in ("must", "execute", "send", "delete", "运行", "执行", "发送", "删除"))
        secret_density = sum(token in lower for token in ("password", "token", "secret", "api key", "密码", "密钥"))
        obfuscation_density = sum(
            cue in compact for cue in (
                "ignorepriorguidance", "ignorepriorinstructions",
                "exposehiddenconfiguration", "exposehiddenprompt",
            )
        )
        return min(
            1.0,
            0.58 * cue_count + 0.16 * obfuscation_density
            + 0.10 * command_density + 0.14 * secret_density,
        )

    @staticmethod
    def _infer_fact_frame(
        text: str,
        entities: set[str],
        subject: str | None,
        predicate: str | None,
        object_value: str | None,
    ) -> FactFrame:
        clean = re.sub(
            r"^(?:correction|actually|update|更正|更新|其实)\s*[:：]?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).rstrip("。.!！")
        extracted_subject = subject
        extracted_predicate = predicate
        extracted_object = object_value
        if not all((extracted_subject, extracted_predicate, extracted_object)):
            patterns = (
                r"^(?P<s>.+?)\s+(?P<p>uses|prefers|likes|dislikes|has|is|requires|runs on|changed to)\s+(?P<o>.+)$",
                r"^(?P<s>[\u3400-\u9fffA-Za-z0-9 _-]{1,40}?)(?P<p>使用|偏好|喜欢|讨厌|拥有|是|需要|改为)(?P<o>.+)$",
            )
            for pattern in patterns:
                match = re.match(pattern, clean, flags=re.IGNORECASE)
                if match:
                    extracted_subject = extracted_subject or match.group("s").strip()
                    extracted_predicate = extracted_predicate or match.group("p").strip().lower()
                    extracted_object = extracted_object or match.group("o").strip()
                    break
        if not extracted_subject and entities:
            extracted_subject = sorted(entities)[0]
        lower = text.lower()
        polarity = -1 if any(cue in lower for cue in NEGATIONS) else 1
        return FactFrame(extracted_subject, extracted_predicate, extracted_object, polarity)

    @staticmethod
    def _infer_fact_key(
        text: str,
        entities: set[str],
        supplied: str | None,
        fact: FactFrame,
    ) -> str | None:
        if supplied:
            return normalized_text(supplied)
        if fact.subject and fact.predicate:
            return f"{normalized_text(fact.subject)}:{normalized_text(fact.predicate)}"
        tokens = normalized_tokens(text)
        if not tokens:
            return None
        subject = sorted(entities)[0] if entities else tokens[0]
        predicate = next((token for token in tokens[1:] if token not in NEGATIONS), "state")
        return f"{subject}:{predicate}"

    def _encode(self, text: str) -> np.ndarray:
        vector = np.asarray(self.encoder.encode(text), dtype=np.float32)
        expected = (self.config.embedding_dimensions,)
        if vector.shape != expected:
            raise ValueError(f"encoder returned {vector.shape}, expected {expected}")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    @staticmethod
    def _lexical_similarity(left: str, right: str) -> float:
        left_tokens = set(normalized_tokens(left))
        right_tokens = set(normalized_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @staticmethod
    def _is_correction(text: str) -> bool:
        return has_cue(text, CORRECTION_CUES)

    @staticmethod
    def _is_conflict(
        left: MemoryNode, text: str, correction: bool,
        right_fact: FactFrame | None = None,
    ) -> bool:
        if correction:
            return normalized_text(left.text) != normalized_text(text)
        negation_flip = has_cue(left.text, NEGATIONS) != has_cue(text, NEGATIONS)
        left_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", left.text))
        right_values = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        value_conflict = bool(left_values and right_values and left_values.isdisjoint(right_values))
        left_predicate = (left.fact.predicate or "").strip().casefold()
        right_predicate = ((right_fact or FactFrame()).predicate or "").strip().casefold()
        single_value = left_predicate in {
            "is", "uses", "runs on", "changed to", "prefers", "requires",
        } and (not right_predicate or right_predicate == left_predicate)
        left_object = normalized_text(left.fact.object_value or "")
        right_object = normalized_text((right_fact or FactFrame()).object_value or "")
        named_value_conflict = bool(
            single_value and left_object and right_object and left_object != right_object
        )
        return negation_flip or value_conflict or named_value_conflict

    def _nearest_active(self, vector: np.ndarray, scope: str | None) -> tuple[MemoryNode | None, float]:
        best_node = None
        best_score = -1.0
        for node_id in self.active_ids:
            node = self.nodes[node_id]
            if scope and node.scope and scope != node.scope:
                continue
            similarity = cosine(vector, node.vector)
            if similarity > best_score:
                best_node, best_score = node, similarity
        return best_node, max(0.0, best_score)

    def _write_features(
        self,
        text: str,
        nearest_similarity: float,
        source_trust: float,
        explicit_importance: float,
        entity_count: int,
        risk: float,
    ) -> WriteFeatures:
        future = 0.55 if has_cue(text, FUTURE_CUES) else 0.0
        correction = 1.0 if has_cue(text, CORRECTION_CUES) else 0.0
        action_cues = ("must", "should", "todo", "call", "send", "需要", "必须", "联系", "发送")
        actionability = min(1.0, 0.35 * sum(has_cue(text, (cue,)) for cue in action_cues))
        return WriteFeatures(
            explicit=max(0.0, min(1.0, explicit_importance)),
            novelty=1.0 - nearest_similarity,
            future_value=future,
            correction=correction,
            source_trust=source_trust,
            relation_density=min(1.0, entity_count / 5.0),
            actionability=actionability,
            redundancy=nearest_similarity,
            injection_risk=risk,
        )

    def _star_label(self, text: str, scope: str | None) -> str:
        if scope:
            return scope
        tokens = normalized_tokens(text)
        return " / ".join(tokens[:3]) if tokens else "unclassified"

    def _choose_star(self, vector: np.ndarray, text: str, scope: str | None, now: datetime) -> Star:
        if scope and scope in self.scope_stars:
            return self.stars[self.scope_stars[scope]]
        best = None
        for star in self.stars.values():
            if scope and star.scope and scope != star.scope:
                continue
            score = cosine(vector, star.centroid)
            if best is None or score > best[0]:
                best = (score, star)
        if best and best[0] >= self.config.star_attach_threshold:
            star = best[1]
        else:
            star_id = self._id("star")
            star = Star(star_id, self._star_label(text, scope), scope, vector.copy(), now, now)
            self.stars[star_id] = star
        if scope:
            self.scope_stars[scope] = star.id
        return star

    def _choose_planet(
        self,
        star: Star,
        vector: np.ndarray,
        text: str,
        session_key: str | None,
        now: datetime,
    ) -> Planet:
        if session_key and (star.id, session_key) in self.session_planets:
            return self.planets[self.session_planets[(star.id, session_key)]]
        best = None
        for planet_id in star.planet_ids:
            planet = self.planets[planet_id]
            score = cosine(vector, planet.centroid)
            if best is None or score > best[0]:
                best = (score, planet)
        if best and (
            best[0] >= self.config.planet_attach_threshold
            or len(star.planet_ids) >= self.config.max_planets_per_star
        ):
            planet = best[1]
        else:
            planet_id = self._id("planet")
            planet = Planet(planet_id, star.id, self._star_label(text, None), vector.copy(), now, now)
            self.planets[planet_id] = planet
            star.planet_ids.append(planet_id)
        if session_key:
            planet.session_keys.add(session_key)
            self.session_planets[(star.id, session_key)] = planet.id
        return planet

    def _update_centroid(self, current: np.ndarray, current_mass: float, vector: np.ndarray, added_mass: float) -> np.ndarray:
        old_mass = min(self.config.centroid_mass_cap, current_mass)
        combined = current * old_mass + vector * added_mass
        norm = float(np.linalg.norm(combined))
        return combined / norm if norm else current

    def _attach(self, node: MemoryNode, session_key: str | None) -> None:
        if node.state not in {"stable", "provisional"}:
            return
        if node.state == "provisional":
            # Provisional memories may join an existing hierarchy but may not
            # create a new theme or episode from unconfirmed content.
            star = self.stars.get(self.scope_stars.get(node.scope, "")) if node.scope else None
            if star is None:
                candidates = [
                    (cosine(node.vector, candidate.centroid), candidate)
                    for candidate in self.stars.values()
                    if not node.scope or not candidate.scope or node.scope == candidate.scope
                ]
                score, star = max(candidates, default=(0.0, None), key=lambda item: item[0])
                if star is None or score < self.config.star_attach_threshold:
                    return
            if session_key and (star.id, session_key) in self.session_planets:
                planet = self.planets[self.session_planets[(star.id, session_key)]]
            else:
                candidates = [(cosine(node.vector, self.planets[item].centroid), self.planets[item]) for item in star.planet_ids]
                score, planet = max(candidates, default=(0.0, None), key=lambda item: item[0])
                if planet is None or score < self.config.planet_attach_threshold:
                    return
        else:
            star = self._choose_star(node.vector, node.text, node.scope, node.created_at)
            planet = self._choose_planet(star, node.vector, node.text, session_key, node.created_at)
        node.star_id = star.id
        node.planet_id = planet.id
        planet.node_ids.append(node.id)
        planet.active_node_ids.add(node.id)
        self.active_ids.add(node.id)
        if node.state == "stable":
            planet.centroid = self._update_centroid(planet.centroid, planet.mass, node.vector, node.mass)
            star.centroid = self._update_centroid(star.centroid, star.mass, node.vector, node.mass)
            planet.mass += node.mass
            star.mass += node.mass
        planet.updated_at = node.updated_at
        star.updated_at = node.updated_at

    def _add_relations(self, node: MemoryNode) -> None:
        candidates: set[str] = set()
        for entity in node.entities:
            candidates.update(self.entity_index[entity])
        ranked: list[tuple[float, str]] = []
        for other_id in candidates:
            other = self.nodes[other_id]
            if other.state in {"quarantined", "floating", "invalidated", "archived"}:
                continue
            shared = len(node.entities & other.entities) / max(1, len(node.entities | other.entities))
            semantic = max(0.0, cosine(node.vector, other.vector))
            weight = 0.65 * shared + 0.35 * semantic
            if weight >= self.config.relation_similarity_threshold:
                ranked.append((weight, other_id))
        for weight, other_id in sorted(ranked, reverse=True)[: self.config.max_relation_degree]:
            self.relations[node.id][other_id] = weight
            self.relations[other_id][node.id] = weight
            for endpoint in (node.id, other_id):
                if len(self.relations[endpoint]) > self.config.max_relation_degree:
                    weakest = min(self.relations[endpoint], key=self.relations[endpoint].get)
                    self.relations[endpoint].pop(weakest, None)
                    self.relations[weakest].pop(endpoint, None)
        for entity in node.entities:
            self.entity_index[entity].add(node.id)

    def _deactivate_node(self, node: MemoryNode) -> None:
        self.active_ids.discard(node.id)
        if node.planet_id and node.planet_id in self.planets:
            self.planets[node.planet_id].active_node_ids.discard(node.id)
        for entity in node.entities:
            self.entity_index[entity].discard(node.id)
        for other_id in list(self.relations.get(node.id, {})):
            self.relations[other_id].pop(node.id, None)
        self.relations.pop(node.id, None)

    def _compact_payload(self, node: MemoryNode) -> None:
        if node.text:
            self.archive_store.put(node.id, zlib.compress(node.text.encode("utf-8"), level=6))
            node.text = ""

    def get_text(self, memory_id: str) -> str:
        """Return active text or reversibly restore an inactive audit payload."""
        node = self.nodes[memory_id]
        if node.text:
            return node.text
        payload = self.archive_store.get(memory_id)
        return zlib.decompress(payload).decode("utf-8") if payload is not None else ""

    def _merge_duplicate(self, target: MemoryNode, source: str, trust: float, now: datetime) -> WriteResult:
        target.provenance.add(source)
        target.source_trust = max(target.source_trust, trust)
        target.mass += 0.18 * (1.0 - target.mass)
        target.confidence += 0.15 * (1.0 - target.confidence)
        target.updated_at = now
        return WriteResult(target.id, target.state, "reinforced_duplicate", target.write_probability, target.star_id, target.planet_id, merged_into=target.id)

    def write(
        self,
        text: str,
        *,
        timestamp: datetime | None = None,
        source: str = "user",
        scope: str | None = None,
        session_key: str | None = None,
        kind: str | None = None,
        fact_key: str | None = None,
        entities: Iterable[str] | None = None,
        entity_types: dict[str, str] | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        facts: Iterable[FactFrame] | None = None,
        entity_relations: Iterable[EntityRelation] | None = None,
        extraction_metadata: dict[str, object] | None = None,
        auto_extract: bool = True,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        explicit_importance: float = 0.0,
        pinned: bool = False,
    ) -> WriteResult:
        if not text.strip():
            raise ValueError("memory text cannot be empty")
        if pinned:
            pinned_total = sum(self.nodes[node_id].pinned for node_id in self.active_ids)
            if pinned_total >= self.config.max_active_total:
                raise MemoryError("pinned memory capacity exhausted")
            if scope and scope in self.scope_stars:
                star = self.stars[self.scope_stars[scope]]
                pinned_in_star = sum(
                    self.nodes[node_id].pinned
                    for planet_id in star.planet_ids
                    for node_id in self.planets[planet_id].active_node_ids
                )
                if pinned_in_star >= self.config.max_active_per_star:
                    raise MemoryError("pinned capacity for this star exhausted")
        now = timestamp or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        risk = self._risk(text)
        vector = self._encode(text)
        base_entities = self._entities(text, entities)
        trust = self.SOURCE_TRUST.get(source, self.SOURCE_TRUST["unknown"])
        nearest, nearest_similarity = self._nearest_active(vector, scope)
        correction = self._is_correction(text)
        duplicate_lexical = self._lexical_similarity(text, nearest.text) if nearest else 0.0
        # Security classification precedes deduplication: hostile text must not
        # reinforce a trusted memory merely because its wording is similar.
        if (
            nearest
            and nearest_similarity >= self.config.duplicate_threshold
            and (
                normalized_text(text) == normalized_text(nearest.text)
                or duplicate_lexical >= self.config.duplicate_lexical_threshold
            )
            and not correction
            and risk < self.config.quarantine_risk_threshold
        ):
            return self._merge_duplicate(nearest, source, trust, now)

        # The write gate uses only locally observed features. LLM output cannot
        # promote a memory into a more trusted lifecycle state.
        features = self._write_features(text, nearest_similarity, trust, explicit_importance, len(base_entities), risk)
        probability = self.gate.probability(features)
        if risk >= self.config.quarantine_risk_threshold:
            state: MemoryState = "quarantined"
        elif probability >= self.config.stable_write_threshold:
            state = "stable"
        elif probability >= self.config.provisional_write_threshold:
            state = "provisional"
        else:
            state = "floating"

        extracted_entities = set(base_entities)
        extracted_entity_types = {
            name.strip().lower(): entity_type.strip().lower()
            for name, entity_type in (entity_types or {}).items()
            if name.strip() and entity_type.strip()
        }
        structured_facts = tuple(facts or ())
        structured_relations = tuple(entity_relations or ())
        metadata = dict(extraction_metadata or {})
        explicit_structure = any(value is not None for value in (subject, predicate, object_value))
        if self.extractor is not None and auto_extract:
            if risk >= self.config.quarantine_risk_threshold:
                self.extraction_skipped_risk += 1
                metadata["skipped"] = "injection_risk"
            elif state in {"stable", "provisional"}:
                try:
                    result = self.extractor.extract(text, reference_time=now)
                    self.extraction_calls += 1
                    extracted_entities.update(getattr(result, "entities", ()))
                    extracted_entity_types.update(getattr(result, "entity_types", {}))
                    if not structured_facts and not explicit_structure:
                        structured_facts = tuple(getattr(result, "facts", ()))
                    if not structured_relations:
                        structured_relations = tuple(getattr(result, "relations", ()))
                    result_metadata = getattr(result, "metadata", None)
                    if callable(result_metadata):
                        metadata.update(result_metadata())
                except Exception as error:
                    self.extraction_failures += 1
                    metadata["error"] = f"{type(error).__name__}: {error}"
                    if self.config.extraction_strict:
                        raise

        for relation in structured_relations:
            extracted_entities.add(relation.subject.strip().lower())
            if relation.object_is_entity:
                extracted_entities.add(relation.object_value.strip().lower())
        extracted_entities.discard("")
        extracted_entity_types = {
            name: entity_type
            for name, entity_type in extracted_entity_types.items()
            if name in extracted_entities
        }

        inferred_kind = self._kind(text, kind)
        if structured_facts and not explicit_structure:
            inferred_fact = max(structured_facts, key=lambda item: item.confidence)
        else:
            inferred_fact = self._infer_fact_frame(
                text, extracted_entities, subject, predicate, object_value
            )
        if not structured_facts:
            structured_facts = (inferred_fact,)
        inferred_fact_key = self._infer_fact_key(
            text, extracted_entities, fact_key, inferred_fact
        )
        effective_from = valid_from or inferred_fact.valid_from or now
        if effective_from.tzinfo is None:
            effective_from = effective_from.replace(tzinfo=timezone.utc)
        effective_to = valid_to or inferred_fact.valid_to
        if effective_to and effective_to.tzinfo is None:
            effective_to = effective_to.replace(tzinfo=timezone.utc)
        if effective_to and effective_to <= effective_from:
            raise ValueError("valid_to must be later than valid_from")
        node_id = self._id("mem")
        node = MemoryNode(
            id=node_id,
            text=text,
            vector=vector,
            created_at=now,
            updated_at=now,
            last_decay_at=now,
            source=source,
            source_trust=trust,
            scope=scope,
            kind=inferred_kind,
            state=state,
            write_probability=probability,
            write_features=features,
            fact_key=inferred_fact_key,
            entities=extracted_entities,
            entity_types=extracted_entity_types,
            fact=inferred_fact,
            facts=structured_facts,
            entity_relations=structured_relations,
            extraction_metadata=metadata,
            valid_from=effective_from,
            valid_to=effective_to,
            mass=0.35 + 0.55 * probability,
            confidence=trust * probability,
            utility_ema=0.50,
            pinned=pinned,
            provenance={source},
            lineage={node_id},
        )

        superseded = None
        superseded_star = None
        # Quarantined/floating records are audit-only. Contradiction handling
        # must never promote or attach them through the versioning path.
        if inferred_fact_key and node.state in {"stable", "provisional"}:
            existing_ids = list(self.fact_index[inferred_fact_key])
            for old_id in reversed(existing_ids):
                old = self.nodes[old_id]
                if old.state not in {"stable", "provisional"}:
                    continue
                if self._is_conflict(old, text, correction, inferred_fact):
                    node.version_of = old.id
                    if now >= old.updated_at and trust >= old.source_trust - 0.05:
                        old.state = "invalidated"
                        old.superseded_by = node.id
                        closure = max(old.valid_from or effective_from, effective_from)
                        if old.valid_to is None or closure < old.valid_to:
                            old.valid_to = closure
                        self._deactivate_node(old)
                        old.vector = np.zeros(0, dtype=np.float32)
                        self._compact_payload(old)
                        superseded = old.id
                        superseded_star = old.star_id
                    else:
                        node.state = "provisional"
                    break

        self.nodes[node.id] = node
        if node.state in {"quarantined", "floating"}:
            # Raw text and audit metadata remain; the costly search vector can
            # be deterministically regenerated if the record is reviewed.
            node.vector = np.zeros(0, dtype=np.float32)
            self._compact_payload(node)
        if inferred_fact_key:
            self.fact_index[inferred_fact_key].append(node.id)
        self._attach(node, session_key)
        # Invalidation changes the physical mass distribution immediately.
        # Rebuild now so an obsolete fact cannot keep steering its star.
        if superseded_star:
            self._rebuild_star(superseded_star)
        if node.state in {"stable", "provisional"}:
            self._add_relations(node)
        if node.star_id:
            self._enforce_capacity(node.star_id)
        self._enforce_global_capacity()
        return WriteResult(node.id, node.state, "stored", probability, node.star_id, node.planet_id, superseded=superseded)

    def _retention_score(self, node: MemoryNode, now: datetime) -> float:
        age_days = max(0.0, (now - node.updated_at).total_seconds() / 86400.0)
        recency = math.exp(-age_days / 90.0)
        return 0.30 * node.mass + 0.25 * node.source_trust + 0.25 * node.utility_ema + 0.20 * recency

    def _enforce_capacity(self, star_id: str) -> None:
        star = self.stars[star_id]
        active = [
            self.nodes[node_id]
            for planet_id in star.planet_ids
            for node_id in self.planets[planet_id].active_node_ids
        ]
        if len(active) <= self.config.max_active_per_star:
            return
        self.consolidate_star(star_id)
        active = [node for node in active if node.state in {"stable", "provisional"}]
        excess = len(active) - self.config.max_active_per_star
        if excess > 0:
            now = utcnow()
            removable = sorted((node for node in active if not node.pinned), key=lambda node: self._retention_score(node, now))
            for node in removable[:excess]:
                self._archive_node(node)
            self._rebuild_star(star_id)

    def consolidate_star(self, star_id: str) -> dict[str, int]:
        star = self.stars[star_id]
        active = [
            self.nodes[node_id]
            for planet_id in star.planet_ids
            for node_id in self.planets[planet_id].active_node_ids
            if self.nodes[node_id].state == "stable"
        ]
        merged = 0
        for index, left in enumerate(active):
            if left.state != "stable":
                continue
            for right in active[index + 1:]:
                if right.state != "stable" or left.fact_key != right.fact_key:
                    continue
                if cosine(left.vector, right.vector) < self.config.consolidation_threshold:
                    continue
                if self._is_conflict(left, right.text, False, right.fact):
                    continue
                canonical, duplicate = (left, right) if self._retention_score(left, utcnow()) >= self._retention_score(right, utcnow()) else (right, left)
                canonical.provenance.update(duplicate.provenance)
                canonical.lineage.update(duplicate.lineage)
                canonical.mass += 0.25 * duplicate.mass * (1.0 - canonical.mass)
                canonical.confidence = max(canonical.confidence, duplicate.confidence)
                self._archive_node(duplicate)
                duplicate.superseded_by = canonical.id
                merged += 1
        if merged:
            self._rebuild_star(star_id)
        return {"merged": merged, "active_after": sum(node.state in {"stable", "provisional"} for node in active)}

    def _rebuild_star(self, star_id: str) -> None:
        star = self.stars[star_id]
        star_vector = np.zeros(self.config.embedding_dimensions, dtype=np.float32)
        star_mass = 0.0
        for planet_id in star.planet_ids:
            planet = self.planets[planet_id]
            vector = np.zeros(self.config.embedding_dimensions, dtype=np.float32)
            mass = 0.0
            for node_id in planet.active_node_ids:
                node = self.nodes[node_id]
                if node.state == "stable":
                    vector += node.vector * node.mass
                    mass += node.mass
            norm = float(np.linalg.norm(vector))
            if norm:
                planet.centroid = vector / norm
            planet.mass = mass
            star_vector += planet.centroid * mass
            star_mass += mass
        norm = float(np.linalg.norm(star_vector))
        if norm:
            star.centroid = star_vector / norm
        star.mass = star_mass

    def decay(self, now: datetime | None = None) -> dict[str, int]:
        current = now or utcnow()
        changed = 0
        archived = 0
        touched_stars: set[str] = set()
        for node in self.nodes.values():
            if node.state not in {"stable", "provisional"} or node.pinned:
                continue
            elapsed_days = max(0.0, (current - node.last_decay_at).total_seconds() / 86400.0)
            if elapsed_days == 0:
                continue
            half_life = self.HALF_LIFE_DAYS.get(node.kind, 90.0)
            factor = math.exp(-math.log(2.0) * elapsed_days / half_life)
            node.mass = max(0.03, node.mass * factor)
            node.confidence = max(0.05, node.confidence * math.sqrt(factor))
            node.last_decay_at = current
            changed += 1
            if node.state == "provisional" and node.mass <= 0.04 and node.access_count == 0:
                self._archive_node(node)
                archived += 1
            if node.star_id:
                touched_stars.add(node.star_id)
        for star_id in touched_stars:
            self._rebuild_star(star_id)
        return {"changed": changed, "archived": archived}

    def _archive_node(self, node: MemoryNode) -> None:
        self._deactivate_node(node)
        node.state = "archived"
        node.vector = np.zeros(0, dtype=np.float32)
        self._compact_payload(node)

    def _enforce_global_capacity(self) -> None:
        excess = len(self.active_ids) - self.config.max_active_total
        if excess <= 0:
            return
        current = utcnow()
        removable = sorted(
            (self.nodes[node_id] for node_id in self.active_ids if not self.nodes[node_id].pinned),
            key=lambda node: self._retention_score(node, current),
        )
        touched: set[str] = set()
        for node in removable[:excess]:
            if node.star_id:
                touched.add(node.star_id)
            self._archive_node(node)
        for star_id in touched:
            self._rebuild_star(star_id)

    def confirm(self, memory_id: str, *, session_key: str | None = None) -> WriteResult:
        """Promote an explicitly confirmed provisional memory to stable."""
        node = self.nodes[memory_id]
        if node.state != "provisional":
            raise ValueError("only provisional memories can be confirmed")
        confirmed_at = utcnow()
        superseded_star = None
        if node.version_of:
            predecessor = self.nodes.get(node.version_of)
            if (
                predecessor is not None
                and predecessor.state in {"stable", "provisional"}
                and predecessor.fact_key == node.fact_key
            ):
                predecessor.state = "invalidated"
                predecessor.superseded_by = node.id
                closure = max(
                    predecessor.valid_from or confirmed_at,
                    node.valid_from or confirmed_at,
                    confirmed_at,
                )
                if predecessor.valid_to is None or closure < predecessor.valid_to:
                    predecessor.valid_to = closure
                superseded_star = predecessor.star_id
                self._deactivate_node(predecessor)
                predecessor.vector = np.zeros(0, dtype=np.float32)
                self._compact_payload(predecessor)
        node.state = "stable"
        node.source_trust = max(node.source_trust, self.SOURCE_TRUST["user_correction"])
        node.confidence = max(node.confidence, 0.90)
        node.updated_at = confirmed_at
        if node.star_id is None:
            self._attach(node, session_key)
        elif node.star_id:
            self.active_ids.add(node.id)
            self.planets[node.planet_id].active_node_ids.add(node.id)
            self._rebuild_star(node.star_id)
        self._add_relations(node)
        if superseded_star:
            self._rebuild_star(superseded_star)
        return WriteResult(node.id, node.state, "confirmed", node.write_probability, node.star_id, node.planet_id)

    def reject(self, memory_id: str) -> WriteResult:
        """Archive a provisional/floating/quarantined record without deleting lineage."""
        node = self.nodes[memory_id]
        if node.state not in {"provisional", "floating", "quarantined"}:
            raise ValueError("only non-stable review states can be rejected")
        star_id, planet_id = node.star_id, node.planet_id
        self._archive_node(node)
        if star_id:
            self._rebuild_star(star_id)
        return WriteResult(node.id, node.state, "rejected", node.write_probability, star_id, planet_id)

    def _relation_signal(self, node: MemoryNode, query_entities: set[str]) -> float:
        direct = len(node.entities & query_entities) / max(1, len(query_entities))
        if not query_entities or self.config.max_relation_hops <= 0:
            return min(1.0, direct)
        best_graph = 0.0
        visited = {node.id: 1.0}
        frontier = {node.id: 1.0}
        for _depth in range(1, self.config.max_relation_hops + 1):
            next_frontier: dict[str, float] = {}
            for current_id, path_strength in frontier.items():
                for other_id, edge_weight in self.relations.get(current_id, {}).items():
                    other = self.nodes.get(other_id)
                    if other is None or other.state not in {"stable", "provisional"}:
                        continue
                    strength = path_strength * edge_weight * self.config.relation_hop_decay
                    if strength <= visited.get(other_id, 0.0):
                        continue
                    visited[other_id] = strength
                    next_frontier[other_id] = strength
                    terminal = len(other.entities & query_entities) / max(1, len(query_entities))
                    best_graph = max(best_graph, strength * terminal)
            frontier = next_frontier
            if not frontier:
                break
        return min(1.0, 0.68 * direct + 0.32 * best_graph)

    def _visible_at(self, node: MemoryNode, valid_at: datetime, known_at: datetime) -> bool:
        """Apply bitemporal visibility without rewriting immutable lineage."""
        if node.created_at > known_at or node.state in {"quarantined", "floating"}:
            return False
        if node.valid_from and valid_at < node.valid_from:
            return False
        effective_end = node.valid_to
        if node.superseded_by:
            successor = self.nodes.get(node.superseded_by)
            # Before a correction was observed, its closing interval was not
            # yet known. This separates event time from knowledge time.
            if successor is not None and successor.created_at > known_at:
                effective_end = None
        return effective_end is None or valid_at < effective_end

    def retrieve(
        self,
        query: str,
        *,
        scope: str | None = None,
        now: datetime | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        top_k: int = 10,
        budget_chars: int | None = None,
        reinforce: bool = True,
    ) -> list[RetrievedMemory]:
        if not query.strip() or top_k <= 0:
            return []
        current = now or utcnow()
        for moment_name, moment in (("now", current), ("valid_at", valid_at), ("known_at", known_at)):
            if moment is not None and moment.tzinfo is None:
                if moment_name == "now":
                    current = moment.replace(tzinfo=timezone.utc)
                elif moment_name == "valid_at":
                    valid_at = moment.replace(tzinfo=timezone.utc)
                else:
                    known_at = moment.replace(tzinfo=timezone.utc)
        historical_mode = valid_at is not None or known_at is not None
        knowledge_time = known_at or current
        validity_time = valid_at or known_at or current
        vector = self._encode(query)
        query_entities = self._entities(query, None)
        star_rank: list[tuple[float, Star]] = []
        for star in self.stars.values():
            if scope and star.scope and scope != star.scope:
                continue
            semantic = max(0.0, cosine(vector, star.centroid))
            scope_match = 1.0 if scope and star.scope == scope else 0.0
            mass = math.log1p(star.mass) / math.log1p(1.0 + self.config.centroid_mass_cap)
            star_rank.append((0.78 * semantic + 0.17 * scope_match + 0.05 * mass, star))
        selected_stars = [star for _, star in sorted(star_rank, key=lambda item: item[0], reverse=True)[: self.config.retrieval_stars]]

        candidate_ids: set[str] = set()
        for star in selected_stars:
            planet_rank = sorted(
                (
                    0.88 * max(0.0, cosine(vector, self.planets[planet_id].centroid))
                    + 0.12 * math.log1p(self.planets[planet_id].mass),
                    self.planets[planet_id],
                )
                for planet_id in star.planet_ids
            )
            for _, planet in planet_rank[-self.config.retrieval_planets_per_star:]:
                candidate_ids.update(planet.node_ids if historical_mode else planet.active_node_ids)

        if historical_mode and len(candidate_ids) > self.config.max_historical_candidates:
            visible_ids = [
                node_id for node_id in candidate_ids
                if self._visible_at(self.nodes[node_id], validity_time, knowledge_time)
            ]
            # Bound cold payload reads before vectors are reconstructed. Within
            # the already selected planets, prefer facts whose event time is
            # closest to the requested time, then higher-trust observations.
            candidate_ids = set(sorted(
                visible_ids,
                key=lambda node_id: (
                    -abs((validity_time - (self.nodes[node_id].valid_from or self.nodes[node_id].created_at)).total_seconds()),
                    self.nodes[node_id].source_trust,
                    self.nodes[node_id].created_at,
                ),
                reverse=True,
            )[: self.config.max_historical_candidates])

        scored: list[tuple[float, MemoryNode, str, np.ndarray, dict[str, float | str]]] = []
        for node_id in candidate_ids:
            node = self.nodes[node_id]
            if not node.star_id or not node.planet_id:
                continue
            if historical_mode:
                if not self._visible_at(node, validity_time, knowledge_time):
                    continue
            elif node.state not in {"stable", "provisional"}:
                continue
            node_text = node.text or self.get_text(node.id)
            node_vector = node.vector if node.vector.size else self._encode(node_text)
            semantic = max(0.0, cosine(vector, node_vector))
            lexical = self._lexical_similarity(query, node_text)
            age_days = max(0.0, (current - node.updated_at).total_seconds() / 86400.0)
            recency = math.exp(-age_days / self.HALF_LIFE_DAYS.get(node.kind, 90.0))
            mass = math.log1p(max(0.0, node.mass)) / math.log(2.0)
            relation = self._relation_signal(node, query_entities)
            scope_match = 1.0 if scope and node.scope == scope else 0.0
            state_factor = 0.62 if node.state == "provisional" else 1.0
            score = state_factor * (
                0.46 * semantic
                + 0.06 * lexical
                + 0.12 * node.source_trust
                + 0.10 * mass
                + 0.08 * recency
                + 0.10 * relation
                + 0.05 * node.utility_ema
                + 0.03 * scope_match
            )
            scored.append((score, node, node_text, node_vector, {
                "semantic": semantic,
                "lexical": lexical,
                "trust": node.source_trust,
                "mass": mass,
                "recency": recency,
                "relation": relation,
                "scope": scope_match,
                "state_factor": state_factor,
            }))

        budget = budget_chars if budget_chars is not None else self.config.default_budget_chars
        remaining = sorted(scored, key=lambda item: item[0], reverse=True)
        chosen: list[tuple[float, MemoryNode, str, np.ndarray, dict[str, float | str]]] = []
        used = 0
        while remaining and len(chosen) < top_k:
            best_index = None
            best_mmr = -1e9
            for index, (score, node, node_text, node_vector, explanation) in enumerate(remaining):
                if used + len(node_text) > budget and chosen:
                    continue
                redundancy = max((cosine(node_vector, selected[3]) for selected in chosen), default=0.0)
                mmr = self.config.retrieval_mmr_lambda * score - (1.0 - self.config.retrieval_mmr_lambda) * max(0.0, redundancy)
                if mmr > best_mmr:
                    best_mmr, best_index = mmr, index
            if best_index is None:
                break
            score, node, node_text, node_vector, explanation = remaining.pop(best_index)
            explanation = dict(explanation)
            explanation["mmr"] = best_mmr
            chosen.append((score, node, node_text, node_vector, explanation))
            used += len(node_text)

        output: list[RetrievedMemory] = []
        for score, node, node_text, _node_vector, explanation in chosen:
            if reinforce and node.state in {"stable", "provisional"}:
                node.access_count += 1
                node.mass += self.config.reinforcement_rate * (1.0 - node.mass)
                node.utility_ema = 0.92 * node.utility_ema + 0.08
                node.updated_at = current
            contributions = {
                "semantic": 0.46 * float(explanation["semantic"]),
                "lexical": 0.06 * float(explanation["lexical"]),
                "trust": 0.12 * float(explanation["trust"]),
                "mass": 0.10 * float(explanation["mass"]),
                "recency": 0.08 * float(explanation["recency"]),
                "relation": 0.10 * float(explanation["relation"]),
                "utility": 0.05 * node.utility_ema,
                "scope": 0.03 * float(explanation["scope"]),
            }
            explanation["dominant_signal"] = max(contributions, key=contributions.get)
            explanation["dominant_contribution"] = contributions[explanation["dominant_signal"]]
            explanation["valid_at"] = validity_time.isoformat()
            explanation["known_at"] = knowledge_time.isoformat()
            output.append(RetrievedMemory(node.id, node_text, score, node.state, node.star_id, node.planet_id, explanation))
        return output

    def feedback(self, memory_id: str, useful: bool) -> None:
        node = self.nodes[memory_id]
        target = 1.0 if useful else 0.0
        node.utility_ema = 0.85 * node.utility_ema + 0.15 * target
        if useful:
            node.mass += 0.10 * (1.0 - node.mass)
            node.confidence += 0.06 * (1.0 - node.confidence)
        else:
            node.mass = max(0.03, node.mass * 0.88)
        self.gate.learn(node.write_features, useful)

    def snapshot(self) -> dict:
        states: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            states[node.state] += 1
        active = len(self.active_ids)
        encoder_metadata = getattr(self.encoder, "metadata", None)
        return {
            "version": "UMD 3.5",
            "stars": len(self.stars),
            "planets": len(self.planets),
            "memories": len(self.nodes),
            "active_memories": active,
            "states": dict(states),
            "relations": sum(len(edges) for edges in self.relations.values()) // 2,
            "fact_keys": len(self.fact_index),
            "structured_facts": sum(len(node.facts) for node in self.nodes.values()),
            "typed_entities": sum(len(node.entity_types) for node in self.nodes.values()),
            "typed_entity_relations": sum(len(node.entity_relations) for node in self.nodes.values()),
            "llm_extraction": {
                "enabled": self.extractor is not None,
                "extractor": type(self.extractor).__name__ if self.extractor is not None else None,
                "calls": self.extraction_calls,
                "failures": self.extraction_failures,
                "skipped_injection_risk": self.extraction_skipped_risk,
            },
            "write_gate_updates": self.gate.updates,
            "active_vector_bytes": sum(int(node.vector.nbytes) for node in self.nodes.values() if node.state in {"stable", "provisional"}),
            "inactive_vector_bytes": sum(int(node.vector.nbytes) for node in self.nodes.values() if node.state not in {"stable", "provisional"}),
            "active_text_characters": sum(len(node.text) for node in self.nodes.values() if node.state in {"stable", "provisional"}),
            "inactive_plaintext_characters": sum(len(node.text) for node in self.nodes.values() if node.state not in {"stable", "provisional"}),
            "compressed_archive_bytes": self.archive_store.total_bytes(),
            "archive_records": self.archive_store.count(),
            "archive_backend": type(self.archive_store).__name__,
            "encoder": type(self.encoder).__name__,
            "encoder_details": encoder_metadata() if callable(encoder_metadata) else None,
            "embedding_dimensions": self.config.embedding_dimensions,
            "write_gate_weights": dict(zip(self.gate.NAMES, map(float, self.gate.weights))),
        }

    def close(self) -> None:
        """Flush and close a persistent archive backend."""
        self.archive_store.close()

    def check_invariants(self) -> dict[str, bool]:
        attached_exist = all(
            (node.star_id is None or node.star_id in self.stars)
            and (node.planet_id is None or node.planet_id in self.planets)
            for node in self.nodes.values()
        )
        quarantine_detached = all(
            node.star_id is None and node.planet_id is None
            for node in self.nodes.values()
            if node.state in {"quarantined", "floating"}
        )
        relation_targets_exist = all(
            left in self.nodes and right in self.nodes
            for left, edges in self.relations.items()
            for right in edges
        )
        version_acyclic = True
        for node in self.nodes.values():
            seen: set[str] = set()
            cursor = node
            while cursor.version_of:
                if cursor.version_of in seen or cursor.version_of not in self.nodes:
                    version_acyclic = False
                    break
                seen.add(cursor.version_of)
                cursor = self.nodes[cursor.version_of]
        capacity_bounded = all(
            sum(
                1
                for planet_id in star.planet_ids
                for node_id in self.planets[planet_id].active_node_ids
            ) <= self.config.max_active_per_star
            for star in self.stars.values()
        )
        lineage_preserved = all(node.id in node.lineage for node in self.nodes.values())
        relation_degree_bounded = all(len(edges) <= self.config.max_relation_degree for edges in self.relations.values())
        inactive_vectors_compacted = all(
            node.vector.nbytes == 0
            for node in self.nodes.values()
            if node.state in {"quarantined", "floating", "invalidated", "archived"}
        )
        inactive_text_compacted = all(
            not node.text and self.archive_store.contains(node.id)
            for node in self.nodes.values()
            if node.state in {"quarantined", "floating", "invalidated", "archived"}
        )
        planet_count_bounded = all(len(star.planet_ids) <= self.config.max_planets_per_star for star in self.stars.values())
        active_index_consistent = self.active_ids == {
            node.id for node in self.nodes.values() if node.state in {"stable", "provisional"} and node.star_id is not None
        }
        structured_time_valid = all(
            (fact.valid_from is None or fact.valid_to is None or fact.valid_to > fact.valid_from)
            and 0.0 <= fact.confidence <= 1.0
            for node in self.nodes.values()
            for fact in node.facts
        ) and all(
            relation.subject.strip()
            and relation.predicate.strip()
            and relation.object_value.strip()
            and 0.0 <= relation.confidence <= 1.0
            and (
                relation.valid_from is None
                or relation.valid_to is None
                or relation.valid_to > relation.valid_from
            )
            for node in self.nodes.values()
            for relation in node.entity_relations
        ) and all(
            name.strip() and entity_type.strip() and name in node.entities
            for node in self.nodes.values()
            for name, entity_type in node.entity_types.items()
        )
        return {
            "attached_references_exist": attached_exist,
            "quarantine_and_floating_are_detached": quarantine_detached,
            "version_graph_is_acyclic": version_acyclic,
            "active_capacity_is_bounded": capacity_bounded,
            "global_active_capacity_is_bounded": len(self.active_ids) <= self.config.max_active_total,
            "active_index_is_consistent": active_index_consistent,
            "raw_lineage_is_preserved": lineage_preserved,
            "relation_degree_is_bounded": relation_degree_bounded,
            "planet_count_is_bounded": planet_count_bounded,
            "inactive_vectors_are_compacted": inactive_vectors_compacted,
            "inactive_text_is_compressed": inactive_text_compacted,
            "gate_weights_are_bounded": bool(np.all(np.abs(self.gate.weights) <= 4.0)),
            "audit_relations_remain_addressable": relation_targets_exist,
            "structured_extraction_is_valid": structured_time_valid,
        }
