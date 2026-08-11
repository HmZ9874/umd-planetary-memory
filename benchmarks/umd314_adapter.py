"""Gold-blind array adapter for UMD 3.14 relative-gravity retrieval.

The production runtime stores durable nodes and builds transient bridges. Public
benchmarks already expose immutable text arrays, groups and dates, so this file
mirrors the query-time policy without reading answers, evidence IDs or question
types. Parameters are frozen here and can be selected only on development data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from umd310_cognitive import UMD310CognitiveMemory


ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b|\b[A-Za-z]+[-_.:/]\d+(?:\.\d+)*\b")
SUBGOAL_RE = re.compile(
    r"\s*(?:,|;|\band\b|\bthen\b|\bincluding\b|\bas well as\b|以及|并且|然后|包括|、)\s*",
    re.IGNORECASE,
)
FAMILY_QUERY_CUES: dict[str, tuple[str, ...]] = {
    "format": ("code", "snippet", "implement", "syntax", "format", "代码", "实现", "格式"),
    "dependencies": ("library", "libraries", "dependency", "framework", "tool", "库", "依赖", "框架", "工具"),
    "response_style": ("answer", "reply", "response", "explain", "回答", "回复", "解释"),
    "language": ("language", "english", "chinese", "语言", "英文", "中文"),
    "privacy": ("security", "private", "privacy", "secret", "安全", "隐私", "秘密"),
    "workflow": ("how", "implement", "build", "create", "setup", "如何", "实现", "创建", "设置"),
}
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "my", "your", "our", "this", "that", "is", "are", "was", "were", "be",
    "project", "user", "assistant",
}


def _tokens(text: str) -> set[str]:
    return {
        value.casefold() for value in re.findall(r"[A-Za-z0-9_.:/-]+|[\u3400-\u9fff]+", text)
        if value.casefold() not in STOP
    }


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left))


def _directive_score(query: str, text: str, metadata: tuple[str, str] | None) -> float:
    return directive_gravity_values(query, (text,), (metadata,))[0]


def directive_gravity_values(
    query: str,
    texts: Sequence[str],
    metadata: Sequence[tuple[str, str] | None],
) -> list[float]:
    """Evaluate the transient directive field once for an entire query.

    Query mode, cue regexes and query tokens are invariant across sources.  A
    batched field therefore preserves the 3.14 force law while avoiding their
    recomputation for every directive-bearing memory and again for the bridge.
    """
    mode = UMD310CognitiveMemory._query_mode(query)
    lowered = query.casefold()
    history_lookup = bool(re.search(
        r"\b(?:did\s+(?:i|you|we)|when\s+did|where\s+did|what\s+.+\s+mention(?:ed)?|"
        r"how\s+(?:much|long)\s+did|said\s+i|was\s+recommended)\b",
        lowered,
    ))
    multi_evidence = bool(re.search(
        r"\b(?:across|between|considering|combined|in\s+total|how\s+many\s+(?:different|total)|"
        r"evolved|influence(?:d)?|from\s+.+\s+to)\b",
        lowered,
    ))
    preference_query = bool(re.search(
        r"\b(?:prefer|preference|preferred|my\s+usual)\b", lowered,
    ))
    query_tokens = _tokens(query)
    values: list[float] = []
    for text, item in zip(texts, metadata):
        if item is None or history_lookup or multi_evidence:
            values.append(0.0)
            continue
        role, family = item
        if role == "instruction" and not mode.instructions:
            values.append(0.0)
            continue
        if role == "preference" and (not mode.preferences or not preference_query):
            values.append(0.0)
            continue
        family_match = any(cue in lowered for cue in FAMILY_QUERY_CUES.get(family, ()))
        lexical = _overlap(query_tokens, _tokens(text))
        # A directive is durable, but durability does not make it applicable.
        if not family_match and lexical < 0.15:
            values.append(0.0)
            continue
        floor = 0.45 if role == "instruction" else 0.30
        applicability = min(
            1.0, max(floor, 0.58 * lexical + 0.42 * float(family_match)),
        )
        values.append(0.24 + 0.14 * applicability)
    if len(values) < len(texts):
        values.extend([0.0] * (len(texts) - len(values)))
    return values


def relative_gravity_order(
    query: str,
    texts: Sequence[str],
    groups: Sequence[int],
    dates: Sequence[str],
    base_order: Sequence[int],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    directive_meta: Sequence[tuple[str, str] | None],
    *,
    top_k: int,
    directive_values: Sequence[float] | None = None,
) -> tuple[list[int], list[float]]:
    """Return the UMD 3.14 source order and its adjusted transient field."""
    if not texts:
        return [], []
    adjusted = [float(value) for value in force]
    gravity_values = list(directive_values) if directive_values is not None else (
        directive_gravity_values(query, texts, directive_meta)
    )
    for index, bonus in enumerate(gravity_values):
        adjusted[index] += bonus

    # Conservation rule: never displace a stable atomic anchor. Applicable
    # directives are attached by ``transient_directive_bridge`` instead.
    return list(base_order), adjusted


def transient_directive_bridge(
    query: str,
    texts: Sequence[str],
    base_order: Sequence[int],
    directive_meta: Sequence[tuple[str, str] | None],
    *,
    directive_values: Sequence[float] | None = None,
) -> tuple[int, ...] | None:
    """Pair the stable top anchor with one applicable directive, gold-blind."""
    if not base_order:
        return None
    values = list(directive_values) if directive_values is not None else (
        directive_gravity_values(query, texts, directive_meta)
    )
    eligible = [index for index, value in enumerate(values) if value > 0.0]
    if not eligible:
        return None
    directive = min(eligible, key=lambda index: (-values[index], -index))
    bridge = tuple(dict.fromkeys((base_order[0], directive)))
    return bridge if len(bridge) > 1 else None


def fuse_transient_bridge(
    capsule_order: Sequence[tuple[int, ...]],
    bridge: tuple[int, ...] | None,
) -> list[tuple[int, ...]]:
    """Fuse a directive into capsule one without replacing any old source."""
    output = list(capsule_order)
    if bridge is None or not output:
        return output
    output[0] = tuple(dict.fromkeys(output[0] + bridge))
    return output


def lagrange_capsules(
    texts: Sequence[str], groups: Sequence[int], *, max_sources: int = 4,
) -> tuple[list[tuple[int, ...]], dict[int, list[int]]]:
    """Build bounded query-independent episode and rare-entity capsules."""
    capsules: list[tuple[int, ...]] = [(index,) for index in range(len(groups))]
    group_members: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_members[group].append(index)
    for members in group_members.values():
        capsules.extend(tuple(members[index:index + 2]) for index in range(len(members) - 1))
        capsules.extend(
            tuple(members[index:index + max_sources])
            for index in range(0, len(members), max_sources - 1)
            if len(members[index:index + max_sources]) >= 2
        )

    entity_members: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        for entity in {value.casefold() for value in ENTITY_RE.findall(text)}:
            entity_members[entity].append(index)
    for members in entity_members.values():
        unique = list(dict.fromkeys(members))
        if not 2 <= len(unique) <= 64:
            continue
        capsules.extend(
            tuple(unique[index:index + max_sources])
            for index in range(0, len(unique), max_sources - 1)
            if len(unique[index:index + max_sources]) >= 2
        )

    capsules = list(dict.fromkeys(capsules))
    by_source: dict[int, list[int]] = defaultdict(list)
    for capsule_index, capsule in enumerate(capsules):
        for source in capsule:
            by_source[source].append(capsule_index)
    return capsules, by_source
