"""Gold-blind UMD 3.16 state-field and constellation coverage adapter.

The adapter keeps UMD 3.15 as a frozen control.  It adds four query-time laws:

* signed state mass for current-state questions;
* a bounded first-capsule Lagrange selector;
* marginal constellation coverage for broad set questions;
* adaptive, query-derived satellite capacity with a hard ceiling.

No answer, evidence identifier, benchmark category, or evaluator label is an
input to any function in this module.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from benchmarks.public_benchmarks import tokenize, unit_scores


SET_QUERY_RE = re.compile(
    r"\b(?:all|every|each|list|remain(?:s|ing)?|current (?:tasks?|items?|projects?)|"
    r"what (?:tasks?|activities|events|books|movies|places|things|preferences)|"
    r"which (?:tasks?|activities|events|books|movies|places|things)|"
    r"summari[sz]e|overview|in total|complete list|across (?:the|all)|"
    r"todo list|to-do list|upcoming events?|total|how many|how much|"
    r"which (?:week|month|day).*(?:most|least)|meeting my .+ goal|"
    r"recommend(?:ation|ations|ed)?|suggest(?:ion|ions|ed)?|"
    r"based on my preferences|topics I(?:'m| am) interested in|"
    r"genres I enjoy|directors I prefer|artists I like|preferred types?)\b|"
    r"(?:全部|所有|每个|列出|还有哪些|剩余|当前任务|完整列表|总结)",
    re.IGNORECASE,
)
CURRENT_QUERY_RE = re.compile(
    r"\b(?:current|currently|now|latest|still|remain(?:s|ing)?|active|today|"
    r"present|up[- ]to[- ]date|most recent|todo|to-do)\b|"
    r"(?:当前|现在|最新|仍然|剩余|有效|今天)",
    re.IGNORECASE,
)
HISTORICAL_QUERY_RE = re.compile(
    r"\b(?:previously|formerly|used to|historical|history|before|old|past|"
    r"what changed|how .* changed|evolv(?:e|ed|ing))\b|"
    r"(?:以前|过去|历史|曾经|如何变化|演变)",
    re.IGNORECASE,
)
NEGATIVE_STATE_RE = re.compile(
    r"\b(?:delete(?:d)?|remove(?:d)?|cancel(?:led|ed)?|complete(?:d)?|finished|"
    r"done with|no longer|stopped|drop(?:ped)?|abandon(?:ed)?|forget|forgot|"
    r"replaced|superseded|outdated|obsolete|expired)\b|"
    r"(?:删除|移除|取消|完成了|不再|停止|放弃|忘记|替换|过期|作废)",
    re.IGNORECASE,
)
POSITIVE_STATE_RE = re.compile(
    r"\b(?:add(?:ed)?|create(?:d)?|schedule(?:d)?|plan(?:ned)?|want(?:s|ed)?|"
    r"prefer(?:s|red)?|still|remain(?:s|ing)?|active|current|ongoing|keep)\b|"
    r"(?:新增|创建|计划|想要|偏好|仍然|剩余|有效|进行中|保留)",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b|\b[A-Za-z]+[-_.:/]\d+(?:\.\d+)*\b")
STOP = {
    "a", "an", "and", "are", "be", "did", "do", "does", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class PhysicsQuery:
    broad_set: bool
    current_state: bool
    historical: bool


def query_physics(query: str) -> PhysicsQuery:
    """Infer physical retrieval mode from query text only."""
    historical = bool(HISTORICAL_QUERY_RE.search(query))
    return PhysicsQuery(
        broad_set=bool(SET_QUERY_RE.search(query)),
        current_state=bool(CURRENT_QUERY_RE.search(query)) and not historical,
        historical=historical,
    )


def state_mass_adjustment(query: str, texts: Sequence[str]) -> list[float]:
    """Return signed state mass without deleting historical provenance.

    For current-state questions, deletion/completion statements receive a
    negative answer-orbit mass and explicit active statements a small positive
    mass. Historical/evolution questions keep both sides neutral so the full
    transition remains retrievable.
    """
    mode = query_physics(query)
    if not mode.current_state:
        return [0.0] * len(texts)
    values: list[float] = []
    for text in texts:
        negative = bool(NEGATIVE_STATE_RE.search(text))
        positive = bool(POSITIVE_STATE_RE.search(text))
        if negative and not positive:
            values.append(-0.18)
        elif positive and not negative:
            values.append(0.08)
        elif negative and positive:
            # Update sentences are useful audit evidence but ambiguous answer
            # evidence; keep them near neutral instead of discarding them.
            values.append(-0.04)
        else:
            values.append(0.0)
    return values


def apply_state_field(force: Sequence[float], adjustment: Sequence[float]) -> list[float]:
    if len(force) != len(adjustment):
        raise ValueError("force and state adjustment lengths must match")
    return [max(0.0, float(value) + float(delta)) for value, delta in zip(force, adjustment)]


def split_state_orbits(
    query: str,
    capsules: Sequence[tuple[str, ...]],
    source_ids: Sequence[str],
    texts: Sequence[str],
) -> dict[str, list[tuple[str, ...]]]:
    """Separate immutable audit evidence from current answer evidence.

    Negative-mass sources remain in ``audit`` for provenance and transition
    reasoning. They are removed only from ``answer`` for explicit current-state
    questions. Mixed update sources have a small negative mass and therefore
    stay audit-only until a fact-level state compiler resolves their contents.
    """
    adjustments = state_mass_adjustment(query, texts)
    negative = {
        str(source_ids[index]) for index, value in enumerate(adjustments)
        if value < 0.0
    }
    answer = [
        tuple(source for source in capsule if source not in negative)
        for capsule in capsules
    ]
    return {
        "audit": [tuple(capsule) for capsule in capsules],
        "answer": [capsule for capsule in answer if capsule],
    }


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOP and len(token) > 1}


def _entity_set(text: str) -> set[str]:
    return {value.casefold() for value in ENTITY_RE.findall(text)}


def first_capsule_lagrange_order(
    query: str,
    capsules: Sequence[tuple[int, ...]],
    texts: Sequence[str],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    entity: Sequence[float],
    temporal: Sequence[float],
    *,
    candidate_limit: int = 10,
    max_tethers: int = 2,
) -> list[tuple[int, ...]]:
    """Tether one high-confidence source to L1 without moving any capsule.

    The selected source remains in its home capsule, so every old prefix source
    set is conserved while L1 can gain one evidence-bearing satellite. The
    total number of unique retrieved sources is unchanged.
    """
    if not capsules:
        return []
    query_tokens = _content_tokens(query)
    limit = min(max(1, candidate_limit), len(capsules))
    scores: list[float] = []
    for rank, capsule in enumerate(capsules[:limit]):
        members = list(capsule)
        union_tokens = set().union(*(_content_tokens(texts[i]) for i in members))
        coverage = len(query_tokens & union_tokens) / max(1, len(query_tokens))
        member_values = [
            0.34 * force[i]
            + 0.30 * semantic[i]
            + 0.17 * lexical[i]
            + 0.11 * entity[i]
            + 0.08 * temporal[i]
            for i in members
        ]
        peak = max(member_values, default=0.0)
        density = sum(sorted(member_values, reverse=True)[:2]) / max(1, min(2, len(member_values)))
        rank_mass = 1.0 / (1.0 + rank)
        scores.append(0.48 * peak + 0.27 * density + 0.19 * coverage + 0.06 * rank_mass)
    winners = [
        index for index in sorted(range(1, limit), key=lambda i: (-scores[i], i))
        if scores[index] > scores[0]
    ][:max(0, max_tethers)]
    if not winners:
        return list(capsules)
    output = list(capsules)
    tethered: list[int] = []
    for winner in winners:
        tethered.append(max(
            capsules[winner],
            key=lambda i: (
                0.34 * force[i]
                + 0.30 * semantic[i]
                + 0.17 * lexical[i]
                + 0.11 * entity[i]
                + 0.08 * temporal[i],
                -i,
            ),
        ))
    output[0] = tuple(dict.fromkeys(output[0] + tuple(tethered)))
    return output


def constellation_budget(query: str, *, default: int = 16, ceiling: int = 48) -> int:
    """Allocate more satellites only for query-detectable collection breadth."""
    if ceiling <= 0:
        return 0
    mode = query_physics(query)
    if mode.broad_set:
        return min(40, ceiling)
    if mode.historical:
        return min(24, ceiling)
    return min(default, ceiling)


def conserved_constellation_orbit(
    stable_orbit: Sequence[int], coverage_orbit: Sequence[int], *, conserved: int = 16,
) -> list[int]:
    """Reorder but never discard the frozen 3.15 secondary evidence set.

    The first ``conserved`` sources are the old event satellites. Their order
    is immutable because striping order determines every capsule prefix.
    Coverage energy may only append novel sources.
    """
    protected = list(dict.fromkeys(stable_orbit[:max(0, conserved)]))
    protected_set = set(protected)
    return protected + [source for source in coverage_orbit if source not in protected_set]


def rank_constellation_coverage(
    query: str,
    atomic_orbit: Sequence[int],
    stable_capsules: Sequence[tuple[int, ...]],
    texts: Sequence[str],
    groups: Sequence[int],
    dates: Sequence[str],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    pool_size: int = 192,
    source_token_sets: Sequence[set[str] | frozenset[str]] | None = None,
) -> list[int]:
    """Greedy marginal coverage orbit for broad, multi-fact questions.

    The law rewards relevance first, then uncovered entities, content facets,
    sessions and time phases. Redundant candidates lose potential energy. This
    is a deterministic, label-blind submodular approximation.
    """
    stable_sources = {source for capsule in stable_capsules for source in capsule}
    pool = [source for source in atomic_orbit if source not in stable_sources][:pool_size]
    if not pool:
        return []
    mode = query_physics(query)
    if not mode.broad_set:
        return sorted(
            pool,
            key=lambda i: (-(0.46 * force[i] + 0.34 * semantic[i] + 0.20 * lexical[i]), i),
        )

    feature_sources = stable_sources | set(pool)
    tokens_by_source = {
        source: (
            {
                token for token in source_token_sets[source]
                if token not in STOP and len(token) > 1
            }
            if source_token_sets is not None and source < len(source_token_sets)
            else _content_tokens(texts[source])
        )
        for source in feature_sources
    }
    entities_by_source = {
        source: _entity_set(texts[source]) for source in feature_sources
    }
    force_unit = unit_scores([force[i] for i in pool])
    semantic_unit = unit_scores([semantic[i] for i in pool])
    lexical_unit = unit_scores([lexical[i] for i in pool])
    base = {
        source: 0.44 * force_unit[pos] + 0.36 * semantic_unit[pos] + 0.20 * lexical_unit[pos]
        for pos, source in enumerate(pool)
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_tokens = set().union(*(tokens_by_source[i] for i in stable_sources)) if stable_sources else set()
    covered_entities = set().union(*(entities_by_source[i] for i in stable_sources)) if stable_sources else set()
    covered_groups = {groups[i] for i in stable_sources}
    covered_dates = {
        dates[groups[i]] for i in stable_sources
        if groups[i] < len(dates) and dates[groups[i]]
    }
    query_tokens = _content_tokens(query)

    # UMD 3.29 incremental coverage field.  The original implementation
    # rebuilt ``tokens & covered_tokens`` and ``tokens | covered_tokens`` for
    # every remaining source on every greedy step.  That is mathematically
    # correct, but it copies large token sets O(pool^2) times on long-memory
    # workloads.  Keep the exact same potential-energy equation while
    # maintaining each source's overlap mass through a local inverted index.
    # This changes execution cost, not ordering semantics.
    token_postings: dict[str, list[int]] = defaultdict(list)
    overlap_mass: dict[int, int] = {}
    token_mass = {source: len(tokens_by_source[source]) for source in pool}
    for source in pool:
        tokens = tokens_by_source[source]
        overlap_mass[source] = len(tokens & covered_tokens)
        for token in tokens:
            token_postings[token].append(source)

    while len(selected) < len(pool):
        best = None
        best_score = float("-inf")
        for source in pool:
            if source in selected_set:
                continue
            tokens = tokens_by_source[source]
            entities = entities_by_source[source]
            group = groups[source]
            date = dates[group] if group < len(dates) else ""
            query_coverage = len(tokens & query_tokens) / max(1, len(query_tokens))
            overlap = overlap_mass[source]
            source_token_mass = token_mass[source]
            new_tokens = (source_token_mass - overlap) / max(1, source_token_mass)
            redundancy = overlap / max(
                1, source_token_mass + len(covered_tokens) - overlap,
            )
            score = (
                base[source]
                + 0.10 * query_coverage
                + 0.09 * bool(entities - covered_entities)
                + 0.07 * (group not in covered_groups)
                + 0.04 * bool(date and date not in covered_dates)
                + 0.08 * new_tokens
                - 0.13 * redundancy
            )
            if score > best_score or (score == best_score and (best is None or source < best)):
                best, best_score = source, score
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        tokens = tokens_by_source[best]
        newly_covered = tokens - covered_tokens
        for token in newly_covered:
            for source in token_postings.get(token, ()):
                overlap_mass[source] += 1
        covered_tokens.update(newly_covered)
        covered_entities.update(entities_by_source[best])
        covered_groups.add(groups[best])
        group = groups[best]
        if group < len(dates) and dates[group]:
            covered_dates.add(dates[group])
    return selected
