"""Gold-blind UMD 3.17 fact planets and bounded focus gravity.

UMD 3.16 retrieves provenance-bearing dialogue sources.  UMD 3.17 compiles
each source into smaller, query-independent fact atoms and uses those atoms as
an additional coverage field.  Evidence is still returned at the original
source granularity; atoms never replace, invent, or detach provenance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from benchmarks.public_benchmarks import tokenize, unit_scores
from benchmarks.umd316_adapter import (
    NEGATIVE_STATE_RE,
    POSITIVE_STATE_RE,
    _content_tokens,
    _entity_set,
    query_physics,
)


CLAUSE_RE = re.compile(r"(?:[.!?;]+|\n+|\s+(?:but|however|although|while)\s+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\w)(?:\$|£|€)?\d+(?:[.,:]\d+)*(?:%|[a-z]{1,4})?(?!\w)", re.IGNORECASE)
GENERIC_QUERY = {
    "all", "current", "every", "list", "memory", "remain", "remaining",
    "thing", "things", "total", "tell", "summarize", "summary",
}


@dataclass(frozen=True)
class FactAtom:
    tokens: frozenset[str]
    entities: frozenset[str]
    numbers: frozenset[str]
    state: int


def extract_fact_atoms(text: str) -> tuple[FactAtom, ...]:
    """Compile deterministic fact atoms without using a query or labels."""
    atoms: list[FactAtom] = []
    for clause in CLAUSE_RE.split(text):
        tokens = frozenset(_content_tokens(clause))
        entities = frozenset(_entity_set(clause))
        numbers = frozenset(value.casefold() for value in NUMBER_RE.findall(clause))
        if not tokens and not entities and not numbers:
            continue
        negative = bool(NEGATIVE_STATE_RE.search(clause))
        positive = bool(POSITIVE_STATE_RE.search(clause))
        state = -1 if negative and not positive else (1 if positive and not negative else 0)
        atoms.append(FactAtom(tokens, entities, numbers, state))
    if atoms:
        return tuple(atoms)
    # An empty/very short source remains a provenance-bearing atom.
    return (FactAtom(frozenset(_content_tokens(text)), frozenset(), frozenset(), 0),)


def compile_fact_planets(texts: Sequence[str]) -> list[tuple[FactAtom, ...]]:
    return [extract_fact_atoms(text) for text in texts]


def fact_constellation_budget(query: str, *, default: int = 16, ceiling: int = 48) -> int:
    """Reserve eight additive fact satellites only for broad set queries."""
    mode = query_physics(query)
    if mode.broad_set:
        return min(48, ceiling)
    if mode.historical:
        return min(24, ceiling)
    return min(default, ceiling)


def conserved_fact_constellation(
    stable_orbit: Sequence[int],
    legacy_coverage: Sequence[int],
    fact_coverage: Sequence[int],
    *,
    conserved: int = 16,
    legacy_budget: int = 40,
) -> list[int]:
    """Keep the complete 3.16 budget, then append novel fact satellites."""
    protected = list(dict.fromkeys(stable_orbit[:max(0, conserved)]))
    protected_set = set(protected)
    for source in legacy_coverage:
        if source not in protected_set:
            protected.append(source)
            protected_set.add(source)
        if len(protected) >= legacy_budget:
            break
    extras = [source for source in fact_coverage if source not in protected_set]
    tail = [
        source for source in legacy_coverage
        if source not in protected_set and source not in extras
    ]
    return protected + extras + tail


def _atom_overlap(atom: FactAtom, query_tokens: set[str], query_entities: set[str], query_numbers: set[str]) -> float:
    token_overlap = len(atom.tokens & query_tokens) / max(1, len(query_tokens))
    entity_overlap = len(atom.entities & query_entities) / max(1, len(query_entities))
    number_overlap = len(atom.numbers & query_numbers) / max(1, len(query_numbers))
    return 0.62 * token_overlap + 0.25 * entity_overlap + 0.13 * number_overlap


def first_capsule_focus_tether(
    query: str,
    capsules: Sequence[tuple[int, ...]],
    facts: Sequence[tuple[FactAtom, ...]],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    candidate_limit: int = 10,
) -> list[tuple[int, ...]]:
    """Add at most one fact-focused source to L1; never move/remove sources."""
    if not capsules:
        return []
    query_tokens = _content_tokens(query) - GENERIC_QUERY
    query_entities = _entity_set(query)
    query_numbers = {value.casefold() for value in NUMBER_RE.findall(query)}
    first = set(capsules[0])
    candidates = list(dict.fromkeys(
        source for capsule in capsules[:max(1, candidate_limit)] for source in capsule
    ))
    if not candidates:
        return list(capsules)

    # Rarity behaves like gravitational density: a shared rare facet has more
    # pull than dialogue boilerplate repeated throughout the candidate system.
    df: dict[str, int] = {}
    for source in candidates:
        source_tokens = set().union(*(atom.tokens for atom in facts[source]))
        for token in source_tokens & query_tokens:
            df[token] = df.get(token, 0) + 1
    rare_mass = {
        token: math.log(1.0 + (len(candidates) + 0.5) / (count + 0.5))
        for token, count in df.items()
    }
    total_rare = sum(rare_mass.values()) or 1.0

    def score(source: int) -> tuple[float, float, int]:
        atoms = facts[source]
        atom_peak = max(
            (_atom_overlap(atom, query_tokens, query_entities, query_numbers) for atom in atoms),
            default=0.0,
        )
        source_tokens = set().union(*(atom.tokens for atom in atoms))
        rare = sum(rare_mass.get(token, 0.0) for token in source_tokens & query_tokens) / total_rare
        value = (
            0.31 * force[source] + 0.23 * semantic[source] + 0.13 * lexical[source]
            + 0.21 * atom_peak + 0.12 * rare
        )
        return value, atom_peak, -source

    alternatives = [source for source in candidates if source not in first]
    if not alternatives:
        return list(capsules)
    winner = max(alternatives, key=score)
    winner_value, winner_overlap, _ = score(winner)
    first_value = max((score(source)[0] for source in first), default=0.0)
    # Require actual query/fact contact and a competitive field. The tether is
    # additive, so recall prefixes are mathematically monotonic.
    if winner_overlap < 0.08 or winner_value + 0.02 < first_value:
        return list(capsules)
    output = list(capsules)
    output[0] = tuple(dict.fromkeys(output[0] + (winner,)))
    return output


def rank_fact_planet_coverage(
    query: str,
    atomic_orbit: Sequence[int],
    stable_capsules: Sequence[tuple[int, ...]],
    facts: Sequence[tuple[FactAtom, ...]],
    groups: Sequence[int],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    pool_size: int = 256,
) -> list[int]:
    """Rank extra sources by relevant fact coverage and state-aware novelty."""
    stable_sources = {source for capsule in stable_capsules for source in capsule}
    pool = [source for source in atomic_orbit if source not in stable_sources][:pool_size]
    if not pool:
        return []
    mode = query_physics(query)
    if not mode.broad_set:
        return sorted(
            pool,
            key=lambda source: (
                -(0.46 * force[source] + 0.34 * semantic[source] + 0.20 * lexical[source]),
                source,
            ),
        )

    query_tokens = _content_tokens(query) - GENERIC_QUERY
    query_entities = _entity_set(query)
    query_numbers = {value.casefold() for value in NUMBER_RE.findall(query)}
    force_unit = unit_scores([force[source] for source in pool])
    semantic_unit = unit_scores([semantic[source] for source in pool])
    lexical_unit = unit_scores([lexical[source] for source in pool])
    base = {
        source: 0.40 * force_unit[pos] + 0.34 * semantic_unit[pos] + 0.16 * lexical_unit[pos]
        for pos, source in enumerate(pool)
    }

    def keys(source: int) -> set[tuple]:
        output: set[tuple] = set()
        for atom in facts[source]:
            relevant = atom.tokens & query_tokens
            # Fact signatures retain multiple independent details within the
            # same session instead of collapsing them to one bag of words.
            # Keep both the query anchor and the value-bearing remainder. Two
            # project-task atoms must not collapse merely because they share
            # the words "project" and "task".
            remainder = atom.tokens - query_tokens
            signature_tokens = tuple(
                sorted(relevant)[:4] + sorted(remainder)[:4]
            )
            output.add((signature_tokens, tuple(sorted(atom.entities)[:3]), tuple(sorted(atom.numbers)[:3]), atom.state))
        return output

    covered = set().union(*(keys(source) for source in stable_sources)) if stable_sources else set()
    covered_groups = {groups[source] for source in stable_sources}
    selected: list[int] = []
    remaining = set(pool)
    while remaining:
        best_source = -1
        best_score = float("-inf")
        for source in pool:
            if source not in remaining:
                continue
            source_keys = keys(source)
            novelty = len(source_keys - covered) / max(1, len(source_keys))
            atom_relevance = max(
                (_atom_overlap(atom, query_tokens, query_entities, query_numbers) for atom in facts[source]),
                default=0.0,
            )
            active_mass = max((atom.state for atom in facts[source]), default=0)
            state_term = 0.04 * active_mass if mode.current_state else 0.0
            score = (
                base[source] + 0.18 * atom_relevance + 0.14 * novelty
                + 0.05 * (groups[source] not in covered_groups) + state_term
            )
            if score > best_score or (score == best_score and source < best_source):
                best_source, best_score = source, score
        if best_source < 0:
            break
        selected.append(best_source)
        remaining.remove(best_source)
        covered.update(keys(best_source))
        covered_groups.add(groups[best_source])
    return selected
