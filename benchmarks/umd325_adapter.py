"""UMD 3.25 antimatter forgetting and galactic-census retrieval.

The adapter is deterministic and gold-blind.  It treats explicit invalidation
events as negative mass in the answer orbit while retaining them in an audit
orbit.  Broad aggregate questions receive a provenance-preserving census
budget large enough for hundreds of independent source sessions.
"""

from __future__ import annotations

import re
from typing import Sequence

from benchmarks.public_benchmarks import tokenize, unit_scores
from benchmarks.umd316_adapter import HISTORICAL_QUERY_RE, query_physics


AGGREGATE_CENSUS_RE = re.compile(
    r"\b(?:total|sum|average|how many|how much|spent|spending|expenses?|"
    r"which (?:week|month|day).*(?:most|least|highest|lowest)|"
    r"most|least|maximum|minimum|across (?:the|all)|meeting my .+ goal)\b",
    re.IGNORECASE,
)
EXPLICIT_ANTIMATTER_RE = re.compile(
    r"\boperation\s+(?:delete|remove|forget|cancel)|"
    r"\bsession operation\s+(?:delete|remove|forget|cancel)\b|"
    r"\bstatus\s+(?:deleted|removed|forgotten|cancelled|canceled|obsolete)\b",
    re.IGNORECASE,
)

ALIASES = {
    "spend": {"expense", "expenses", "amount", "cost"},
    "spending": {"expense", "expenses", "amount", "cost"},
    "spent": {"expense", "expenses", "amount", "cost"},
    "food": {"breakfast", "lunch", "dinner", "coffee", "snack"},
    "task": {"tasks", "todo", "description"},
    "tasks": {"task", "todo", "description"},
    "todo": {"task", "tasks", "description"},
    "steps": {"step", "stepcount", "step_count"},
    "books": {"book", "reading", "read"},
    "book": {"books", "author", "authors", "genre", "genres", "read", "reading", "already"},
    "movies": {"movie", "film", "films", "genre", "genres", "actor", "actors", "director", "directors", "watched"},
    "movie": {"movies", "film", "films", "genre", "genres", "actor", "actors", "director", "directors", "watched"},
    "music": {"artist", "artists", "genre", "genres", "decade", "decades", "listened"},
    "destination": {"destinations", "visited", "country", "countries", "travel", "types"},
    "destinations": {"destination", "visited", "country", "countries", "travel", "types"},
    "travel": {"destination", "destinations", "visited", "country", "countries", "types"},
    "preferences": {"preference", "prefer", "preferred", "like"},
}

DOMAIN_TERMS = {
    "movie": {"movie", "movies", "film", "films", "actor", "actors", "director", "directors", "watched"},
    "book": {"book", "books", "author", "authors", "read", "reading"},
    "music": {"music", "song", "songs", "album", "albums", "artist", "artists", "listened", "decade", "decades"},
    "destination": {"destination", "destinations", "travel", "visited", "country", "countries", "region", "regions", "site", "sites", "climate", "climates", "ceremony", "ceremonies", "archaeological"},
    "travel": {"destination", "destinations", "travel", "trip", "visited", "country", "countries", "region", "regions", "site", "sites", "climate", "climates", "types"},
    "visit": {"destination", "destinations", "travel", "trip", "visited", "country", "countries", "region", "regions", "site", "sites", "climate", "climates", "types"},
    "place": {"destination", "destinations", "travel", "trip", "visited", "country", "countries", "region", "regions", "site", "sites", "climate", "climates", "types"},
}


def _expanded_query_tokens(query: str) -> set[str]:
    tokens = set(tokenize(query))
    expanded = set(tokens)
    for token in tokens:
        alias_values: set[str] = set()
        for key, values in ALIASES.items():
            if token in tokenize(key.replace("_", " ")):
                alias_values.update(values)
        for alias in alias_values:
            expanded.update(tokenize(alias.replace("_", " ")))
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def _domain_terms(query: str) -> set[str]:
    query_tokens = set(tokenize(query))
    query_tokens.update(token for value in list(query_tokens) for token in tokenize(value))
    output: set[str] = set()
    for anchor, values in DOMAIN_TERMS.items():
        anchor_tokens = set(tokenize(anchor))
        anchor_tokens.update(token for value in list(anchor_tokens) for token in tokenize(value))
        if query_tokens & anchor_tokens:
            for value in values:
                output.update(tokenize(value))
    return output


def galactic_census_budget(query: str, source_count: int, *, default: int = 32) -> int:
    """Allocate provenance capacity from query breadth, never gold size."""
    count = max(0, int(source_count))
    if AGGREGATE_CENSUS_RE.search(query):
        return min(count, 512)
    mode = query_physics(query)
    if mode.broad_set:
        return min(count, 160)
    if mode.historical:
        return min(count, 96)
    return min(count, max(1, int(default)))


def is_explicit_antimatter(text: str) -> bool:
    """Identify an explicit structured invalidation event."""
    return bool(EXPLICIT_ANTIMATTER_RE.search(text))


def rank_galactic_census(
    query: str,
    atomic_orbit: Sequence[int],
    stable_capsules: Sequence[tuple[int, ...]],
    texts: Sequence[str],
    source_token_sets: Sequence[frozenset[str]],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    extra_query_tokens: set[str] | None = None,
    broad_override: bool = False,
) -> list[int]:
    """Rank the complete horizon for large-set provenance recovery.

    Unlike the 3.17 coverage field, this law does not truncate its candidate
    universe at 256.  Structured category contact dominates aggregate queries;
    current mutable sets additionally favor recent positive events.  Explicit
    invalidations remain addressable but fall to the tail of the answer census.
    """
    stable = {source for capsule in stable_capsules for source in capsule}
    pool = [source for source in atomic_orbit if source not in stable]
    if not pool:
        return []

    qtokens = _expanded_query_tokens(query) | set(extra_query_tokens or ())
    domain_terms = _domain_terms(query)
    mode = query_physics(query)
    aggregate = bool(AGGREGATE_CENSUS_RE.search(query))
    force_u = unit_scores([force[source] for source in pool])
    semantic_u = unit_scores([semantic[source] for source in pool])
    lexical_u = unit_scores([lexical[source] for source in pool])
    denominator = max(1, len(qtokens))
    last = max(1, len(texts) - 1)
    scored: list[tuple[float, int]] = []
    for position, source in enumerate(pool):
        if source < len(source_token_sets):
            source_tokens = source_token_sets[source]
        else:
            source_tokens = set(tokenize(texts[source]))
        contact = len(source_tokens & qtokens) / denominator
        exact_contacts = len(source_tokens & qtokens)
        domain_contacts = len(source_tokens & domain_terms)
        recency = source / last
        invalid = is_explicit_antimatter(texts[source])
        positive = bool(re.search(r"\boperation\s+add\b", texts[source], re.IGNORECASE))
        if aggregate:
            score = (
                0.37 * lexical_u[position] + 0.20 * force_u[position]
                + 0.13 * semantic_u[position] + 0.22 * contact
                + 0.02 * min(4, exact_contacts) + 0.06 * float(positive)
            )
        elif mode.broad_set or broad_override:
            score = (
                0.30 * lexical_u[position] + 0.22 * force_u[position]
                + 0.14 * semantic_u[position] + 0.20 * contact
                + 0.08 * recency + 0.06 * float(positive)
                + 0.62 * float(domain_contacts > 0) + 0.03 * min(4, domain_contacts)
            )
        else:
            score = 0.43 * force_u[position] + 0.34 * semantic_u[position] + 0.23 * lexical_u[position]
        if invalid and not mode.historical:
            score -= 2.0
        scored.append((score, source))
    return [source for _, source in sorted(scored, key=lambda item: (-item[0], item[1]))]


def split_antimatter_orbits(
    query: str,
    capsules: Sequence[tuple[str, ...]],
    source_ids: Sequence[str],
    texts: Sequence[str],
    *,
    negative_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, list[tuple[str, ...]]]:
    """Keep invalidations in audit history but remove them from live answers."""
    audit = [tuple(capsule) for capsule in capsules]
    if HISTORICAL_QUERY_RE.search(query):
        return {"audit": audit, "answer": audit}
    if negative_ids is None:
        negative_ids = {
            str(source_ids[index])
            for index, text in enumerate(texts)
            if is_explicit_antimatter(text)
        }
    answer = [
        tuple(source for source in capsule if source not in negative_ids)
        for capsule in capsules
    ]
    return {"audit": audit, "answer": [capsule for capsule in answer if capsule]}
