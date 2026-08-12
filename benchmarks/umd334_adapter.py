"""UMD 3.34 query fission and open-text semantic periapsis.

The field observes only the visible query and source text.  It never receives
answers, evidence identifiers, benchmark categories, or evaluator feedback.
Compound questions are split into bounded semantic bodies; direct memory
questions also receive conservative declarative rewrites.  The resulting
orbits are used for candidate discovery and confidence-gated rank-one repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from benchmarks.public_benchmarks import BM25, ranking, tokenize, unit_scores


_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s*(?:;|\?|\.(?=\s+[A-Z])|\b(?:and|but|while|whereas)\b)\s*",
    re.IGNORECASE,
)
_QUESTION_HEAD_RE = re.compile(
    r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+"
    r"(?:tell|remind|show)\s+me\s+",
    re.IGNORECASE,
)
_CONTENT_WORDS = {
    "what", "where", "when", "which", "who", "whose", "how", "why",
    "did", "does", "do", "was", "were", "is", "are", "have", "has",
    "had", "my", "i", "me", "the", "a", "an", "of", "to", "in",
    "on", "at", "for", "from", "with", "about", "that", "this",
}
_VERB_HINT_RE = re.compile(
    r"\b(?:did|does|do|was|were|is|are|have|has|had|went|go|got|bought|"
    r"made|created|changed|started|finished|attended|worked|lived|used|"
    r"prefer|liked|wanted|planned|graduated|spent|took|take)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryFissionResult:
    subqueries: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    discovery_sources: tuple[int, ...] = ()
    coverage_sources: tuple[int, ...] = ()
    semantic_scores: tuple[float, ...] = ()
    periapsis_scores: tuple[float, ...] = ()
    contact_scores: tuple[float, ...] = ()
    periapsis_source: int | None = None
    confidence_ratio: float = 1.0
    confidence_margin: float = 0.0
    episodic_nucleus_active: bool = False


def _clean_query(query: str) -> str:
    cleaned = " ".join(str(query).strip().split())
    return _QUESTION_HEAD_RE.sub("", cleaned).strip(" .?")


def _content_count(value: str) -> int:
    return sum(
        1 for token in tokenize(value)
        if len(token) > 2 and token not in _CONTENT_WORDS
    )


def decompose_query(query: str, *, maximum: int = 4) -> tuple[str, ...]:
    """Split only clause-like conjunctions, preserving short noun phrases."""
    cleaned = _clean_query(query)
    if not cleaned:
        return ()
    pieces = [piece.strip(" ,.") for piece in _CLAUSE_BOUNDARY_RE.split(cleaned)]
    valid = [
        piece for piece in pieces
        if _content_count(piece) >= 2 and _VERB_HINT_RE.search(piece)
    ]
    if len(valid) < 2:
        return (cleaned,)
    # The complete question is conserved as the central star; clauses are
    # satellites and cannot erase relational meaning such as before/after.
    return tuple(dict.fromkeys([cleaned, *valid[:maximum]]))


def declarative_variants(query: str) -> tuple[str, ...]:
    """Create bounded first-person declarative forms for symmetric encoders."""
    cleaned = _clean_query(query)
    if not cleaned:
        return ()
    output = [cleaned]
    rules = (
        (r"^what (.+?) did i (.+)$", r"I \2 \1"),
        (r"^where did i (.+)$", r"I \1 at a place"),
        (r"^when did i (.+)$", r"I \1 on a date"),
        (r"^how much did i (.+)$", r"I \1 an amount"),
        (r"^how long did (?:it take me to|i) (.+)$", r"I \1 for a duration"),
        (r"^how many (.+?) did i (.+)$", r"I \2 a number of \1"),
        (r"^what was my (.+)$", r"My \1 was"),
        (r"^what is the name of (.+)$", r"The name of \1 is"),
        (r"^who did i (.+)$", r"I \1 a person"),
    )
    for pattern, replacement in rules:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            output.append(re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE))
            break
    return tuple(dict.fromkeys(value.strip(" .?") for value in output if value.strip()))


_USER_TURN_RE = re.compile(
    r"(?:^|\s)user:\s*(.*?)(?=\s+(?:assistant|system|tool|user):|$)",
    re.IGNORECASE | re.DOTALL,
)


def episodic_nucleus(text: str) -> str:
    """Keep autobiographical user turns while retaining non-dialogue sources."""
    turns = [" ".join(match.split()) for match in _USER_TURN_RE.findall(text)]
    return " ".join(turn for turn in turns if turn) or text


def _round_robin(orders: Sequence[Sequence[int]], limit: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(output) < limit and any(depth < len(order) for order in orders):
        for order in orders:
            if depth >= len(order):
                continue
            source = int(order[depth])
            if source not in seen:
                output.append(source)
                seen.add(source)
                if len(output) >= limit:
                    break
        depth += 1
    return output


class QueryFissionField:
    """Bounded query-side superposition over an existing visible index."""

    def __init__(self, encoder: Any, *, discovery_budget: int = 96) -> None:
        self.encoder = encoder
        self.discovery_budget = max(16, int(discovery_budget))
        self._nucleus_vectors: dict[int, np.ndarray] = {}

    def solve(
        self, query: str, *, texts: Sequence[str], lexical_index: Any,
        source_vectors: dict[int, np.ndarray], base_candidates: Sequence[int],
    ) -> QueryFissionResult:
        size = len(texts)
        if not size:
            return QueryFissionResult()
        subqueries = decompose_query(query)
        variants = tuple(dict.fromkeys(
            variant
            for subquery in subqueries
            for variant in declarative_variants(subquery)
        ))
        if not variants:
            return QueryFissionResult()

        lexical_fields = [unit_scores(lexical_index.scores(value)) for value in subqueries]
        lexical_orders = [ranking(values) for values in lexical_fields]
        nuclei = [episodic_nucleus(text) for text in texts]
        nucleus_active = any(_USER_TURN_RE.search(text) for text in texts)
        nucleus_index = BM25(nuclei)
        nucleus_fields = [unit_scores(nucleus_index.scores(value)) for value in subqueries]
        nucleus_orders = [ranking(values) for values in nucleus_fields]
        discovery = _round_robin(
            [
                list(base_candidates),
                *(order[:20] for order in lexical_orders),
                *(order[:20] for order in nucleus_orders),
            ],
            min(size, self.discovery_budget),
        )
        missing = [source for source in discovery if source not in source_vectors]
        if missing:
            encoded = self.encoder.encode_many([texts[source] for source in missing])
            source_vectors.update(zip(missing, encoded))

        missing_nuclei = [source for source in discovery if source not in self._nucleus_vectors]
        if missing_nuclei:
            encoded = self.encoder.encode_many([nuclei[source] for source in missing_nuclei])
            self._nucleus_vectors.update(zip(missing_nuclei, encoded))

        query_vectors = self.encoder.encode_many(variants)
        semantic = [0.0] * size
        full_semantic = [0.0] * size
        for source in discovery:
            nucleus_similarities = [
                max(0.0, float(np.dot(vector, self._nucleus_vectors[source])))
                for vector in query_vectors
            ]
            full_similarities = [
                max(0.0, float(np.dot(vector, source_vectors[source])))
                for vector in query_vectors
            ]
            # Conserve the raw question while allowing one paraphrase to pull
            # a semantically equivalent source closer to periapsis.
            semantic[source] = 0.70 * nucleus_similarities[0] + 0.30 * max(nucleus_similarities)
            full_semantic[source] = 0.70 * full_similarities[0] + 0.30 * max(full_similarities)
        semantic_units = unit_scores(semantic)
        full_semantic_units = unit_scores(full_semantic)
        lexical_central = lexical_fields[0]
        nucleus_central = nucleus_fields[0]
        periapsis = [
            0.50 * semantic_units[source]
            + 0.30 * nucleus_central[source]
            + 0.12 * full_semantic_units[source]
            + 0.08 * lexical_central[source]
            for source in range(size)
        ]
        periapsis_order = ranking(periapsis)
        top = periapsis_order[0] if periapsis_order else None
        second = periapsis_order[1] if len(periapsis_order) > 1 else top
        top_score = periapsis[top] if top is not None else 0.0
        second_score = periapsis[second] if second is not None else 0.0
        ratio = top_score / max(1e-9, second_score)
        margin = top_score - second_score
        coverage = _round_robin(
            [order[:12] for order in lexical_orders], min(size, 32),
        )
        return QueryFissionResult(
            subqueries=subqueries,
            variants=variants,
            discovery_sources=tuple(discovery),
            coverage_sources=tuple(coverage),
            semantic_scores=tuple(semantic_units),
            periapsis_scores=tuple(periapsis),
            contact_scores=tuple(nucleus_central),
            periapsis_source=top,
            confidence_ratio=ratio,
            confidence_margin=margin,
            episodic_nucleus_active=nucleus_active,
        )


def should_promote_periapsis(
    field: QueryFissionResult, current_source: int, lexical_scores: Sequence[float],
    *, ratio: float = 1.08, margin: float = 0.055,
) -> bool:
    """Gold-blind dual gate for changing the strict atomic first source."""
    candidate = field.periapsis_source
    if not field.episodic_nucleus_active:
        return False
    if candidate is None or candidate == current_source:
        return False
    if field.confidence_ratio < ratio or field.confidence_margin < margin:
        return False
    # A declarative semantic win must not contradict the central lexical field.
    contact = field.contact_scores or tuple(lexical_scores)
    return contact[candidate] >= 0.55 * contact[current_source]
