"""UMD 3.31 gold-blind evidence-closure gravitational field.

UMD 3.30 proved that a bounded fact slingshot can recover multi-hop answers,
but it selected the terminal relation from the longest cue anywhere in the
question.  That often confused an intermediate dependency ("spouse") with
the requested output ("where did ... pass away").  UMD 3.31 derives the
terminal relation from the question's output slot, extends the small
deterministic fact grammar, and emits a source-atomic closure orbit consisting
of the answer fact, normalized answer echoes, and path provenance.

The adapter observes only the query and visible source text.  Gold answers and
evidence identifiers are evaluator-only inputs and are never accepted here.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

from benchmarks.umd330_adapter import (
    NUMBERED_FACT_RE,
    GraphFact,
    _norm,
    _parse_statement as parse_v330_statement,
    _relation_contact,
    _target_relations as target_v330_relations,
)


# These templates account for nearly all statements left outside the 3.30
# grammar in MemoryAgentBench Conflict Resolution.  They remain deliberately
# narrow so ordinary prose cannot accidentally become a fact universe.
EXTRA_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("worked_city", re.compile(r"^(.+?) worked in the city of (.+?)\.?$")),
    (
        "original_broadcaster",
        re.compile(r"^The orig(?:ianl|inal) broadcaster of (.+?) is (.+?)\.?$"),
    ),
    ("head_government", re.compile(r"^The Prime Minister of (.+?) is (.+?)\.?$")),
    ("head_state", re.compile(r"^The President of (.+?) is (.+?)\.?$")),
    ("head_government", re.compile(r"^The Governor of (.+?) is (.+?)\.?$")),
    ("head_government", re.compile(r"^The Mayor of (.+?) is (.+?)\.?$")),
)


# Ordered output-slot rules.  Earlier expressions are more explicit.  A rule
# can expose two compatible relations when surface language is ambiguous; the
# bounded graph, query contact, and path energy decide between them.
OUTPUT_SLOT_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:which|what|to which) continent\b|\bcontinent of origin\b", re.I), ("continent",)),
    (re.compile(r"\b(?:pass away|passed away|died|death)\b", re.I), ("died_city",)),
    (re.compile(r"\bcapital(?: city)?\b", re.I), ("capital",)),
    (re.compile(r"\b(?:which|what|from which) country\b.*\b(?:sport|origin|birthplace|come from|came from)\b", re.I), ("created_country",)),
    (re.compile(r"\b(?:birthplace|born in)\b", re.I), ("born_city", "created_country")),
    (re.compile(r"\b(?:work location|location of work|city .+ work(?:ed)?|where .+ work(?:ed)?)\b", re.I), ("worked_city",)),
    (re.compile(r"\b(?:official language|language is recognized|language is used)\b", re.I), ("official_language",)),
    (re.compile(r"\b(?:original language)\b", re.I), ("original_language",)),
    (re.compile(r"\b(?:written|work|book|notable work).{0,50}\blanguage\b|\bwhat language\b", re.I), ("written_language", "original_language")),
    (re.compile(r"\b(?:religious affiliation|religion|faith)\b", re.I), ("religion",)),
    (re.compile(r"\b(?:what|which) sport\b|\bsport that\b", re.I), ("sport",)),
    (re.compile(r"\b(?:job title|playing position|which position|what position)\b", re.I), ("position", "works_field")),
    (re.compile(r"\b(?:occupation|profession|field of work|works? in which field)\b", re.I), ("works_field",)),
    (re.compile(r"\b(?:chief|head) of state\b|\bchief public representative\b", re.I), ("head_state",)),
    (re.compile(r"\b(?:head of government|government leader|prime minister)\b", re.I), ("head_government",)),
    (re.compile(r"\bchief executive\b.*\b(?:country|government)\b|\b(?:country|government)\b.*\bchief executive\b", re.I), ("head_government", "head_state")),
    (re.compile(r"\b(?:chief executive officer|\bceo\b)\b", re.I), ("ceo",)),
    (re.compile(r"\b(?:head coach|coach)\b", re.I), ("head_coach",)),
    (re.compile(r"\bheadquarters\b", re.I), ("headquarters_city",)),
    (re.compile(r"\b(?:original broadcaster|origianl broadcaster|broadcaster)\b", re.I), ("original_broadcaster",)),
    (re.compile(r"\b(?:child category|name of the child|whose child)\b", re.I), ("child",)),
    (re.compile(r"\b(?:author|written by|who wrote)\b", re.I), ("author",)),
    (re.compile(r"\b(?:director)\b", re.I), ("director",)),
    (re.compile(r"\b(?:chairperson|chairman|chairwoman)\b", re.I), ("chairperson",)),
    (re.compile(r"\b(?:performer|singer|performed by)\b", re.I), ("performed_by",)),
    (re.compile(r"\b(?:developer|developed by)\b", re.I), ("developed_by",)),
    (re.compile(r"\b(?:producer|produced by|manufacturer)\b", re.I), ("produced_by",)),
    (re.compile(r"\b(?:founder|founded by|who founded)\b", re.I), ("founded_by",)),
    (re.compile(r"\b(?:creator|created by)\b", re.I), ("created_by",)),
    (re.compile(r"\b(?:spouse|married|wife|husband)\b", re.I), ("married",)),
    (re.compile(r"\b(?:citizenship|nationality|citizen of)\b", re.I), ("citizen",)),
)


def _parse_statement(statement: str) -> tuple[str, str, str] | None:
    parsed = parse_v330_statement(statement)
    if parsed is not None:
        return parsed
    for relation, pattern in EXTRA_FACT_PATTERNS:
        match = pattern.match(statement)
        if match:
            subject, obj = (part.strip().rstrip(".") for part in match.groups())
            if subject and obj:
                return subject, relation, obj
    return None


def terminal_relation_weights(query: str) -> dict[str, float]:
    """Infer requested output relations without using an answer or labels."""
    for pattern, relations in OUTPUT_SLOT_RULES:
        if pattern.search(query):
            return {
                relation: 1.0 - 0.08 * position
                for position, relation in enumerate(relations)
            }
    # Preserve the 3.30 fallback for question forms outside the explicit
    # output grammar, but do not pretend its cue length is terminal syntax.
    return {relation: 0.50 for relation in target_v330_relations(query)}


@dataclass(frozen=True)
class EvidenceClosureResult:
    primary_sources: tuple[int, ...] = ()
    echo_sources: tuple[int, ...] = ()
    shadow_sources: tuple[int, ...] = ()
    dependency_sources: tuple[int, ...] = ()
    answer: str = ""
    path: tuple[tuple[str, str, str], ...] = ()
    terminal_relations: tuple[str, ...] = ()


class EvidenceClosureGraph:
    """Version-aware four-hop graph with an answer-echo closure orbit."""

    def __init__(self, texts: Sequence[str]) -> None:
        collected: dict[tuple[int, str], tuple[str, str, str, set[int]]] = {}
        for source, text in enumerate(texts):
            for raw_line in text.splitlines():
                numbered = NUMBERED_FACT_RE.match(raw_line)
                if not numbered:
                    continue
                ordinal = int(numbered.group(1))
                statement = numbered.group(2).strip()
                parsed = _parse_statement(statement)
                if parsed is None:
                    continue
                key = (ordinal, _norm(statement))
                current = collected.get(key)
                if current is None:
                    subject, relation, obj = parsed
                    collected[key] = (subject, relation, obj, {source})
                else:
                    current[3].add(source)
        self.facts = tuple(
            GraphFact(subject, relation, obj, ordinal, tuple(sorted(sources)))
            for (ordinal, _), (subject, relation, obj, sources) in sorted(collected.items())
        )
        self.texts = list(texts)
        self._normalized_texts = [_norm(text) for text in self.texts]
        self.enabled = len(self.facts) >= max(20, 2 * len(texts))
        self._subjects = sorted(
            {_norm(fact.subject) for fact in self.facts},
            key=lambda value: (-len(value), value),
        )

    def _current_adjacency(self, size: int) -> dict[str, list[GraphFact]]:
        current: dict[tuple[str, str], GraphFact] = {}
        for fact in self.facts:
            visible_sources = tuple(source for source in fact.sources if source < size)
            if not visible_sources:
                continue
            visible = GraphFact(
                fact.subject, fact.relation, fact.object, fact.ordinal, visible_sources,
            )
            key = (_norm(fact.subject), fact.relation)
            prior = current.get(key)
            if prior is None or visible.ordinal > prior.ordinal:
                current[key] = visible
        adjacency: dict[str, list[GraphFact]] = defaultdict(list)
        for (subject, _), fact in current.items():
            adjacency[subject].append(fact)
        return adjacency

    def solve(
        self, query: str, *, size: int, max_hops: int = 4,
    ) -> EvidenceClosureResult:
        if not self.enabled or size <= 0:
            return EvidenceClosureResult()
        terminal = terminal_relation_weights(query)
        if not terminal:
            return EvidenceClosureResult()
        query_norm = _norm(query)
        anchors = [subject for subject in self._subjects if subject in query_norm][:6]
        if not anchors:
            return EvidenceClosureResult()
        adjacency = self._current_adjacency(size)
        queue = deque((anchor, (), 0.0) for anchor in anchors)
        visited: dict[tuple[str, int], float] = {}
        solutions: list[tuple[float, int, int, GraphFact, tuple[GraphFact, ...]]] = []
        while queue:
            entity, path, score = queue.popleft()
            depth = len(path)
            if depth > max_hops:
                continue
            state = (entity, depth)
            if visited.get(state, float("-inf")) >= score:
                continue
            visited[state] = score
            for fact in adjacency.get(entity, ()):
                contact = _relation_contact(query, fact.relation)
                next_score = score + 0.20 + min(3.0, contact / 5.0)
                next_path = path + (fact,)
                terminal_mass = terminal.get(fact.relation)
                if terminal_mass is not None:
                    solutions.append((
                        next_score + 8.0 + terminal_mass,
                        -len(next_path), fact.ordinal, fact, next_path,
                    ))
                if len(next_path) < max_hops:
                    queue.append((_norm(fact.object), next_path, next_score))
        if not solutions:
            return EvidenceClosureResult(terminal_relations=tuple(terminal))
        _, _, _, answer_fact, best_path = max(solutions, key=lambda item: item[:3])
        answer_norm = _norm(answer_fact.object)
        echo = tuple(
            source for source, text in enumerate(self._normalized_texts[:size])
            if answer_norm and answer_norm in text
        )
        dependency = tuple(dict.fromkeys(
            source
            for fact in best_path
            for source in fact.sources
            if source < size and source not in answer_fact.sources
        ))
        return EvidenceClosureResult(
            primary_sources=answer_fact.sources,
            echo_sources=echo,
            dependency_sources=dependency,
            answer=answer_fact.object,
            path=tuple((fact.subject, fact.relation, fact.object) for fact in best_path),
            terminal_relations=tuple(terminal),
        )
